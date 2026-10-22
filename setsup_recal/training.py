"""Stage-II training and evaluation functions retained from the study."""
import hashlib,json,math,statistics,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .core import compute_stage2_objective
from .data import RaterCountBatchSampler,collate_multirater_samples
from .metrics import METRIC_NAMES,aggregate_metric_rows,evaluate_distribution_batch
from .fingerprint import state_dict_sha256
from .augmentation import _augment

def _uses_graph(arm):
    return arm != "dpersona_stage2"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _make_loader(dataset, batch_size, seed, shuffle, num_workers):
    sampler = RaterCountBatchSampler(
        [len(record["rater_ids"]) for record in dataset.records],
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        collate_fn=collate_multirater_samples,
    )


def _build_model(
    model_factory,
    *,
    source_root,
    stage1_checkpoint,
    relation_prior_path,
    dataset,
    device,
):
    return model_factory(
        source_root=source_root,
        stage1_checkpoint=stage1_checkpoint,
        relation_prior_path=relation_prior_path,
        input_channels=dataset.manifest["image_channels"],
        rater_count=len(dataset.records[0]["rater_ids"]),
        device=device,
    )


def _persona_dice(probabilities, targets, eps=1e-6):
    predicted = (probabilities >= 0.5).to(torch.float32).flatten(start_dim=2)
    targets = (targets >= 0.5).to(torch.float32).flatten(start_dim=2)
    intersection = (predicted * targets).sum(dim=2)
    denominator = predicted.sum(dim=2) + targets.sum(dim=2)
    return (2.0 * intersection + eps) / (denominator + eps)


def evaluate_persona_loader(
    model,
    loader,
    *,
    arm,
    device,
    seed,
    candidate_samples,
):
    was_training = model.training
    model.eval()
    rows = []
    rater_values = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["masks"].to(device)
            probabilities = model.sample_personas(
                images,
                candidate_samples=candidate_samples,
                generator=torch.Generator(device=device).manual_seed(
                    seed + batch_index
                ),
                use_graph=_uses_graph(arm),
            )
            metrics = evaluate_distribution_batch(probabilities, masks)
            persona_dice = _persona_dice(probabilities, masks)
            rater_values.append(persona_dice.cpu())
            for item_index, (patient_id, sample_id) in enumerate(
                zip(batch["patient_id"], batch["sample_id"])
            ):
                row = {"patient_id": patient_id, "sample_id": sample_id}
                for metric in METRIC_NAMES:
                    row[metric] = float(metrics[metric][item_index].cpu())
                row["per_rater_dice"] = persona_dice[item_index].cpu().tolist()
                rows.append(row)
    if was_training:
        model.train()
    aggregate = aggregate_metric_rows(rows)
    per_rater = torch.cat(rater_values).mean(dim=0).tolist()
    return {
        "image_macro": aggregate["sample_macro"],
        "per_rater_dice": per_rater,
        "mean_per_rater_dice": statistics.fmean(per_rater),
        "rows": rows,
    }


def _compute_objective(
    model,
    images,
    masks,
    *,
    arm,
    candidate_samples,
    center_weight,
    relation_weight,
    generator,
):
    logits = model.persona_logits(
        images,
        candidate_samples=candidate_samples,
        generator=generator,
        use_graph=_uses_graph(arm),
    )
    return compute_stage2_objective(
        logits,
        masks,
        arm=arm,
        center_weight=center_weight,
        relation_weight=relation_weight,
    )


def _train_epoch(
    *,
    model,
    loader,
    optimizer,
    arm,
    epoch,
    seed,
    candidate_samples,
    center_weight,
    relation_weight,
    device,
    batch_augmenter=None,
):
    model.train()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    totals = {key: 0.0 for key in ("total", "base", "center", "relation")}
    steps = 0
    for batch_index, batch in enumerate(loader):
        batch_seed = seed + epoch * 1_000_000 + batch_index
        images = batch["image"].to(device)
        masks = batch["masks"].to(device)
        augmenter = batch_augmenter or _augment
        images, masks = augmenter(images, masks, batch_seed)
        optimizer.zero_grad(set_to_none=True)
        report = _compute_objective(
            model,
            images,
            masks,
            arm=arm,
            candidate_samples=candidate_samples,
            center_weight=center_weight,
            relation_weight=relation_weight,
            generator=torch.Generator(device=device).manual_seed(
                seed + 20_000_000 + epoch * 100_000 + batch_index
            ),
        )
        if not bool(torch.isfinite(report["total"])):
            raise FloatingPointError(
                f"non-finite Stage-II loss at epoch {epoch}, batch {batch_index}"
            )
        report["total"].backward()
        gradients = [
            parameter.grad
            for parameter in model.trainable_parameters(arm)
            if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ):
            raise FloatingPointError(
                f"non-finite Stage-II gradient at epoch {epoch}, batch {batch_index}"
            )
        optimizer.step()
        steps += 1
        for key in totals:
            totals[key] += float(report[key].detach())
    return {
        "optimizer_steps": steps,
        "means": {key: value / steps for key, value in totals.items()},
    }


def _run_arm(
    *,
    arm,
    output_dir,
    model_factory,
    source_root,
    stage1_checkpoint,
    relation_prior_path,
    dataset,
    train_loader,
    validation_loader,
    initial_state,
    center_weight,
    relation_weight,
    seed,
    epochs,
    learning_rate,
    candidate_samples,
    device,
    hashes,
    validation_evaluator=None,
    selection_scope="image_macro",
    batch_augmenter=None,
):
    output_dir.mkdir()
    model = _build_model(
        model_factory,
        source_root=source_root,
        stage1_checkpoint=stage1_checkpoint,
        relation_prior_path=relation_prior_path,
        dataset=dataset,
        device=device,
    )
    model.load_state_dict(initial_state)
    initial_hash = state_dict_sha256(model)
    parameters = tuple(model.trainable_parameters(arm))
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if selection_scope == "image_macro":
        best_metrics_path = output_dir / "best_validation_image_metrics.json"
        best_rows_path = output_dir / "best_validation_per_image.jsonl"
    elif selection_scope == "source_macro":
        best_metrics_path = output_dir / "best_validation_source_metrics.json"
        best_rows_path = output_dir / "best_validation_per_view.jsonl"
    elif selection_scope == "patient_macro":
        best_metrics_path = output_dir / "best_validation_patient_metrics.json"
        best_rows_path = output_dir / "best_validation_per_patient.jsonl"
    else:
        raise ValueError("selection scope must be image_macro, source_macro or patient_macro")
    config = {
        "schema": "riga_rg_persona_arm_config/1",
        "arm": arm,
        "task_id": dataset.manifest["task_id"],
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "candidate_samples": candidate_samples,
        "output_masks": len(dataset.records[0]["rater_ids"]),
        "center_weight": center_weight,
        "relation_weight": relation_weight,
        "selection_metric": f"validation.{selection_scope}.ged",
        "hashes": hashes,
        "test_opened": False,
    }
    _write_json(output_dir / "config.json", config)
    best_ged = math.inf
    best_epoch = None
    best_per_rater = None
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        training = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            arm=arm,
            epoch=epoch,
            seed=seed,
            candidate_samples=candidate_samples,
            center_weight=center_weight,
            relation_weight=relation_weight,
            device=device,
            batch_augmenter=batch_augmenter,
        )
        optimizer_steps += training["optimizer_steps"]
        evaluator = validation_evaluator or evaluate_persona_loader
        validation = evaluator(
            model,
            validation_loader,
            arm=arm,
            device=device,
            seed=seed + 90_000_000,
            candidate_samples=candidate_samples,
        )
        selection_metrics = validation[selection_scope]
        row = {
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": training["means"],
            f"validation_{selection_scope}": selection_metrics,
            "validation_mean_per_rater_dice": validation["mean_per_rater_dice"],
            "validation_per_rater_dice": validation["per_rater_dice"],
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        with (output_dir / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        checkpoint = {
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "validation": validation,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        ged = selection_metrics["ged"]
        if ged < best_ged:
            best_ged = ged
            best_epoch = epoch + 1
            best_per_rater = validation["per_rater_dice"]
            torch.save(checkpoint, output_dir / "best.pt")
            _write_json(best_metrics_path, selection_metrics)
            _write_jsonl(best_rows_path, validation["rows"])
        scheduler.step()
    summary = {
        "schema": "riga_rg_persona_arm_summary/1",
        "status": "COMPLETE",
        "arm": arm,
        "completed_epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "selection_scope": selection_scope,
        f"best_validation_{selection_scope}": json.loads(best_metrics_path.read_text()),
        "best_validation_per_rater_dice": best_per_rater,
        "best_validation_mean_per_rater_dice": statistics.fmean(best_per_rater),
        "initial_parameter_sha256": initial_hash,
        "best_checkpoint_sha256": _sha256(output_dir / "best.pt"),
        "last_checkpoint_sha256": _sha256(output_dir / "last.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "test_opened": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
