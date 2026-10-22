# Initial RIGA split

This directory supplies the 131-view RIGA development split used by the component ablations, training repeats, and bootstrap analyses. Four development views share a raw-image source with training. These experiments average view-level metrics.

RIGA cup and disc share the 524 training and 131 development views. REFUGE and NPC settings match `../protocol/`. Configuration files, ordered records, and checkpoint identifiers accompany the split.

Use `--protocol protocol_initial` with the commands in [REPRODUCE.md](../REPRODUCE.md). The main RIGA comparison uses `--protocol protocol`. Input preparation is described in [PREPROCESSING.md](../PREPROCESSING.md).
