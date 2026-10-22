"""Run the original RIGA contour conversion and cache construction."""
import argparse,json
from pathlib import Path
from riga_q2_mask_converter import materialize_development
from riga_q2_dev_cache import build_riga_development_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--mask-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path('protocol'))
    args=parser.parse_args()
    inventory=json.loads((args.protocol/'RIGA_INVENTORY.json').read_text())
    materialize_development(args.raw_root,args.mask_root,
        expected_by_leaf=inventory['groups_by_leaf'],
        expected_incomplete_count=inventory['incomplete_groups'])
    receipt=build_riga_development_cache(raw_root=args.raw_root,mask_root=args.mask_root,
        output_root=args.output_root,expected_groups_by_leaf=inventory['groups_by_leaf'],
        validation_fraction=.2,roi_size=512,output_size=256)
    observed=json.loads((args.output_root/'split.json').read_text())
    expected=json.loads((args.protocol/'RIGA_SPLIT.json').read_text())
    if observed['records']!=expected['records']:
        raise ValueError('Generated RIGA split differs from the published ordered split')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
