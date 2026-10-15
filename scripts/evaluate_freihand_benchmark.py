"""Official FreiHAND split, upstream eval metadata and official scoring functions.

WiLoR publishes no benchmark entrypoint. The input path follows HaMeR eval.py:
rescale_factor=-1, official freihand_val.npz, ViT BBOX_SHAPE=[192,256], no augment.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import open_dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_freihand import build_cfg, load_model, move
from wilor.datasets.image_dataset import ImageDataset
from wilor.utils.official_freihand_eval import evaluate_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',type=Path,default=Path('evaluation_data/FreiHAND_pub_v2_eval'))
    p.add_argument('--metadata',default='evaluation_data/freihand_val.npz')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    cfg=build_cfg(args.batch_size,args.workers)
    with open_dict(cfg):
        cfg.MODEL.BBOX_SHAPE=[192,256]
    dataset=ImageDataset(cfg,args.metadata,str(args.data_root/'evaluation/rgb'),train=False,rescale_factor=-1)
    if len(dataset)!=3960:
        raise ValueError('Expected the 3960-image official evaluation split')
    names=[str(x.decode() if isinstance(x,bytes) else x) for x in dataset.imgname]
    if [int(Path(x).stem) for x in names]!=list(range(3960)):
        raise ValueError('Metadata must follow official sample order')
    model=load_model(cfg,args.checkpoint,args.device)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    joints,verts=[],[]
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            out=model.forward_step(move(batch,args.device),train=False)
            joints.append(out['pred_keypoints_3d'].cpu().numpy())
            verts.append(out['pred_vertices'].cpu().numpy())
            if i%20==0:print(f'Batches {i+1}/{len(loader)}',flush=True)
    pj,pv=np.concatenate(joints),np.concatenate(verts)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'pred.json').write_text(json.dumps([pj.tolist(),pv.tolist()]))
    gj=json.loads((args.data_root/'evaluation_xyz.json').read_text())
    gv=json.loads((args.data_root/'evaluation_verts.json').read_text())
    result=evaluate_predictions(pj,pv,gj,gv)
    result.update(checkpoint=args.checkpoint,protocol='FreiHAND official 3960 evaluation samples',
                  input_processing='HaMeR official eval metadata; WiLoR crop tools; BBOX_SHAPE=[192,256]',
                  caveat='WiLoR does not publish a benchmark entrypoint; exact original eval preprocessing is not confirmed')
    (args.output_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
