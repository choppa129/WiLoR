"""Read-only model/code audit; writes diagnostic artifacts, not model changes."""
import json
import csv
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wilor.utils.renderer import cam_crop_to_full
BASE=ROOT/'evaluation/scale_diagnostic'


def main():
    out=BASE/'projection_audit';out.mkdir(exist_ok=True)
    records=json.loads((BASE/'fixed_inputs/manifest.json').read_text())['samples']
    rows=[]
    for rec in records:
        with np.load(BASE/'fixed_inputs'/rec['tensor_file']) as z: fixed=dict(z)
        points=np.array([[0.,0.,1.],[128.,128.,1.],[256.,256.,1.]])
        back=points@fixed['crop_to_original'].T@fixed['original_to_crop'].T
        assert np.max(abs(points-back))<1e-6
        for model,stage in [('official','before'),('official','after'),('trained','before'),('trained','after'),('hamer','final')]:
            with np.load(BASE/'predictions'/model/f"{rec['id']}_{stage}.npz") as z:d=dict(z)
            cam=d['pred_cam'].copy();cam[1]*=2*float(d['right'])-1
            t=cam_crop_to_full(torch.tensor(cam[None],dtype=torch.float64),
                torch.tensor(d['box_center'][None],dtype=torch.float64),
                torch.tensor([float(d['box_size'])],dtype=torch.float64),
                torch.tensor(d['img_size'][None],dtype=torch.float64),float(d['focal_length'])).numpy()[0]
            assert np.max(abs(t-d['cam_t_full']))<1e-5
            xyz=d['joints_3d'].astype(float).copy()
            s,tx,ty=d['pred_cam'].astype(float)
            crop_t=np.array([tx,ty,2*5000/(256*s+1e-9)])
            # Compare direct crop inverse with full-image perspective conversion.
            def compare(flat):
                p=xyz.copy()
                if flat:p[:,2]=0
                q=p+crop_t
                uv=q[:,:2]/q[:,2:]*5000+128
                inverse=np.c_[uv,np.ones(len(uv))]@fixed['crop_to_original'].T
                p[:,0]*=2*float(d['right'])-1
                q=p+t
                full=q[:,:2]/q[:,2:]*d['focal_length']+d['img_size']/2
                return float(np.max(np.linalg.norm(inverse[:,:2]-full,axis=1)))
            flat=compare(True)
            assert flat<.01
            # OpenCV xyz -> OpenGL (x,-y,-z); top-left viewport agrees analytically.
            q=d['vertices_full']+t
            gl=q*np.array([1,-1,-1])
            uv_gl=np.c_[-gl[:,0]/gl[:,2],gl[:,1]/gl[:,2]]*d['focal_length']+d['img_size']/2
            uv_cv=q[:,:2]/q[:,2:]*d['focal_length']+d['img_size']/2
            assert np.max(abs(uv_gl-uv_cv))<1e-9
            rows.append(dict(sample=rec['id'],model=model,stage=stage,right=int(d['right']),
                camera_error=float(np.max(abs(t-d['cam_t_full']))),
                planar_crop_full_error_px=flat,nonplanar_crop_full_max_error_px=compare(False)))
    with (out/'per_sample.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Projection and coordinate audit','',
        'Scope: 225 saved predictions on 45 fixed crops. No training or inference code changed.',
        '', '## Confirmed reporting bug',
        'Dataset utils.py normalizes xy as pixel / patch_width - 0.5. Model projection divides focal length by IMAGE_SIZE.',
        'compute_validation_metrics uses IMAGE_SIZE / 2 for the pixel error; it must use IMAGE_SIZE for this convention.',
        'Therefore the historical reprojection metric is half the correct crop-pixel value. Training loss uses normalized xy directly and is unaffected.',
        'Historical values doubled (arithmetic correction, not a fresh evaluation): trained 6.743328 px; official 3.192446 px; initial 172.234450 px.',
        '3D metrics and relative reprojection rankings are unchanged. Historical CSV and PPT are preserved.',
        '', '## Checked coordinate paths',
        f'Actual cam_crop_to_full vs archived translation: max error {max(r["camera_error"] for r in rows):.9g}.',
        f'Zero-depth points: inverse crop vs full projection max error {max(r["planar_crop_full_error_px"] for r in rows):.9g} px.',
        'Crop affine round trips pass <1e-6; both left and right proposals are included.',
        'Left mirroring negates vertex X and camera tx once. Image-flip width-1 convention is included in saved transforms.',
        'Renderer vertices_to_trimesh applies Rx(180 degrees); its OpenGL projection agrees analytically with OpenCV coordinates.',
        'No offscreen rasterization/pixel-mask test was performed; this is a formula and numerical-coordinate audit.',
        '', '## Crop-to-full conversion is approximate for nonzero local Z',
        'Replacing crop focal/depth with full-image focal/depth preserves the weak-perspective planar case.',
        'For a 3D hand, inverse-warping crop perspective projections need not equal full-camera perspective projections.',
        'The observed difference is not by itself a renderer bug; it also exists in the official conversion convention.',
        '| Model/stage | Median of per-hand maximum discrepancy (px) | Maximum (px) |',
        '|---|---:|---:|']
    for model,stage in dict.fromkeys((r['model'],r['stage']) for r in rows):
        v=[r['nonplanar_crop_full_max_error_px'] for r in rows if r['model']==model and r['stage']==stage]
        lines.append(f'| {model}/{stage} | {np.median(v):.4f} | {max(v):.4f} |')
    lines+=['','Demo uses img_size.max() across a batch; current demo batches belong to one image, so this does not mix resolutions.',
        'The demo mutates pred_cam tx for left hands before saving legacy NPZ; fixed diagnostic exports preserve canonical camera separately.',
        'Next: correct metric scaling, regenerate labeled reports, then compare with ground-truth annotations. Do not change camera conversion based only on reference agreement.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
