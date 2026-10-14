"""Swap orientation and articulation, holding trained shape and camera fixed."""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wilor.models.mano_wrapper import MANO
from swap_scale_cameras import project, extent

BASE = ROOT / 'evaluation/scale_diagnostic'


def main():
    torch.set_num_threads(4)
    mano = MANO(model_path=str(ROOT/'mano_data'), gender='neutral',
                num_hand_joints=15, create_body_pose=False).eval()
    out = BASE/'pose_ablation'
    out.mkdir(exist_ok=True)
    records = json.loads((BASE/'fixed_inputs/manifest.json').read_text())['samples']
    assert len(records) == 45
    rows, errors, projection_errors = [], [], []
    variants = ['baseline', 'official_orientation', 'official_fingers', 'official_both']
    for rec in records:
        sid = rec['id']
        def load(model):
            with np.load(BASE/'predictions'/model/f'{sid}_after.npz') as z:
                return dict(z)
        trained, official = load('trained'), load('official')
        for key in ['right','box_center','box_size','img_size','focal_length']:
            np.testing.assert_allclose(trained[key], official[key])
        image = cv2.imread(str(ROOT/rec['image']))
        assert image is not None
        panels = []
        for variant in variants:
            orient = official['global_orient'] if variant in ['official_orientation','official_both'] else trained['global_orient']
            pose = official['hand_pose'] if variant in ['official_fingers','official_both'] else trained['hand_pose']
            with torch.inference_mode():
                result = mano(global_orient=torch.from_numpy(orient).unsqueeze(0),
                    hand_pose=torch.from_numpy(pose).unsqueeze(0),
                    betas=torch.from_numpy(trained['betas']).unsqueeze(0), pose2rot=False)
            verts, joints = result.vertices[0].numpy(), result.joints[0].numpy()
            assert np.isfinite(verts).all() and np.isfinite(joints).all()
            uv = project(dict(trained,joints_3d=joints), trained, 'joints_3d')
            mirrored = verts.copy(); mirrored[:,0] *= 2*float(trained['right'])-1
            vv = project(dict(trained,vertices_full=mirrored), trained, 'vertices_full')
            if variant == 'baseline':
                err = max(float(np.max(np.abs(verts-trained['vertices']))), float(np.max(np.abs(joints-trained['joints_3d']))))
                errors.append(err)
                projection_errors.append(float(np.max(np.abs(uv-trained['joints_2d_full']))))
                assert err < 1e-5 and projection_errors[-1] < .001
            rows.append(dict(sample=sid,variant=variant,joint_extent_px=extent(uv),
                vertex_extent_px=extent(vv),joint_ratio_to_trained=extent(uv)/extent(trained['joints_2d_full']),
                joint_ratio_to_official=extent(uv)/extent(official['joints_2d_full']),
                center_shift_from_trained_px=float(np.linalg.norm(uv.mean(0)-trained['joints_2d_full'].mean(0)))))
            np.savez_compressed(out/f'{sid}_{variant}.npz', vertices=verts,joints_3d=joints,
                joints_2d_full=uv,vertices_2d_full=vv,betas=trained['betas'],
                global_orient=orient,hand_pose=pose,cam_t_full=trained['cam_t_full'],
                focal_length=trained['focal_length'],img_size=trained['img_size'],right=trained['right'])
            scale = 360/rec['box_size']; center = np.array(rec['box_center'])
            trans = np.array([[scale,0,180-scale*center[0]],[0,scale,180-scale*center[1]]])
            panel = cv2.warpAffine(image,trans,(360,360))
            for points,color in [(trained['joints_2d_full'],(160,160,160)),(uv,(0,220,0))]:
                points = points*scale+trans[:,2]
                for start in [1,5,9,13,17]:
                    for a,b in zip([0,start,start+1,start+2],[start,start+1,start+2,start+3]):
                        aa,bb = points[a],points[b]
                        if np.max(np.abs([aa,bb])) < 10000:
                            cv2.line(panel,tuple(np.rint(aa).astype(int)),tuple(np.rint(bb).astype(int)),color,1)
                for x,y in points:
                    if 0<=x<360 and 0<=y<360: cv2.circle(panel,(round(x),round(y)),3,color,-1)
            panel = cv2.copyMakeBorder(panel,48,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(panel,variant,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(panel,f'extent {extent(uv):.1f}px; ratio {rows[-1]["joint_ratio_to_trained"]:.3f}',
                        (5,39),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            panels.append(panel)
        assert cv2.imwrite(str(out/f'{sid}.jpg'),np.concatenate(panels,axis=1))
    with (out/'per_sample.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    baseline={r['sample']:r for r in rows if r['variant']=='baseline'}
    lines=['# Pose ablation', '', '45 fixed crops; trained final betas and camera held fixed. CPU MANO.',
        f'Max baseline reproduction error: {max(errors):.9g} m; projection: {max(projection_errors):.9g} px.', '',
        '| Variant | Median extent / trained | P10 | P90 | Median extent / official | Closer to official extent /45 | Median vertex extent / trained |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for variant in variants:
        subset=[r for r in rows if r['variant']==variant]
        ratios=[r['joint_ratio_to_trained'] for r in subset]
        q=np.percentile(ratios,[10,50,90])
        closer=sum(abs(r['joint_ratio_to_official']-1)<abs(baseline[r['sample']]['joint_ratio_to_official']-1)-1e-6 for r in subset)
        lines.append(f'| {variant} | {q[1]:.4f} | {q[0]:.4f} | {q[2]:.4f} | '+
            f'{np.median([r["joint_ratio_to_official"] for r in subset]):.4f} | {closer} | '+
            f'{np.median([r["vertex_extent_px"]/baseline[r["sample"]]["vertex_extent_px"] for r in subset]):.4f} |')
    lines += ['', 'Extent = diagonal of the axis-aligned projected joint/vertex bounding box.',
        'Orientation swaps the global rotation; fingers swaps the 15 local MANO joint rotations.',
        'Gray skeleton = trained baseline; green = intervention. Identical image window in all panels.',
        'Official predictions are not ground truth; extent agreement does not establish pose accuracy.',
        'Ratios are paired descriptive medians, not additive causal shares. Eight images may contain correlated or false detections.',
        '180 archives, 45 four-panel images; no model training or checkpoint modification.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
