"""CPU counterfactual projections using saved, matched reconstruction outputs."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evaluation/scale_diagnostic'


def project(mesh, camera, key):
    points = mesh[key].copy().astype(np.float64)
    if key == 'joints_3d':
        points[:, 0] *= 2 * float(mesh['right']) - 1
    q = points + camera['cam_t_full']
    assert np.isfinite(q).all() and (q[:, 2] > 0).all()
    return q[:, :2] / q[:, 2:] * camera['focal_length'] + camera['img_size'] / 2


def extent(x):
    return float(np.linalg.norm(np.ptp(x, axis=0)))


def main():
    out = BASE / 'camera_swap'
    out.mkdir(exist_ok=True)
    manifest = json.loads((BASE/'fixed_inputs/manifest.json').read_text())
    rows = []
    checks = []
    for rec in manifest['samples']:
        sid = rec['id']
        data = {}
        for name in ['official_before', 'official_after', 'trained_before', 'trained_after']:
            model, stage = name.split('_')
            with np.load(BASE/'predictions'/model/f'{sid}_{stage}.npz') as z:
                data[name] = dict(z)
        ref = data['official_after']
        for d in data.values():
            for key in ['right', 'box_center', 'box_size', 'img_size', 'focal_length']:
                np.testing.assert_allclose(d[key], ref[key])
            error = np.max(np.abs(project(d, d, 'joints_3d') - d['joints_2d_full']))
            assert error < 1e-3, error
            checks.append(float(error))
        panels = []
        for comparison, a, b in [
            ('final', 'official_after', 'trained_after'),
            ('initial', 'official_before', 'trained_before'),
            ('trained_refinement', 'trained_before', 'trained_after')]:
            baseline = project(data[a], data[a], 'joints_3d')
            baseline_v = project(data[a], data[a], 'vertices_full')
            for geom, cam in [(a,a), (b,b), (b,a), (a,b)]:
                uv = project(data[geom], data[cam], 'joints_3d')
                vv = project(data[geom], data[cam], 'vertices_full')
                rows.append(dict(sample=sid, comparison=comparison, geometry=geom, camera=cam,
                    joint_extent_ratio=extent(uv)/extent(baseline),
                    vertex_extent_ratio=extent(vv)/extent(baseline_v),
                    center_shift_px=float(np.linalg.norm(uv.mean(0)-baseline.mean(0))),
                    joint_extent_px=extent(uv)))
                if comparison == 'final':
                    image = cv2.imread(str(ROOT/rec['image']))
                    # Fixed display window shared by all four counterfactuals.
                    center = np.array(rec['box_center'])
                    size = float(rec['box_size'])
                    scale = 360 / size
                    trans = np.array([[scale,0,180-scale*center[0]], [0,scale,180-scale*center[1]]])
                    panel = cv2.warpAffine(image, trans, (360,360))
                    for points,color in [(baseline,(160,160,160)), (uv,(0,220,0))]:
                        pts = points * scale + trans[:,2]
                        for x,y in pts:
                            if 0 <= x < 360 and 0 <= y < 360:
                                cv2.circle(panel,(round(x),round(y)),3,color,-1)
                    panel = cv2.copyMakeBorder(panel,45,0,0,0,cv2.BORDER_CONSTANT)
                    cv2.putText(panel, f'Mesh {geom.split("_")[0]} / Cam {cam.split("_")[0]}',
                                (5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    cv2.putText(panel, f'extent ratio {extent(uv)/extent(baseline):.3f}',
                                (5,37),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    panels.append(panel)
        cv2.imwrite(str(out/f'{sid}.jpg'), np.concatenate(panels,axis=1))
    with (out/'per_sample.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# Camera swap diagnostic', '',
        '45 fixed detector crops from 8 images; CPU reprojection only. No training.',
        'Geometry includes pose and shape. Official predictions are a reference, not ground truth.',
        f'Identity reprojection maximum absolute error: {max(checks):.9g} pixels.', '',
        '| Comparison | Geometry | Camera | Median joint extent ratio | Smaller / 45 | Median vertex extent ratio |',
        '|---|---|---|---:|---:|---:|']
    for key in dict.fromkeys((r['comparison'],r['geometry'],r['camera']) for r in rows):
        subset = [r for r in rows if (r['comparison'],r['geometry'],r['camera']) == key]
        ratios = np.array([r['joint_extent_ratio'] for r in subset])
        lines.append('| '+' | '.join(key)+f' | {np.median(ratios):.4f} | {sum(ratios<1)} | '+
                     f'{np.median([r["vertex_extent_ratio"] for r in subset]):.4f} |')
    lines += ['', 'Ratios use the first model/stage in each comparison as baseline: official after for final,',
        'official before for initial, trained before for trained_refinement.',
        'Visualizations: gray joints = official final reference, green = tested combination.',
        'The four panels share an identical unmirrored image window; no depth sorting or mesh rendering is applied.',
        'Hybrid predictions are interventions for diagnosis, not physically calibrated estimates.',
        'Medians are paired sample statistics and cannot be added to allocate causal percentages.',
        'Eight images contain correlated detections and possible false positives; these are descriptive results.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
