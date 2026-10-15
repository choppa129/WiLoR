"""Run pinned, unmodified FreiHAND metric definitions with a modern distance backend.

The upstream scorer uses an obsolete Open3D API. SciPy cKDTree supplies the same
Euclidean nearest-neighbor distances; alignment, F-score and EvalUtil are loaded
verbatim from the vendored official source, without its pip-install/HTML side effects.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree


class _PointCloud:
    def paint_uniform_color(self, color):
        pass


def _distance(a, b):
    return cKDTree(np.asarray(b.points)).query(np.asarray(a.points))[0]


def load_official_metrics():
    root = Path(__file__).resolve().parents[2] / 'third-party/freihand_eval'
    namespace = {'np': np, 'orthogonal_procrustes': orthogonal_procrustes,
                 'o3d': SimpleNamespace(PointCloud=_PointCloud, Vector3dVector=np.asarray,
                                       compute_point_cloud_to_point_cloud_distance=_distance)}
    for filename, names in [('eval.py', {'verts2pcd', 'calculate_fscore', 'align_w_scale'}),
                            ('eval_util.py', {'EvalUtil'})]:
        tree = ast.parse((root / filename).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if len(nodes) != len(names):
            raise RuntimeError('Official metric snapshot is incomplete')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root/filename), 'exec'), namespace)
    return namespace


def evaluate_predictions(pred_joints, pred_vertices, gt_joints, gt_vertices):
    arrays = [np.asarray(x, dtype=np.float64) for x in (pred_joints, pred_vertices, gt_joints, gt_vertices)]
    pj, pv, gj, gv = arrays
    if pj.shape != gj.shape or pv.shape != gv.shape or pj.shape[1:] != (21, 3) or pv.shape[1:] != (778, 3):
        raise ValueError('Expected matching [N,21,3] joints and [N,778,3] vertices in meters')
    if len(pj) != len(pv) or not len(pj) or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Predictions and labels must be nonempty, finite and equally sized')
    official = load_official_metrics()
    joints, vertices = official['EvalUtil'](), official['EvalUtil'](num_kp=778)
    scores = []
    for a, b, c, d in zip(pj, pv, gj, gv):
        aj = official['align_w_scale'](c, a)
        av = official['align_w_scale'](d, b)
        joints.feed(c, np.ones(21), aj)
        vertices.feed(d, np.ones(778), av)
        scores.append([official['calculate_fscore'](d, av, t)[0] for t in (.005, .015)])
    jm = joints.get_measures(0., .05, 100)
    vm = vertices.get_measures(0., .05, 100)
    fs = np.mean(scores, axis=0)
    return {'samples': len(pj), 'pa_mpjpe_mm': float(jm[0]*1000),
            'pa_mpvpe_mm': float(vm[0]*1000), 'f_at_5mm': float(fs[0]),
            'f_at_15mm': float(fs[1]), 'auc_j': float(jm[2]), 'auc_v': float(vm[2])}
