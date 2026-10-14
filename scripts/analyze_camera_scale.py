"""Compare camera, intrinsic mesh size, and projected size across model outputs."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


MODELS = {
    "official_wilor": "demo_out",
    "trained_wilor": "demo_out_trained",
    "official_hamer": "demo_out_hamer",
}
FIELDS = ["cam_scale", "cam_depth", "cam_x", "cam_y", "mesh_diag_m",
          "mesh_width_m", "mesh_height_m", "joint_diag_m", "joint_2d_diag_px",
          "joint_2d_width_px", "joint_2d_height_px", "beta_norm"]


def measurements(path):
    data = np.load(path)
    vertices = data["vertices"].astype(np.float64)
    joints = data["joints_3d"].astype(np.float64)
    joints_2d = data["joints_2d"].astype(np.float64)
    mesh_extent = np.ptp(vertices, axis=0)
    joint_extent = np.ptp(joints, axis=0)
    projection_extent = np.ptp(joints_2d, axis=0)
    camera = data["pred_cam"].astype(np.float64)
    translation = data["cam_t"].astype(np.float64)
    return {
        "center_2d": joints_2d.mean(axis=0),
        "cam_scale": camera[0], "cam_depth": translation[2],
        "cam_x": translation[0], "cam_y": translation[1],
        "mesh_diag_m": np.linalg.norm(mesh_extent),
        "mesh_width_m": mesh_extent[0], "mesh_height_m": mesh_extent[1],
        "joint_diag_m": np.linalg.norm(joint_extent),
        "joint_2d_diag_px": np.linalg.norm(projection_extent),
        "joint_2d_width_px": projection_extent[0],
        "joint_2d_height_px": projection_extent[1],
        "beta_norm": np.linalg.norm(data["mano_shape"].astype(np.float64)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="evaluation/camera_scale_comparison")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    files = {}
    for model, directory in MODELS.items():
        grouped = {}
        for path in (root / directory).glob("*_params.npz"):
            key = path.name.removesuffix("_params.npz")
            image_name = key.rsplit("_", 1)[0]
            grouped.setdefault(image_name, []).append((key, path, measurements(path)))
        files[model] = grouped

    matched = []
    for image_name in sorted(files["official_wilor"]):
        reference = sorted(files["official_wilor"][image_name])
        mappings = {"official_wilor": {item[0]: item for item in reference}}
        reference_centers = np.stack([item[2]["center_2d"] for item in reference])
        for model in ("trained_wilor", "official_hamer"):
            candidates = sorted(files[model][image_name])
            candidate_centers = np.stack([item[2]["center_2d"] for item in candidates])
            cost = np.linalg.norm(reference_centers[:, None] - candidate_centers[None, :], axis=-1)
            rows_idx, cols_idx = linear_sum_assignment(cost)
            mappings[model] = {reference[row][0]: candidates[col] for row, col in zip(rows_idx, cols_idx)}
        for reference_key, _, _ in reference:
            matched.append((reference_key, {model: mappings[model][reference_key] for model in MODELS}))

    keys = [item[0] for item in matched]
    rows = []
    by_model = {name: {field: [] for field in FIELDS} for name in MODELS}
    paired = {field: [] for field in FIELDS}
    for key, match in matched:
        values = {}
        for model in MODELS:
            source_key, _, values[model] = match[model]
            row = {"sample": key, "source_sample": source_key, "model": model,
                   **{field: values[model][field] for field in FIELDS}}
            rows.append(row)
            for field in FIELDS:
                by_model[model][field].append(values[model][field])
        for field in FIELDS:
            paired[field].append(values["official_wilor"][field] / values["official_hamer"][field])

    with (output / "per_hand.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "source_sample", "model", *FIELDS])
        writer.writeheader(); writer.writerows(rows)

    focus = ["cam_scale", "cam_depth", "mesh_diag_m", "joint_diag_m", "joint_2d_diag_px", "beta_norm"]
    lines = ["# Camera and scale comparison", "", f"Matched hands: {len(keys)}", "",
             "| Quantity | Official WiLoR mean | Trained WiLoR mean | Official HaMeR mean | WiLoR/HaMeR median ratio |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for field in focus:
        means = [np.mean(by_model[name][field]) for name in MODELS]
        lines.append(f"| {field} | {means[0]:.6f} | {means[1]:.6f} | {means[2]:.6f} | {np.median(paired[field]):.4f} |")
    w_proj = np.asarray(by_model["official_wilor"]["joint_2d_diag_px"])
    h_proj = np.asarray(by_model["official_hamer"]["joint_2d_diag_px"])
    w_mesh = np.asarray(by_model["official_wilor"]["mesh_diag_m"])
    h_mesh = np.asarray(by_model["official_hamer"]["mesh_diag_m"])
    t_proj = np.asarray(by_model["trained_wilor"]["joint_2d_diag_px"])
    t_mesh = np.asarray(by_model["trained_wilor"]["mesh_diag_m"])
    t_scale = np.asarray(by_model["trained_wilor"]["cam_scale"])
    t_depth = np.asarray(by_model["trained_wilor"]["cam_depth"])
    w_scale = np.asarray(by_model["official_wilor"]["cam_scale"])
    w_depth = np.asarray(by_model["official_wilor"]["cam_depth"])
    lines += ["", f"Official WiLoR has a smaller projected 2D joint diagonal in {(w_proj < h_proj).mean()*100:.1f}% of matched hands.",
              f"Official WiLoR has a smaller intrinsic mesh diagonal in {(w_mesh < h_mesh).mean()*100:.1f}% of matched hands.", ""]
    lines += ["## Trained WiLoR versus official WiLoR", "",
              f"- Median projected-joint-size ratio: {np.median(t_proj/w_proj):.4f}",
              f"- Trained projection is smaller in {(t_proj < w_proj).mean()*100:.1f}% of hands.",
              f"- Median intrinsic-mesh-size ratio: {np.median(t_mesh/w_mesh):.4f}",
              f"- Median camera-scale ratio: {np.median(t_scale/w_scale):.4f}",
              f"- Median camera-depth ratio: {np.median(t_depth/w_depth):.4f}", ""]
    (output / "report.md").write_text("\n".join(lines))
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
