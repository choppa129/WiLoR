"""Compare trained and official WiLoR checkpoints on full FreiHAND validation."""
import argparse, csv, gc, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule
from wilor.models import WiLoR
from wilor.utils.mesh_renderer import MeshRenderer
from wilor.utils.render_openpose import render_openpose

METRICS = ["val/mpjpe_mm", "val/pa_mpjpe_mm", "val/pve_mm", "val/reprojection_px"]
LABELS = {"val/mpjpe_mm":"MPJPE (mm)", "val/pa_mpjpe_mm":"PA-MPJPE (mm)",
          "val/pve_mm":"PVE (mm)", "val/reprojection_px":"Reprojection (px)"}

def move(value, device):
    if torch.is_tensor(value): return value.to(device, non_blocking=True)
    if isinstance(value, dict): return {k: move(v, device) for k, v in value.items()}
    return value

def build_cfg(batch_size, workers):
    config_dir = str((Path(__file__).resolve().parents[1] / "configs_hydra").resolve())
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        return compose(config_name="train.yaml", overrides=["data=freihand_full",
            "experiment=wilor_freihand_full", "MODEL.BACKBONE.PRETRAINED_WEIGHTS=null",
            f"TRAIN.BATCH_SIZE={batch_size}", f"GENERAL.NUM_WORKERS={workers}"])

def make_loader(cfg):
    data = WiLoRDataModule(cfg, dataset_config(cfg.dataset_config)); data.setup()
    return torch.utils.data.DataLoader(data.val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                       drop_last=False, num_workers=cfg.GENERAL.NUM_WORKERS)

def load_model(cfg, path, device):
    model = WiLoR(cfg, init_renderer=False)
    if path is not None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"[{path}] missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}")
    return model.to(device).eval()

def denormalize(images, cfg):
    mean = torch.tensor(cfg.MODEL.IMAGE_MEAN).view(1,3,1,1)
    std = torch.tensor(cfg.MODEL.IMAGE_STD).view(1,3,1,1)
    return (images.cpu() * std + mean).clamp(0,1)

def panel(rgb, text):
    image = cv2.cvtColor(np.clip(rgb*255,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0,0), (image.shape[1],28), (0,0,0), -1)
    cv2.putText(image, text, (7,19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
    return image

def save_visuals(renderer, cpu_batch, output, name, ordinals, seen, out_dir, cfg):
    images = denormalize(cpu_batch["img"], cfg).permute(0,2,3,1).numpy()
    verts = output["pred_vertices"].detach().float().cpu().numpy()
    cams = output["pred_cam_t"].detach().float().cpu().numpy()
    pred = output["pred_keypoints_2d"].detach().float().cpu().numpy()
    gt = cpu_batch["keypoints_2d"].float().numpy(); selected = set(ordinals)
    for i, image in enumerate(images):
        ordinal = seen+i
        if ordinal not in selected: continue
        pred_kps = np.c_[pred[i], np.ones(pred.shape[1])]
        pred_kps[:,:2] = cfg.MODEL.IMAGE_SIZE * (pred_kps[:,:2] + .5)
        gt_kps = gt[i].copy(); gt_kps[:,:2] = cfg.MODEL.IMAGE_SIZE * (gt_kps[:,:2] + .5)
        overlay = render_openpose(255*image.copy(), gt_kps)/255
        overlay = render_openpose(255*overlay, pred_kps)/255
        front = renderer(verts[i], cams[i].copy(), image.copy(), side_view=False)
        side = renderer(verts[i], cams[i].copy(), image.copy(), side_view=True)
        canvas = np.concatenate([panel(image,"Input"), panel(overlay,"GT + prediction"),
            panel(front,f"{name}: mesh"), panel(side,f"{name}: side")], axis=1)
        cv2.imwrite(str(out_dir/f"sample_{ordinal:05d}_{name}.jpg"), canvas)

def evaluate(name, checkpoint, cfg, args, out_dir):
    model = load_model(cfg, checkpoint, args.device)
    renderer = MeshRenderer(cfg, faces=model.mano.faces) if args.visual_samples else None
    # Each rendered panel creates and closes its own context; discard the unused persistent one.
    if renderer is not None:
        renderer.renderer.delete()
        renderer.renderer = None
    totals, counts = defaultdict(float), defaultdict(int); samples = 0; started = time.time()
    amp = args.device.startswith("cuda") and args.precision != "fp32"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    loader = make_loader(cfg)
    with torch.inference_mode():
        for batch_idx, cpu_batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches: break
            n = cpu_batch["img"].shape[0]; batch = move(cpu_batch, args.device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp):
                output = model.forward_step(batch, train=False)
            for key, value in model.compute_validation_metrics(batch, output).items():
                totals[key] += value.detach().float().item()*n; counts[key] += n
            if renderer: save_visuals(renderer, cpu_batch, output, name, args.visual_samples, samples, out_dir, cfg)
            samples += n
            if (batch_idx+1) % args.log_every == 0 or batch_idx == 0:
                print(f"[{name}] batches={batch_idx+1}, samples={samples}", flush=True)
    result = {"model":name, "checkpoint":str(Path(checkpoint).resolve()) if checkpoint else "ViTPose-H backbone; random WiLoR head",
              "samples":samples, "seconds":time.time()-started}
    result.update({key:totals[key]/counts[key] for key in totals})
    del renderer, model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def write_reports(results, out_dir):
    fields = ["model","checkpoint","samples","seconds",*METRICS]
    with (out_dir/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(results)
    (out_dir/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    headers=["Model","Samples",*[LABELS[k] for k in METRICS]]
    lines=["# FreiHAND validation comparison","","| "+" | ".join(headers)+" |",
           "| "+" | ".join(["---"]*len(headers))+" |"]
    for result in results:
        values=[result["model"],str(result["samples"]),*[f'{result[k]:.3f}' if k in result else "N/A" for k in METRICS]]
        lines.append("| "+" | ".join(values)+" |")
    lines += ["","Lower is better for every metric.",""]
    (out_dir/"report.md").write_text("\n".join(lines))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trained-checkpoint",required=True); p.add_argument("--official-checkpoint",required=True)
    p.add_argument("--initial-backbone",default="hamer_training_data/vitpose_backbone.pth")
    p.add_argument("--output-dir",default="evaluation/freihand_comparison")
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--workers",type=int,default=4)
    p.add_argument("--device",default="cuda"); p.add_argument("--precision",choices=["bf16","fp16","fp32"],default="bf16")
    p.add_argument("--max-batches",type=int,default=None,help="Debug limit; omit for all 10,240 samples")
    p.add_argument("--visual-samples",type=int,nargs="*",default=[0,511,2047,4095,8191])
    p.add_argument("--log-every",type=int,default=50); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True); cfg=build_cfg(args.batch_size,args.workers)
    results=[evaluate("trained",args.trained_checkpoint,cfg,args,out_dir),
             evaluate("official",args.official_checkpoint,cfg,args,out_dir)]
    torch.manual_seed(42)
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = args.initial_backbone
    results.append(evaluate("initial",None,cfg,args,out_dir))
    write_reports(results,out_dir); print((out_dir/"report.md").read_text()); print(f"Results written to {out_dir.resolve()}")
if __name__ == "__main__": main()
