"""Arithmetic correction of archived metrics; no inference and no timing claim."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'evaluation/freihand_full_comparison'
out=ROOT/'evaluation/freihand_full_comparison_corrected'
out.mkdir(exist_ok=True)
rows=json.loads((source/'metrics.json').read_text())
for row in rows:
    row['val/reprojection_px']*=2
    row['historical_seconds']=row.pop('seconds')
    row['provenance']='Arithmetic x2 correction of historical reprojection metric; not a new evaluation'
(out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
with (out/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# Corrected historical FreiHAND comparison','',
    'Arithmetic correction only; no new evaluation. Original files remain in ../freihand_full_comparison.',
    'Normalized coordinates are pixel / IMAGE_SIZE - 0.5. Multiply errors by IMAGE_SIZE, not IMAGE_SIZE/2.',
    'All 3D metrics, sample counts and checkpoint identities are unchanged. Historical timings are not new timings.',
    '', '| Model | Samples | MPJPE mm | PA-MPJPE mm | PVE mm | Corrected reprojection px |',
    '|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f'| {r["model"]} | {r["samples"]} | '+ ' | '.join(f'{r[k]:.6f}' for k in
        ['val/mpjpe_mm','val/pa_mpjpe_mm','val/pve_mm','val/reprojection_px'])+' |')
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
