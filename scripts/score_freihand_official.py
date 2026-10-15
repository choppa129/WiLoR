"""Score official-format pred.json using pinned FreiHAND metric definitions."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilor.utils.official_freihand_eval import evaluate_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--ground-truth', type=Path, required=True,
                   help='Directory containing evaluation_xyz.json and evaluation_verts.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    pred = json.loads(args.predictions.read_text())
    if len(pred) != 2:
        raise ValueError('pred.json must contain [joint_predictions, vertex_predictions]')
    joints = json.loads((args.ground_truth/'evaluation_xyz.json').read_text())
    verts = json.loads((args.ground_truth/'evaluation_verts.json').read_text())
    if len(joints) != 3960:
        raise ValueError('Official FreiHAND evaluation requires all 3960 samples in canonical order')
    result = evaluate_predictions(pred[0], pred[1], joints, verts)
    result['protocol'] = 'official FreiHAND evaluation split; meters input, PA errors reported in mm'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
