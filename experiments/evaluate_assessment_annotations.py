"""Evaluate supplied annotations against actual saved v2 responses.

python experiments/evaluate_assessment_annotations.py --learner-id ID --annotations labels.json [--db PATH]

Input JSON: {"dataset":"annotated subset", "annotation_source":"expert/version",
 "annotations":[{"turn_id":"saved turn ID", "category":"understanding",
 "concept_id":"saved concept ID", "misconception_present":false, "claim_faithful":true}]}

No fixture labels are provided as research results. The caller supplies real labels.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.learner_state_core.assessment.evaluation import evaluate_annotations
from packages.learner_state_core.db import LocalDatabaseManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learner-id', required=True)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--db')
    args = parser.parse_args()
    run = evaluate_annotations(LocalDatabaseManager(args.db), args.learner_id, json.loads(args.annotations.read_text(encoding='utf-8')))
    print(json.dumps({k:v for k,v in run.items() if k != 'records'}, indent=2))
