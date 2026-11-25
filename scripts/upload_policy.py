"""Upload an inference-only policy checkpoint (pretrained_model/) to a branch of a shared Hub model repo.

Each policy lives on its own branch so one repo can hold many variants:
    <policy_type>[-<tag>]-step<step>    e.g. act-step048845, act-chunk50-step100000, pi05-step030000

Usage:
    uv run python scripts/upload_policy.py \
        outputs/act_pick-blocks-from-slot/checkpoints/048845 \
        --repo-id eugene123tw/lerobot-pick-wood-from-slots

Inference:
    lerobot-rollout --policy.path=<repo-id> --policy.pretrained_revision=<branch> ...
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path, help="Checkpoint step dir containing pretrained_model/")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--tag", default=None, help="Optional variant tag, e.g. 'chunk50' or 'aug'")
    parser.add_argument("--branch", default=None, help="Override the auto-generated branch name")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model_dir = args.checkpoint_dir / "pretrained_model"
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"{model_dir}/config.json not found")

    policy_type = json.loads((model_dir / "config.json").read_text())["type"]
    step = args.checkpoint_dir.resolve().name
    branch = args.branch or "-".join(p for p in (policy_type, args.tag, f"step{step}") if p)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.create_branch(args.repo_id, branch=branch, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Upload {policy_type} step {step}",
    )

    print(f"Uploaded to https://huggingface.co/{args.repo_id}/tree/{branch}")
    print(f"  --policy.path={args.repo_id} --policy.pretrained_revision={branch}")


if __name__ == "__main__":
    main()
