"""
Utilities for creating and inspecting RoboCasa agent workflow runs.

This module intentionally uses only the Python standard library so it can run in
an editable RoboCasa checkout without adding workflow-specific dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


WORKFLOW_DIR = ".agent-workflow"
RUNS_DIR = ".agent-runs"
REQUIRED_RUN_FILES = [
    "requirements.md",
    "repo_context.md",
    "implementation_plan.md",
    "implementation_notes.md",
    "test_log.md",
    "functional_review.md",
    "code_review.md",
    "final_validation.md",
]
TEMPLATE_TO_RUN_FILE = {
    "00_requirement_intake.md": "requirements.md",
    "01_repo_analysis.md": "repo_context.md",
    "02_implementation_plan.md": "implementation_plan.md",
    "03_implementation_notes.md": "implementation_notes.md",
    "04_functional_review.md": "functional_review.md",
    "05_code_review.md": "code_review.md",
    "06_final_validation.md": "final_validation.md",
}


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    workflow_dir: Path
    template_dir: Path
    runs_dir: Path
    agents_file: Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest parent directory that looks like the repo root."""

    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "robocasa").is_dir() and (path / "README.md").exists():
            return path
    raise RuntimeError("Could not find RoboCasa repository root from current path.")


def get_paths() -> RepoPaths:
    root = find_repo_root()
    workflow_dir = root / WORKFLOW_DIR
    return RepoPaths(
        root=root,
        workflow_dir=workflow_dir,
        template_dir=workflow_dir / "templates",
        runs_dir=root / RUNS_DIR,
        agents_file=workflow_dir / "agents.json",
    )


def slugify(value: str) -> str:
    """Convert a feature name to a safe directory name."""

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    if not value:
        raise ValueError("Feature name cannot be empty after normalization.")
    return value


def load_agents(paths: RepoPaths) -> dict:
    if not paths.agents_file.exists():
        raise FileNotFoundError(f"Missing workflow config: {paths.agents_file}")
    return json.loads(paths.agents_file.read_text(encoding="utf-8"))


def iter_templates(paths: RepoPaths) -> Iterable[Path]:
    if not paths.template_dir.exists():
        raise FileNotFoundError(f"Missing template directory: {paths.template_dir}")
    return sorted(paths.template_dir.glob("*.md"))


def init_run(feature: str, summary: str, overwrite: bool = False) -> Path:
    paths = get_paths()
    run_name = slugify(feature)
    run_dir = paths.runs_dir / run_name

    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --overwrite to replace it."
        )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    readme = (
        f"# Agent Run: {feature}\n\n"
        f"- Created at: `{created_at}`\n"
        f"- Summary: {summary}\n"
        f"- Workflow config: `{WORKFLOW_DIR}/agents.json`\n\n"
        "## Recommended order\n\n"
        "1. Fill `requirements.md`.\n"
        "2. Fill `repo_context.md`.\n"
        "3. Fill `implementation_plan.md`.\n"
        "4. Implement code and update `implementation_notes.md` plus `test_log.md`.\n"
        "5. Fill `functional_review.md`.\n"
        "6. Fill `code_review.md`.\n"
        "7. Fill `final_validation.md`.\n"
    )
    (run_dir / "README.md").write_text(readme, encoding="utf-8")

    for template_path in iter_templates(paths):
        target_name = TEMPLATE_TO_RUN_FILE.get(template_path.name, template_path.name)
        target_path = run_dir / target_name
        content = template_path.read_text(encoding="utf-8")
        content = content.replace("在这里粘贴用户原始需求。不要改写原意。", summary)
        target_path.write_text(content, encoding="utf-8")

    test_log = run_dir / "test_log.md"
    if not test_log.exists():
        test_log.write_text("# Test Log\n\n", encoding="utf-8")

    return run_dir


def print_agents() -> None:
    paths = get_paths()
    config = load_agents(paths)
    print(f"Workflow: {config.get('workflow_name', 'Unknown')}")
    print(f"Repository: {config.get('repository', 'Unknown')}")
    print("")
    for agent in config.get("agents", []):
        print(f"[{agent['id']}] {agent['name']}")
        print(f"  model: {agent.get('recommended_model', 'not specified')}")
        owns = ", ".join(agent.get("owns", []))
        print(f"  owns: {owns}")
        print("")


def status(run: str) -> int:
    paths = get_paths()
    run_dir = (paths.root / run).resolve() if not Path(run).is_absolute() else Path(run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    missing: list[str] = []
    empty: list[str] = []
    for name in REQUIRED_RUN_FILES:
        path = run_dir / name
        if not path.exists():
            missing.append(name)
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            empty.append(name)

    print(f"Run: {run_dir}")
    if missing:
        print("Missing files:")
        for name in missing:
            print(f"  - {name}")
    if empty:
        print("Empty files:")
        for name in empty:
            print(f"  - {name}")
    if not missing and not empty:
        print("All required workflow files exist and are non-empty.")
    return 1 if missing or empty else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboCasa agent workflow helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new agent run directory")
    init_parser.add_argument("--feature", required=True, help="Feature or task name")
    init_parser.add_argument("--summary", required=True, help="Short requirement summary")
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory with the same feature slug",
    )

    subparsers.add_parser("agents", help="Print configured agent roles")

    status_parser = subparsers.add_parser("status", help="Inspect an agent run directory")
    status_parser.add_argument("--run", required=True, help="Run directory path")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        run_dir = init_run(args.feature, args.summary, args.overwrite)
        print(f"Created agent run: {run_dir}")
        return 0
    if args.command == "agents":
        print_agents()
        return 0
    if args.command == "status":
        return status(args.run)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
