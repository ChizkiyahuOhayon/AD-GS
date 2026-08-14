#!/usr/bin/env python3
"""Fail closed unless the research branch only adds approved isolated files."""

import argparse
import json
import subprocess
from pathlib import Path


OFFICIAL_BASE = "9a208512e49c8ddbaa20387921d9648adcd21cb4"
ALLOWED_FILES = {"experiments.md", "server.md"}
ALLOWED_PREFIXES = (
    "scripts/trust4d/",
    "tests/",
    "research/ara/trust4d_teacher_reliability/",
)


def parse_name_status(text):
    records = []
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"unexpected git name-status record: {line!r}")
        records.append((fields[0], fields[1]))
    return records


def validate_records(records):
    errors = []
    additions = []
    for status, path in records:
        if status != "A":
            errors.append(f"official source is not byte-identical: {status} {path}")
            continue
        if path not in ALLOWED_FILES and not path.startswith(ALLOWED_PREFIXES):
            errors.append(f"unapproved addition relative to official AD-GS: {path}")
            continue
        additions.append(path)
    if errors:
        raise ValueError("; ".join(errors))
    return additions


def audit_repository(repository, official_base=OFFICIAL_BASE):
    repository = Path(repository).expanduser().resolve()
    subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", official_base, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if status:
        raise ValueError(f"research checkout is not clean:\n{status}")
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--name-status",
            "--no-renames",
            official_base,
            "HEAD",
        ],
        text=True,
    )
    additions = validate_records(parse_name_status(diff))
    return {
        "passed": True,
        "repository": str(repository),
        "official_base": official_base,
        "head": head,
        "working_tree_clean": True,
        "allowed_files": sorted(ALLOWED_FILES),
        "allowed_prefixes": list(ALLOWED_PREFIXES),
        "additions": additions,
        "raw_name_status": diff.splitlines(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_repository(args.repository)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
