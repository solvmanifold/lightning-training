#!/usr/bin/env python3
"""Generate the deterministic, LLM-free Atlas tool-routing smoke dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


SEED = 35_003_500
VERSION = "atlas-smoke-v1"
SPLIT_SIZES = {"train": 120, "development": 40, "test": 40, "regression": 24}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "suspend_account",
            "description": "Temporarily prevent an account from signing in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "reason_code": {"enum": ["security_incident", "policy_violation"]},
                },
                "required": ["account_id", "reason_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_account",
            "description": "Permanently schedule an account for deletion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "reason_code": {"enum": ["user_request", "policy_violation"]},
                },
                "required": ["account_id", "reason_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_credential",
            "description": "Replace a credential and return a new active credential.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "credential_id": {"type": "string"},
                    "reason_code": {"enum": ["scheduled_rotation", "suspected_exposure"]},
                },
                "required": ["project_id", "credential_id", "reason_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revoke_credential",
            "description": "Disable a credential without creating a replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "credential_id": {"type": "string"},
                    "reason_code": {"enum": ["suspected_exposure", "no_longer_needed"]},
                },
                "required": ["project_id", "credential_id", "reason_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_project_role",
            "description": "Change a project member's role without changing ownership.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "role": {"enum": ["viewer", "editor", "admin"]},
                },
                "required": ["project_id", "user_id", "role"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_project_ownership",
            "description": "Make a project member the project's owner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "new_owner_id": {"type": "string"},
                },
                "required": ["project_id", "new_owner_id"],
                "additionalProperties": False,
            },
        },
    },
]

TEMPLATES = {
    "train": {
        "suspend": ["Suspend {account}; security flagged it.", "Block sign-in for {account} after the alert."],
        "delete": ["Delete {account} at the user's request.", "Schedule {account} for permanent removal."],
        "rotate": ["Rotate {credential} in {project}; it may be exposed.", "Replace compromised key {credential} for {project}."],
        "revoke": ["Revoke unused key {credential} in {project}.", "Disable {credential} for {project}; no replacement."],
        "role": ["Make {user} an {role} on {project}.", "Set {user}'s {project} role to {role}."],
        "owner": ["Transfer {project} ownership to {user}.", "Make {user} the owner of {project}."],
        "clarify": ["Change Sam's role on {project} to editor.", "Update the Sam on {project} to editor."],
        "noop": ["Suspend {account}; it should stay blocked.", "Block {account} again."],
        "chain": ["Contain {account}: suspend it and revoke {credential} in {project}.", "Block {account}, then disable {credential} for {project}."],
    },
    "development": {
        "suspend": ["Freeze login access for {account} because of a security incident."],
        "delete": ["The owner asked us to permanently remove {account}."],
        "rotate": ["Issue a replacement for exposed {credential} under {project}."],
        "revoke": ["Kill {project}'s obsolete credential {credential} without replacing it."],
        "role": ["On {project}, promote {user} to {role}."],
        "owner": ["Hand control of {project} over to {user}."],
        "clarify": ["Give Sam editor access in {project}."],
        "noop": ["Keep already-suspended {account} suspended."],
        "chain": ["Lock out {account} and invalidate exposed {credential} on {project}."],
    },
    "test": {
        "suspend": ["Security response: prevent {account} from authenticating."],
        "delete": ["Honor the deletion request for account {account}."],
        "rotate": ["Roll {project} key {credential} after suspected leakage."],
        "revoke": ["Retire {credential} from {project}; do not mint another key."],
        "role": ["{user} needs {role} privileges for {project}."],
        "owner": ["{user} should take over {project}."],
        "clarify": ["Set Sam to editor for {project}; you know which one."],
        "noop": ["Ensure {account}, which is blocked, remains blocked."],
        "chain": ["Quarantine {account}, including disabling {credential} on {project}."],
    },
}

REGRESSION_TEMPLATES = [
    "Summarize the current state of {project}.",
    "Hello, can you explain what account suspension means?",
    "How many members does {project} have?",
    "Draft a note about credential hygiene for {project}.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(name: str, **arguments: str) -> dict[str, object]:
    return {"name": name, "arguments": arguments}


def make_case(split: str, index: int, rng: random.Random) -> dict[str, object]:
    account = f"acct_{split[:2]}_{index:04d}"
    project = f"proj_{split[:2]}_{index:04d}"
    credential = f"cred_{index:04d}_{rng.randrange(1000, 9999)}"
    user = f"usr_{index:04d}_{rng.randrange(100, 999)}"
    state = {
        "account": {"id": account, "status": "active"},
        "project": {"id": project, "owner_id": f"usr_owner_{index:04d}"},
        "credential": {"id": credential, "project_id": project, "status": "active"},
        "member": {"id": user, "project_id": project, "role": "viewer"},
    }

    if split == "regression":
        request = REGRESSION_TEMPLATES[index % len(REGRESSION_TEMPLATES)].format(project=project)
        expected = {"decision": "no_action", "reason": "not_an_atlas_action", "tool_calls": []}
        scenario = "unrelated"
    else:
        scenario = ("suspend", "delete", "rotate", "revoke", "role", "owner", "clarify", "noop", "chain")[index % 9]
        role = ("viewer", "editor", "admin")[index % 3]
        values = {"account": account, "project": project, "credential": credential, "user": user, "role": role}
        request = rng.choice(TEMPLATES[split][scenario]).format(**values)
        if scenario == "suspend":
            expected = {"decision": "call", "tool_calls": [call("suspend_account", account_id=account, reason_code="security_incident")]}
        elif scenario == "delete":
            expected = {"decision": "call", "tool_calls": [call("delete_account", account_id=account, reason_code="user_request")]}
        elif scenario == "rotate":
            expected = {"decision": "call", "tool_calls": [call("rotate_credential", project_id=project, credential_id=credential, reason_code="suspected_exposure")]}
        elif scenario == "revoke":
            expected = {"decision": "call", "tool_calls": [call("revoke_credential", project_id=project, credential_id=credential, reason_code="no_longer_needed")]}
        elif scenario == "role":
            expected = {"decision": "call", "tool_calls": [call("change_project_role", project_id=project, user_id=user, role=role)]}
        elif scenario == "owner":
            expected = {"decision": "call", "tool_calls": [call("transfer_project_ownership", project_id=project, new_owner_id=user)]}
        elif scenario == "clarify":
            state["members_named_sam"] = [f"usr_sam_{index:04d}_a", f"usr_sam_{index:04d}_b"]
            expected = {"decision": "clarify", "required_fields": ["user_id"], "tool_calls": []}
        elif scenario == "noop":
            state["account"]["status"] = "suspended"
            expected = {"decision": "no_action", "reason": "already_in_target_state", "tool_calls": []}
        else:
            expected = {
                "decision": "call",
                "tool_calls": [
                    call("suspend_account", account_id=account, reason_code="security_incident"),
                    call("revoke_credential", project_id=project, credential_id=credential, reason_code="suspected_exposure"),
                ],
            }

    user_message = f"State:\n{json.dumps(state, sort_keys=True)}\n\nRequest:\n{request}"
    return {
        "id": f"atlas-{split}-{index:04d}",
        "suite": "atlas-tool-routing",
        "messages": [{"role": "user", "content": user_message}],
        "tool_catalog": VERSION,
        "expected": expected,
        "scorers": ["atlas_exact_workflow"],
        "tags": [f"split:{split}", f"scenario:{scenario}"],
    }


def generate(repo_root: Path) -> None:
    output_dir = repo_root / "data/synthetic/atlas_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "tool_catalog.json"
    catalog_path.write_text(
        json.dumps({"id": VERSION, "tools": TOOLS}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifacts: dict[str, dict[str, object]] = {}
    all_prompts: set[str] = set()
    for split_number, (split, size) in enumerate(SPLIT_SIZES.items()):
        rng = random.Random(SEED + split_number)
        cases = [make_case(split, index, rng) for index in range(size)]
        prompts = [case["messages"][0]["content"] for case in cases]
        if len(set(prompts)) != len(prompts) or all_prompts.intersection(prompts):
            raise RuntimeError(f"duplicate prompt detected in {split}")
        all_prompts.update(prompts)
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n" for case in cases),
            encoding="utf-8",
        )
        artifacts[split] = {
            "path": str(path.relative_to(repo_root)),
            "rows": size,
            "sha256": sha256(path),
        }

    manifest = {
        "artifacts": artifacts,
        "generator": {
            "path": "scripts/generate_atlas_smoke.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "seed": SEED,
        "tool_catalog": {
            "path": str(catalog_path.relative_to(repo_root)),
            "sha256": sha256(catalog_path),
            "tools": len(TOOLS),
        },
        "version": VERSION,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    generate(args.repo_root.resolve())


if __name__ == "__main__":
    main()
