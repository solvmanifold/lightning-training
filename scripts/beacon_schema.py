"""Dependency-free validation and field scoring for Beacon job JSON."""

from __future__ import annotations

import re
from typing import Any


REQUIRED_KEYS = {
    "schema_version",
    "action",
    "resource_id",
    "target_region",
    "urgency",
    "execution",
    "encryption",
    "retention_days",
    "compression",
    "notify",
}
ACTIONS = {"replicate", "snapshot", "restore", "retire"}
REGIONS = {"us-north", "us-south", "eu-central", "ap-east"}
URGENCIES = {"low", "normal", "high", "critical"}
RESOURCE_PATTERN = re.compile(r"^res_[a-z]{2}_[0-9]{5}$")
EMAIL_PATTERN = re.compile(r"^[a-z0-9-]+@example\.invalid$")


def validate_beacon(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["root is not an object"]
    errors: list[str] = []
    keys = set(value)
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"extra keys: {sorted(extra)}")
    if missing:
        return errors

    if value["schema_version"] != "beacon-job-v1":
        errors.append("invalid schema_version")
    if value["action"] not in ACTIONS:
        errors.append("invalid action")
    if not isinstance(value["resource_id"], str) or not RESOURCE_PATTERN.fullmatch(
        value["resource_id"]
    ):
        errors.append("invalid resource_id")
    target = value["target_region"]
    if target is not None and target not in REGIONS:
        errors.append("invalid target_region")
    if value["action"] in {"replicate", "restore"} and target is None:
        errors.append("action requires target_region")
    if value["action"] in {"snapshot", "retire"} and target is not None:
        errors.append("action forbids target_region")
    if value["urgency"] not in URGENCIES:
        errors.append("invalid urgency")
    if value["execution"] not in {"plan", "apply"}:
        errors.append("invalid execution")
    if value["encryption"] not in {"standard", "strict"}:
        errors.append("invalid encryption")
    retention = value["retention_days"]
    if isinstance(retention, bool) or not isinstance(retention, int) or not 1 <= retention <= 365:
        errors.append("invalid retention_days")
    if not isinstance(value["compression"], bool):
        errors.append("invalid compression")
    notify = value["notify"]
    if not isinstance(notify, list):
        errors.append("notify is not an array")
    elif any(not isinstance(item, str) or not EMAIL_PATTERN.fullmatch(item) for item in notify):
        errors.append("invalid notify address")
    elif len(set(notify)) != len(notify):
        errors.append("duplicate notify address")
    return errors


def field_counts(predicted: object, expected: dict[str, Any]) -> tuple[int, int, int]:
    """Return correct, predicted, and expected top-level field counts."""
    if not isinstance(predicted, dict):
        return 0, 0, len(expected)
    correct = sum(key in expected and predicted[key] == expected[key] for key in predicted)
    return correct, len(predicted), len(expected)
