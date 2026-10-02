"""Run a declared v6 recovery DAG through XenWarden, with archived completion receipts.

A private JSON config supplies immutable checkout paths, commands, resources and
scientific output contracts. ``tick`` is a lightweight timer action; only ``run``
executes computation, inside an admitted lease. Failed jobs need inspection and
an explicit reset in the private state file, rather than an endless retry loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def job_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        **config["jobs"][name],
        "execution": {
            "environment": config["environment"],
            "python": config["python"],
            "runner_sha256": digest(Path(config["runner"])),
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def validate_result(path: Path, contract: dict[str, Any]) -> None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: result is not an object")
    for key, expected in contract.get("equals", {}).items():
        if data.get(key) != expected:
            raise ValueError(f"{path}: {key} does not match its contract")
    for model in contract.get("models", []):
        chosen = data.get("chosen", {}).get(model)
        if not chosen or not data.get("curve", {}).get(model):
            raise ValueError(f"{path}: missing measured model {model}")
        if not data.get("stored_vectors", {}).get(model):
            raise ValueError(f"{path}: no retained vectors for {model}")
        if contract.get("controls") and not data.get("controls", {}).get(model):
            raise ValueError(f"{path}: no recording controls for {model}")
    if contract.get("reproduction"):
        rows = data.get("per_endpoint", {})
        if not rows or not all(row.get("passed") for row in rows.values()):
            raise ValueError(f"{path}: v5 reproduction did not pass")
    if contract.get("fixed"):
        expected = set(data.get("models", []))
        rows = data.get("reproduces_fixed_sweep", {})
        if not expected or set(rows) != expected or not all(r["identical"] for r in rows.values()):
            raise ValueError(f"{path}: fixed extraction did not reproduce every requested model")
    for key in contract.get("required", []):
        if key not in data:
            raise ValueError(f"{path}: missing {key}")


def receipt_valid(path: Path, spec: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return (
            data["spec_sha256"] == fingerprint(spec)
            and data["archived"] is True
            and all(Path(p).is_file() and digest(Path(p)) == h for p, h in data["files"].items())
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def archive(path: Path) -> None:
    from xinyenyana.archive import archive_result

    receipt = archive_result(path, prefix=os.environ["XINYENYANA_RESULT_ARCHIVE"])
    if not receipt["archived"]:
        raise RuntimeError(f"archive failed for {path}: {receipt}")


def run_step(root: Path, name: str, step: dict[str, Any]) -> list[str]:
    """Resume only an output tied to the same inputs and successful validation."""

    inputs = {p: digest(Path(p)) for p in step.get("inputs", [])}
    spec = {**step, "input_sha256": inputs}
    receipt = root / "receipts" / f"{name}.json"
    if receipt_valid(receipt, spec):
        print(f"{name}: verified completion receipt", flush=True)
        return list(json.loads(receipt.read_text())["files"])
    intent = root / "intents" / f"{name}.json"
    same_intent = intent.exists() and json.loads(intent.read_text()) == spec
    outputs = step["outputs"]
    # A process can finish writing and be stopped while uploading. Its output
    # can be validated and uploaded again without running the model again.
    can_resume = (same_intent or step.get("adopt_existing", False)) and all(
        Path(p).is_file() for p in outputs
    )
    if not can_resume:
        for output in outputs:
            path = Path(output)
            if path.exists() and not step.get("adopt_existing", False):
                archive(path)
                old = root / "superseded" / f"{digest(path)}-{path.name}"
                old.parent.mkdir(parents=True, exist_ok=True)
                if old.exists() and digest(old) != digest(path):
                    raise ValueError(f"preserved result collision: {old}")
                path.replace(old)
        atomic_json(intent, spec)
        command = list(step["command"])
        if "fixed_from" in step:
            merged = json.loads(Path(step["fixed_from"]).read_text())
            original = json.loads(Path(step["original_sweep"]).read_text())
            changed = [
                model
                for model, choice in merged["chosen"].items()
                if model not in original["chosen"]
                or choice["representation"] != original["chosen"][model]["representation"]
                or choice.get("read_in_windows_of_seconds")
                != original["chosen"][model].get("read_in_windows_of_seconds")
            ]
            if changed:
                for model in sorted(changed):
                    command += ["--model", model]
            else:
                # An explicit archived decision is the output when extraction
                # is unnecessary. Downstream code can read its empty mappings.
                output = Path(next(iter(outputs)))
                atomic_json(
                    output,
                    {
                        "experiment": "PA-V6 fixed-choice decision",
                        "endpoint": merged["endpoint"],
                        "reason": "every merged choice already has full controlled vectors",
                        "sources": inputs,
                        "models": [],
                        "chosen": {},
                        "stored_vectors": {},
                        "nothing_changed": True,
                        "lease": os.environ["XENWARDEN_LEASE"],
                    },
                )
                command = []
        if command:
            print(f"{name}: starting", flush=True)
            subprocess.run(command, check=True)
    for output, contract in outputs.items():
        data = json.loads(Path(output).read_text())
        if step.get("fixed_from") and data.get("nothing_changed"):
            if data.get("sources") != inputs:
                raise ValueError("fixed-choice decision has different inputs")
        else:
            validate_result(Path(output), contract)
        archive(Path(output))
    record = {
        "spec_sha256": fingerprint(spec),
        "files": {**inputs, **{p: digest(Path(p)) for p in outputs}},
        "lease": os.environ["XENWARDEN_LEASE"],
        "archived": True,
    }
    # Archive the receipt before making it visible to the scheduler.
    staging = receipt.with_suffix(".pending")
    atomic_json(staging, record)
    archive(staging)
    staging.replace(receipt)
    return list(record["files"])


def run_job(config: dict[str, Any], name: str) -> None:
    if not os.environ.get("XENWARDEN_LEASE") or not os.environ.get("XINYENYANA_RESULT_ARCHIVE"):
        raise RuntimeError("workers require an admitted lease and the result archive")
    root = Path(config["root"])
    job = job_spec(config, name)
    os.environ.update(config["environment"])
    os.environ.update(job.get("environment", {}))
    outputs = []
    for index, step in enumerate(job["steps"]):
        outputs.extend(
            run_step(
                root,
                f"{name}-{index:03d}",
                {**step, "execution": job["execution"], "environment": job.get("environment", {})},
            )
        )
    record = {
        "spec_sha256": fingerprint(job),
        "files": {p: digest(Path(p)) for p in outputs},
        "lease": os.environ["XENWARDEN_LEASE"],
        "archived": True,
    }
    pending = root / "receipts" / f"job-{name}.pending"
    atomic_json(pending, record)
    archive(pending)
    pending.replace(pending.with_suffix(".json"))


def warden(executable: str, *args: str) -> Any:
    result = subprocess.run(
        [executable, "--json", *args], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def tick(config_path: Path, config: dict[str, Any]) -> None:
    """Submit ready jobs once, retaining request keys across ambiguous responses."""

    import fcntl

    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "workflow-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        executable = config["warden"]
        live = warden(executable, "queue")["live"]
        active = {lease["lease_id"] for lease in live}
        done = {
            name
            for name, job in config["jobs"].items()
            if receipt_valid(root / "receipts" / f"job-{name}.json", job_spec(config, name))
        }
        # Prevent competing large allocations within this workflow, including
        # adopted speech work. Warden remains authoritative across all tenants.
        gpu_jobs = sum(
            1
            for lease in live
            if lease["tenant"] == "xinyenyana" and lease["asked_for"]["gpu"]["count"]
        )
        for name, job in config["jobs"].items():
            entry = state.setdefault(name, {})
            if name in done:
                entry["state"] = "complete"
                continue
            if entry.get("lease_id") in active:
                entry["state"] = "admitted-or-queued"
                continue
            if entry.get("lease_id") and entry.get("state") != "failed":
                terminal = warden(executable, "status", entry["lease_id"])
                entry.update(state="failed", reason=terminal.get("reason"), terminal=terminal)
            if entry.get("state") == "failed":
                continue
            if any(lease in active for lease in job.get("wait_for_leases", [])):
                entry["state"] = "waiting-for-existing-lease"
                continue
            if not set(job.get("after", [])) <= done:
                entry["state"] = "waiting-for-dependencies"
                continue
            if any(not Path(p).is_file() for p in job.get("requires_files", [])):
                entry["state"] = "waiting-for-inputs"
                continue
            is_gpu = bool(job["resources"]["gpu"]["count"])
            if is_gpu and gpu_jobs >= config.get("max_gpu_jobs", 2):
                entry["state"] = "waiting-for-gpu-slot"
                continue
            key = entry.setdefault("request_key", f"xyy-v6-{name}-{fingerprint(job)[:20]}")
            request = {
                "tenant": "xinyenyana",
                "class": job.get("class", "batch"),
                "host": "xen1",
                "resources": {**job["resources"], "scratch_gb": 0},
                "max_runtime_sec": job.get("max_runtime_sec", 172800),
                "preemption": {"signal": "SIGTERM", "grace_sec": 30},
                "command": [
                    "/usr/bin/env",
                    *[f"{k}={v}" for k, v in config["environment"].items()],
                    config["python"],
                    config["runner"],
                    "run",
                    str(config_path),
                    name,
                ],
                "request_key": key,
                "notes": f"PA-V6 corrected recovery: {name}; validated, archived receipts",
            }
            atomic_json(state_path, state)
            response = subprocess.run(
                [executable, "--json", "request", "-f", "-"],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            # The host CLI prints a bare lease id even with --json.
            if not response.startswith("xw-") or "\n" in response:
                raise RuntimeError(f"unexpected request response: {response!r}")
            entry.update(lease_id=response, state="admitted-or-queued")
            gpu_jobs += int(is_gpu)
            atomic_json(state_path, state)
        atomic_json(state_path, state)
        print(json.dumps(state, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tick", "run"))
    parser.add_argument("config", type=Path)
    parser.add_argument("job", nargs="?")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.action == "tick":
        tick(args.config.resolve(), config)
    else:
        if args.job is None:
            parser.error("run requires a job name")
        run_job(config, args.job)


if __name__ == "__main__":
    main()
