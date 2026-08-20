from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .client import DaemonClient, RemoteError
from .errors import DaemonUnavailableError, KernelyraError
from .models import RunConfig

VERSION = "0.3.0a1"
TERMINAL_STATES = {"completed", "stopped", "error", "error_recoverable"}
EXIT_EXPECTED_ERROR = 2
EXIT_AUTHORIZATION = 4
EXIT_UNAVAILABLE = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelyra", description=f"Kernelyra {VERSION}")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--json", action="store_true", help="Emit stable JSON output")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout in seconds")
    parser.add_argument(
        "--daemon-url",
        default=os.environ.get("KERNELYRA_DAEMON_URL", "http://127.0.0.1:8765"),
        help="Kernelyra daemon URL",
    )
    parser.add_argument("--autostart", action="store_true", help="Start the daemon when unavailable")
    parser.add_argument("--api-token", default=os.environ.get("KERNELYRA_API_TOKEN"), help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version")
    commands.add_parser("doctor")
    commands.add_parser("capabilities")
    commands.add_parser("formats", help="List recognized routes and installed trainable adapters")
    commands.add_parser("modes", help="Show the four adaptive execution programs for this computer")
    tune = commands.add_parser("tune", help="Preview deterministic native execution tuning")
    tune.add_argument("--profile", choices=["auto", "eco", "low-memory", "balanced", "performance", "workstation", "custom"], default="auto")
    tune.add_argument("--records", type=int, default=100_000)
    tune.add_argument("--features", type=int, default=32)
    tune.add_argument("--batch-size", type=int, default=64)
    tune.add_argument("--streaming", action="store_true")
    chunk_plan = commands.add_parser("chunk-plan", help="Preview deterministic variable dataset chunks")
    chunk_plan.add_argument("records", type=int, help="Total record count to divide")
    chunk_plan.add_argument("--target-records", type=int, default=4096)
    chunk_plan.add_argument("--minimum-records", type=int)
    chunk_plan.add_argument("--maximum-records", type=int)
    chunk_plan.add_argument("--seed", type=int, default=42)

    def add_training_options(item: argparse.ArgumentParser) -> None:
        item.add_argument("dataset", help="Dataset file or folder path")
        item.add_argument("--config", help="TOML configuration path; defaults to WORKSPACE/kernelyra.toml")
        item.add_argument("--target")
        item.add_argument("--task", choices=["auto", "binary_classification", "multiclass_classification", "regression"])
        item.add_argument("--backend", choices=["auto", "native", "torch", "tensorflow", "numpy"])
        item.add_argument("--architecture", choices=["auto", "linear", "mlp", "transformer", "cnn", "vision-transformer", "rnn", "pointnet", "graph-neural-network"])
        item.add_argument("--model-format", choices=["auto", "kernelyra-npz", "pytorch-state", "keras", "gguf", "safetensors", "onnx"])
        item.add_argument(
            "--profile",
            choices=["auto", "eco", "low-memory", "balanced", "performance", "workstation", "custom"],
        )
        item.add_argument("--batch-size", type=int)
        item.add_argument("--accept-batch-risk", action="store_true")
        item.add_argument("--max-steps", type=int)
        item.add_argument("--target-metric", type=float)
        item.add_argument("--learning-rate", type=float)
        item.add_argument("--weight-decay", type=float)
        item.add_argument("--hidden-layers", help="Comma-separated widths, for example 128,64,32")
        item.add_argument("--precision", choices=["auto", "float64", "float32", "float16", "bfloat16"])
        item.add_argument("--cpu", type=int, help="CPU limit in percent")
        item.add_argument("--ram", type=int, help="RAM limit in percent")
        item.add_argument("--gpu", type=int, help="GPU limit in percent")
        item.add_argument("--data-workers", type=int)
        item.add_argument("--prefetch", type=int)
        item.add_argument("--seed", type=int)
        item.add_argument("--evaluation-interval", type=int, help="Steps between validation checks")
        item.add_argument("--min-improvement", type=float, help="Minimum score gain that counts as progress")
        item.add_argument("--degradation-margin", type=float, help="Allowed score drop from the best checkpoint")
        item.add_argument("--degradation-patience", type=int, help="Consecutive bad validations before rollback")
        item.add_argument("--early-stopping-patience", type=int, help="Validations without progress before stop")
        item.add_argument("--target-patience", type=int, help="Consecutive target hits required to stop")

    plan_command = commands.add_parser("plan", help="Inspect data and print the resolved automatic training plan")
    add_training_options(plan_command)
    train_command = commands.add_parser("train", help="Train directly; no daemon or web UI is required")
    add_training_options(train_command)
    train_command.add_argument("--name")
    finetune_command = commands.add_parser("finetune", help="Fine-tune a safe weights/model file")
    finetune_command.add_argument("model")
    add_training_options(finetune_command)
    finetune_command.add_argument("--name")
    rpc = commands.add_parser("rpc", help="Serve the language-neutral JSONL protocol over stdio")
    rpc.add_argument("--config")

    daemon = commands.add_parser("daemon")
    daemon_commands = daemon.add_subparsers(dest="daemon_command", required=True)
    for name in ("start", "foreground"):
        item = daemon_commands.add_parser(name)
        item.add_argument("--host", default="127.0.0.1")
        item.add_argument("--port", type=int, default=8765)
        if name == "start":
            item.add_argument("--foreground", action="store_true")
    daemon_commands.add_parser("status")
    daemon_commands.add_parser("stop")

    serve = commands.add_parser("serve", help=argparse.SUPPRESS)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    inspect_legacy = commands.add_parser("inspect", help=argparse.SUPPRESS)
    inspect_legacy.add_argument("path")

    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    for name in ("inspect", "import", "add"):
        item = dataset_commands.add_parser(name)
        item.add_argument("path")
        if name in {"import", "add"}:
            item.add_argument("--target")
    dataset_list = dataset_commands.add_parser("list")
    dataset_list.add_argument("--limit", type=int, default=100)
    dataset_list.add_argument("--offset", type=int, default=0)
    for name in ("show", "remove"):
        item = dataset_commands.add_parser(name)
        item.add_argument("dataset_id")

    run = commands.add_parser("run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    create = run_commands.add_parser("create")
    create.add_argument("--dataset", required=True)
    create.add_argument("--task", default="binary_classification", choices=["binary_classification", "multiclass_classification", "regression"])
    create.add_argument("--backend", default="numpy")
    create.add_argument("--architecture", default="auto")
    create.add_argument("--model-format", default="auto")
    create.add_argument("--name", default="new-run")
    create.add_argument("--target-metric", type=float, default=.92)
    create.add_argument("--batch-mode", choices=["auto", "manual"], default="auto")
    create.add_argument("--batch-size", type=int)
    create.add_argument("--accept-batch-risk", action="store_true")
    create.add_argument("--max-steps", type=int, default=1400)
    create.add_argument("--profile", default="auto")
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--start", action="store_true")
    for name in ("start", "pause", "resume", "stop", "show", "get", "watch", "logs", "trace", "export"):
        item = run_commands.add_parser(name)
        item.add_argument("run_id")
        if name in {"start", "resume"}:
            item.add_argument("--foreground", action="store_true")
        if name == "logs":
            item.add_argument("--limit", type=int, default=100)
        if name == "export":
            item.add_argument("--output")
    run_list = run_commands.add_parser("list")
    run_list.add_argument("--status")
    run_list.add_argument("--limit", type=int, default=100)
    run_list.add_argument("--offset", type=int, default=0)
    inference = commands.add_parser("infer", help="Run checkpoint-backed held-out prediction requests")
    inference.add_argument("run_id")
    inference.add_argument("--requests", type=int, default=200)

    native = commands.add_parser("native", help="Inspect or build the dependency-free native training core")
    native_commands = native.add_subparsers(dest="native_command", required=True)
    native_commands.add_parser("status")
    native_build = native_commands.add_parser("build")
    native_build.add_argument("--output")

    approval = commands.add_parser("approval")
    approval_commands = approval.add_subparsers(dest="approval_command", required=True)
    approval_create = approval_commands.add_parser("create")
    approval_create.add_argument("--action", required=True, choices=["run.start", "run.resume", "run.export", "dataset.import"])
    approval_create.add_argument("--resource-id", required=True)
    approval_create.add_argument("--ttl", type=int, default=300)
    approval_revoke = approval_commands.add_parser("revoke")
    approval_revoke.add_argument("token")

    mcp = commands.add_parser("mcp")
    mcp.add_argument("--config")

    config = commands.add_parser("config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate")
    config_validate.add_argument("path")

    commands.add_parser("migrate")
    repair = commands.add_parser("repair")
    repair.add_argument("--apply", action="store_true")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--apply", action="store_true")
    workspace = commands.add_parser("workspace")
    workspace_commands = workspace.add_subparsers(dest="workspace_command", required=True)
    workspace_export = workspace_commands.add_parser("export-manifest")
    workspace_export.add_argument("output")
    workspace_import = workspace_commands.add_parser("import-manifest")
    workspace_import.add_argument("path")
    return parser


def _emit(value: Any, json_mode: bool) -> None:
    if isinstance(value, str) and not json_mode:
        print(value)
        return
    print(json.dumps(value, ensure_ascii=False, indent=None if json_mode else 2, sort_keys=json_mode))


def _write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f".{destination.name}.pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, destination)
    return destination


def _daemon_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise KernelyraError("--daemon-url must look like http://127.0.0.1:8765")
    return parsed.hostname, parsed.port or 80


def _connected_client(args: argparse.Namespace, root: Path) -> DaemonClient:
    client = DaemonClient(args.daemon_url, timeout=args.timeout, api_token=args.api_token)
    try:
        client.health()
    except DaemonUnavailableError:
        if not args.autostart:
            raise
        from .daemon import start_daemon

        host, port = _daemon_host_port(args.daemon_url)
        start_daemon(root, host, port, timeout=max(2.0, args.timeout), api_token=args.api_token)
        client.health()
    parsed = urlparse(args.daemon_url)
    if parsed.hostname and parsed.hostname.lower() in {"127.0.0.1", "localhost", "::1"}:
        from .security import load_user_secret

        return DaemonClient(
            args.daemon_url,
            timeout=args.timeout,
            api_token=args.api_token,
            user_secret=load_user_secret(root),
        )
    return client


def _verify_workspace(client: DaemonClient, root: Path) -> dict[str, Any]:
    health = client.health()
    workspace = health.get("workspace")
    if workspace and Path(str(workspace)).resolve() != root:
        raise KernelyraError(f"Daemon uses another workspace: {workspace}")
    return health


def _watch(client: DaemonClient, run_id: str, json_mode: bool) -> int:
    while True:
        info = client.get_run(run_id)
        if json_mode:
            _emit({key: info.get(key) for key in ("id", "status", "step", "max_steps", "best_score", "batch_size")}, True)
        else:
            print(
                f"{info['status']:18} step={info['step']}/{info['max_steps']} "
                f"score={float(info['best_score']):.3f} batch={info['batch_size']}",
                flush=True,
            )
        if info["status"] in TERMINAL_STATES:
            return 0
        time.sleep(1)


def _wait_for_status(client: DaemonClient, run_id: str, expected: set[str], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, timeout)
    latest = client.get_run(run_id)
    while latest["status"] not in expected and time.monotonic() < deadline:
        time.sleep(.1)
        latest = client.get_run(run_id)
    if latest["status"] not in expected:
        raise KernelyraError(
            f"Run {run_id} did not reach {', '.join(sorted(expected))}; current state is {latest['status']}"
        )
    return latest


def _doctor(root: Path) -> dict[str, Any]:
    from .backends.registry import BackendRegistry
    from .capabilities import CapabilityRegistry
    from .hardware import detect_hardware, recommend_profile
    from .ingestion.registry import IngestorRegistry

    hardware = detect_hardware()
    capabilities = CapabilityRegistry(
        BackendRegistry(),
        IngestorRegistry(),
    ).snapshot()
    supported_python = sys.version_info[:2] in {(3, 11), (3, 12), (3, 13)}
    supported_platform = platform.system().lower() == "windows"
    supported_machine = platform.machine().lower() in {"amd64", "x86_64"}
    checks = {
        "python_supported": supported_python,
        "platform_supported": supported_platform,
        "architecture_supported": supported_machine,
        "workspace_exists": root.exists(),
        "workspace_writable": os.access(root if root.exists() else root.parent, os.W_OK),
        "core_backend_available": any(
            item["name"] in {"native", "numpy"} and item["available"]
            for item in capabilities["backends"]
        ),
        "native_core_available": any(
            item["name"] == "native" and item["available"] for item in capabilities["backends"]
        ),
    }
    required_checks = {key: value for key, value in checks.items() if key != "native_core_available"}
    warnings = []
    if not checks["native_core_available"]:
        warnings.append("Native acceleration is unavailable; NumPy remains the Windows fallback")
    if not supported_platform:
        warnings.append("This release supports Windows only")
    if not supported_machine:
        warnings.append("This CPU architecture is outside the tested release matrix")
    return {
        "ok": all(required_checks.values()),
        "version": VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "checks": checks,
        "support_matrix": {
            "operating_systems": ["Windows"],
            "tested_cpu_architectures": ["x86_64/AMD64"],
            "python_versions": ["3.11", "3.12", "3.13"],
            "fallback": "NumPy on Windows when the native core or heavy backend is unavailable",
            "guarantee": "Windows x64 tested matrix only; every device cannot be guaranteed",
        },
        "warnings": warnings,
        "recommended_profile": recommend_profile(hardware),
        "hardware": hardware,
        "optional_dependencies": capabilities["optional_dependencies"],
    }


def _local_command(args: argparse.Namespace, root: Path) -> tuple[bool, Any]:
    if args.command == "version":
        return True, {"version": VERSION} if args.json else VERSION
    if args.command == "doctor":
        return True, _doctor(root)
    if args.command == "capabilities":
        from .backends.registry import BackendRegistry
        from .capabilities import CapabilityRegistry
        from .ingestion.registry import IngestorRegistry

        return True, CapabilityRegistry(
            BackendRegistry(),
            IngestorRegistry(),
        ).snapshot()
    if args.command == "formats":
        from .architectures import CHECKPOINT_FORMATS, describe_architectures
        from .formats import describe_formats, format_counts
        from .ingestion.registry import IngestorRegistry
        from .ingestion.router import FormatRouter

        ingestors = IngestorRegistry().describe()
        return True, {
            "recognized_routes": FormatRouter.route_count,
            "format_counts": format_counts(),
            "trainable_extensions": sorted(
                {extension for item in ingestors if item["available"] for extension in item["extensions"]}
            ),
            "ingestors": ingestors,
            "formats": describe_formats(),
            "architectures": describe_architectures(),
            "model_formats": list(CHECKPOINT_FORMATS),
            "contract": "recognized != extractable != directly trainable; every item reports its actual level",
        }
    if args.command == "modes":
        from .hardware import EXECUTION_MODES, detect_hardware, execution_policy, recommend_profile

        hardware = detect_hardware()
        recommended = recommend_profile(hardware)
        policy = execution_policy(recommended, hardware)
        return True, {
            "recommended_profile": recommended,
            "recommended_mode": policy["mode"],
            "gpu_available": bool(hardware["gpu_available"]),
            "accelerators": [*hardware["nvidia_gpus"], *hardware["accelerators"]],
            "modes": {
                name: {
                    "label": item["label"],
                    "data_workers": item["data_workers"],
                    "prefetch": item["prefetch"],
                    "stream_limit_mb": None if item["stream_limit"] >= 2**60 else item["stream_limit"] // 1024**2,
                    "cpu_backends": list(item["cpu_backends"]),
                    "gpu_backends": list(item["gpu_backends"]),
                    "native_thread_fraction": item["native_thread_fraction"],
                    "bulk_step_cap": item["bulk_step_cap"],
                    "arena_mb": item["arena_bytes"] // 1024**2,
                    "strategy": item["strategy"],
                }
                for name, item in EXECUTION_MODES.items()
            },
        }
    if args.command == "tune":
        from .hardware import detect_hardware
        from .tuning import autotune_execution

        return True, autotune_execution(
            args.profile,
            detect_hardware(),
            records=args.records,
            features=args.features,
            batch_size=args.batch_size,
            streaming=args.streaming,
        )
    if args.command == "chunk-plan":
        from .planning import ContextChunkPlanner

        planner = ContextChunkPlanner(
            target_records=args.target_records,
            minimum_records=args.minimum_records,
            maximum_records=args.maximum_records,
            seed=args.seed,
        )
        return True, planner.summary(args.records)
    if args.command == "native":
        from .native_core import build_native_core, native_core_status

        if args.native_command == "build":
            output = build_native_core(args.output)
            return True, {**native_core_status(), "built": str(output)}
        return True, native_core_status()
    if args.command in {"plan", "train", "finetune"}:
        from .auto import AutoTrainer

        names = (
            "target", "task", "backend", "architecture", "model_format", "profile", "batch_size", "max_steps", "target_metric",
            "learning_rate", "weight_decay", "hidden_layers", "precision", "cpu", "ram", "gpu",
            "data_workers", "prefetch", "seed", "evaluation_interval", "min_improvement",
            "degradation_margin", "degradation_patience", "early_stopping_patience",
            "target_patience", "accept_batch_risk", "name",
        )
        options = {name: getattr(args, name, None) for name in names if getattr(args, name, None) is not None}
        with AutoTrainer(root, config=args.config) as trainer:
            if args.command == "plan":
                return True, trainer.plan(args.dataset, **options).to_dict()

            def progress(run: Any) -> None:
                if not args.json:
                    print(
                        f"\r{run.status:18} step={run.step}/{run.max_steps} "
                        f"score={run.best_score:.4f} batch={run.batch_size}",
                        end="" if run.status not in TERMINAL_STATES else "\n",
                        flush=True,
                    )

            if args.command == "finetune":
                result = trainer.finetune(args.model, args.dataset, progress=progress, **options)
            else:
                result = trainer.train(args.dataset, progress=progress, **options)  # type: ignore[arg-type]
            return True, result.to_dict()
    if args.command == "infer":
        from .workspace import Workspace

        with Workspace.open(root) as workspace:
            return True, workspace.inference_check(args.run_id, args.requests)
    if args.command == "config":
        from .maintenance import validate_config

        return True, validate_config(args.path)
    if args.command == "migrate":
        from .maintenance import inspect_workspace, migrate_workspace

        if any(item["kind"] == "live_daemon" for item in inspect_workspace(root)["findings"]):
            raise KernelyraError("Stop the workspace daemon before migration")
        return True, migrate_workspace(root)
    if args.command == "repair":
        from .maintenance import repair_workspace

        return True, repair_workspace(root, apply=args.apply)
    if args.command == "cleanup":
        from .maintenance import cleanup_workspace

        return True, cleanup_workspace(root, apply=args.apply)
    if args.command == "workspace":
        from .maintenance import export_workspace_manifest, import_workspace_manifest, inspect_workspace

        if any(item["kind"] == "live_daemon" for item in inspect_workspace(root)["findings"]):
            raise KernelyraError("Stop the workspace daemon before manifest maintenance")
        if args.workspace_command == "export-manifest":
            output = export_workspace_manifest(root, args.output)
            return True, {"ok": True, "output": str(output)}
        return True, import_workspace_manifest(root, args.path)
    return False, None


def _main(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve()
    if args.command == "rpc":
        from .protocol import run_stdio

        return run_stdio(root, args.config)
    handled, result = _local_command(args, root)
    if handled:
        _emit(result, args.json)
        return 0

    if args.command == "daemon":
        if args.daemon_command in {"start", "foreground"}:
            foreground = args.daemon_command == "foreground" or getattr(args, "foreground", False)
            if foreground:
                from .server import serve

                serve(root, args.host, args.port, api_token=args.api_token)
                return 0
            from .daemon import start_daemon

            _emit(start_daemon(root, args.host, args.port, timeout=max(2.0, args.timeout), api_token=args.api_token), args.json)
            return 0
        if args.daemon_command == "status":
            client = _connected_client(args, root)
            _emit(_verify_workspace(client, root), args.json)
            client.close()
            return 0
        from .daemon import stop_daemon

        _emit(stop_daemon(root, args.daemon_url, args.api_token), args.json)
        return 0

    if args.command == "serve":
        from .server import serve

        serve(root, args.host, args.port, api_token=args.api_token)
        return 0

    if args.command == "mcp":
        from .mcp_server import run_mcp

        run_mcp(root, args.daemon_url, args.config, args.api_token)
        return 0

    client = _connected_client(args, root)
    try:
        _verify_workspace(client, root)
        if args.command == "inspect":
            _emit(client.inspect(args.path), args.json)
            return 0
        if args.command == "dataset":
            value: Any
            if args.dataset_command == "inspect":
                value = client.inspect(args.path)
            elif args.dataset_command in {"import", "add"}:
                value = client.add_dataset(args.path, args.target)
            elif args.dataset_command == "list":
                value = client.list_datasets(args.limit, args.offset)
            elif args.dataset_command == "show":
                value = client.get_dataset(args.dataset_id)
            else:
                client.remove_dataset(args.dataset_id)
                value = {"removed": True, "dataset_id": args.dataset_id}
            _emit(value, args.json)
            return 0
        if args.command == "approval":
            if args.approval_command == "create":
                from .security import load_user_secret

                value = client.issue_approval(args.action, args.resource_id, args.ttl, load_user_secret(root))
            else:
                value = client.revoke_approval(args.token)
            _emit(value, args.json)
            return 0

        if args.run_command == "create":
            created = client.create_run(
                RunConfig(
                    dataset=args.dataset,
                    backend=args.backend,
                    objective=args.task,
                    architecture=args.architecture,
                    model_format=args.model_format,
                    name=args.name,
                    target_metric=args.target_metric,
                    batch_mode=args.batch_mode,
                    batch_size=args.batch_size,
                    accept_batch_risk=args.accept_batch_risk,
                    max_steps=args.max_steps,
                    profile=args.profile,
                    seed=args.seed,
                )
            )
            value = client.command(created["id"], "start") if args.start else created
            _emit(value, args.json)
            return 0
        if args.run_command == "list":
            _emit(client.list_runs(args.limit, args.offset, args.status), args.json)
            return 0
        if args.run_command in {"show", "get"}:
            _emit(client.get_run(args.run_id), args.json)
            return 0
        if args.run_command == "watch":
            return _watch(client, args.run_id, args.json)
        if args.run_command == "logs":
            _emit(client.get_run_logs(args.run_id, args.limit), args.json)
            return 0
        if args.run_command == "trace":
            from .trace import TrainingTrace

            metrics = client.get_run_metrics(args.run_id)
            trace = TrainingTrace.from_metrics(metrics.get("metrics", metrics))
            _emit({"run_id": args.run_id, "summary": trace.summary(), "trace": trace.events}, args.json)
            return 0
        if args.run_command == "export":
            exported = client.export_run(args.run_id)
            output = args.output or f"kernelyra-run-{args.run_id}.json"
            destination = _write_json(output, exported)
            _emit({"run_id": args.run_id, "output": str(destination), "contract_version": exported["contract_version"]}, args.json)
            return 0
        value = client.command(args.run_id, args.run_command)
        if args.run_command == "pause" and value["status"] == "pausing":
            value = _wait_for_status(client, args.run_id, {"paused"}, args.timeout)
        if args.run_command == "stop" and value["status"] == "stopping":
            value = _wait_for_status(client, args.run_id, {"stopped", "error_recoverable"}, args.timeout)
        _emit(value, args.json)
        if getattr(args, "foreground", False):
            return _watch(client, args.run_id, args.json)
        return 0
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in raw
    raw = [item for item in raw if item != "--json"]
    args = _parser().parse_args(raw)
    args.json = json_mode
    try:
        return _main(args)
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrupted", "type": "KeyboardInterrupt"}) if args.json else "Interrupted", file=sys.stderr)
        return 130
    except DaemonUnavailableError as error:
        _emit_error(error, args.json)
        return EXIT_UNAVAILABLE
    except RemoteError as error:
        _emit_error(error, args.json)
        return EXIT_AUTHORIZATION if error.status in {401, 403} else EXIT_EXPECTED_ERROR
    except (KernelyraError, OSError, ValueError) as error:
        _emit_error(error, args.json)
        return EXIT_EXPECTED_ERROR


def _emit_error(error: Exception, json_mode: bool) -> None:
    payload: dict[str, Any] = {"error": str(error), "type": type(error).__name__}
    if isinstance(error, RemoteError):
        payload["status"] = error.status
        payload["remote_type"] = error.error_type
    print(json.dumps(payload, ensure_ascii=False, sort_keys=json_mode) if json_mode else f"Error: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
