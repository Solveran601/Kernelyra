"""Stable, side-effect-free public Kernelyra SDK surface."""

from .async_client import AsyncKernelyraClient
from .auto import AutoTrainer, TrainingPlan, TrainingResult, finetune, plan, train
from .client import DaemonClient, RemoteError
from .data_health import analyze_inspection, inspect_path, recommend_chunk_policy
from .easy import Config, Engine, Settings, TrainingConfig, fit
from .errors import (
    AccessDeniedError,
    ApprovalError,
    ConfigurationError,
    DaemonUnavailableError,
    DatasetError,
    DatasetNotFoundError,
    KernelyraError,
    RunError,
    RunNotFoundError,
    RunStateError,
    WorkerCrashedError,
    WorkerError,
    WorkerProtocolError,
    WorkerTimeoutError,
)
from .extraction import TextChunk, extract_folder, extract_text, text_format_count
from .format_intelligence import advise_path
from .inference import run_inference_check
from .models import (
    BackendInfo,
    DatasetInfo,
    DatasetManifest,
    DatasetSchema,
    IngestorInfo,
    RunConfig,
    RunInfo,
    RunMetrics,
    RunStatus,
    TaskType,
)
from .native_core import NativeTensorArena
from .planning import ContextChunk, ContextChunkPlanner
from .quality import QualityGate
from .reports import build_experiment_report, write_experiment_report
from .tuning import autotune_execution
from .workspace import Kernelyra, RunHandle, Workspace

__version__ = "0.5.0a1"

Dataset = DatasetInfo
Run = RunInfo
KernelyraClient = DaemonClient

__all__ = [
    "AccessDeniedError",
    "ApprovalError",
    "AsyncKernelyraClient",
    "AutoTrainer",
    "analyze_inspection",
    "BackendInfo",
    "ConfigurationError",
    "Config",
    "ContextChunk",
    "ContextChunkPlanner",
    "DaemonClient",
    "DaemonUnavailableError",
    "Dataset",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetInfo",
    "DatasetManifest",
    "DatasetSchema",
    "IngestorInfo",
    "Engine",
    "RemoteError",
    "QualityGate",
    "Run",
    "RunConfig",
    "RunError",
    "RunNotFoundError",
    "RunHandle",
    "RunInfo",
    "RunMetrics",
    "RunStateError",
    "RunStatus",
    "Settings",
    "TaskType",
    "TextChunk",
    "Kernelyra",
    "KernelyraClient",
    "KernelyraError",
    "NativeTensorArena",
    "TrainingPlan",
    "TrainingConfig",
    "TrainingResult",
    "WorkerCrashedError",
    "WorkerError",
    "WorkerProtocolError",
    "WorkerTimeoutError",
    "Workspace",
    "finetune",
    "extract_folder",
    "extract_text",
    "fit",
    "build_experiment_report",
    "inspect_path",
    "recommend_chunk_policy",
    "run_inference_check",
    "text_format_count",
    "plan",
    "train",
    "write_experiment_report",
    "autotune_execution",
    "advise_path",
]
