"""Stable, side-effect-free public Kernelyra SDK surface."""

from .async_client import AsyncKernelyraClient
from .auto import AutoTrainer, TrainingPlan, TrainingResult, finetune, plan, train
from .client import DaemonClient, RemoteError
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
from .planning import ContextChunk, ContextChunkPlanner
from .quality import QualityGate
from .tuning import autotune_execution
from .native_core import NativeTensorArena
from .workspace import Kernelyra, RunHandle, Workspace

__version__ = "0.3.0a1"

Dataset = DatasetInfo
Run = RunInfo
KernelyraClient = DaemonClient

__all__ = [
    "AccessDeniedError",
    "ApprovalError",
    "AsyncKernelyraClient",
    "AutoTrainer",
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
    "run_inference_check",
    "text_format_count",
    "plan",
    "train",
    "autotune_execution",
    "advise_path",
]
