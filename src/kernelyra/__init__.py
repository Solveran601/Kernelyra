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
from .packs import (
    PACK_ALGORITHMS,
    add_pack_algorithm,
    algorithm_pack_path,
    algorithm_pack_table,
    create_algorithm_pack,
    delete_algorithm_pack,
    get_algorithm_pack,
    list_algorithm_packs,
    remove_pack_algorithm,
)
from .planning import ContextChunk, ContextChunkPlanner
from .quality import QualityGate
from .reports import build_experiment_report, write_experiment_report
from .text_training import (
    ByteTokenizer,
    MaskedTextBatch,
    MaskedTextExample,
    batch_masked_text_examples,
    iter_masked_text_batches,
    plan_text_for_training,
    prepare_masked_text_examples,
)
from .tuning import autotune_execution
from .workspace import Kernelyra, RunHandle, Workspace

__version__ = "0.5.0a3"

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
    "ByteTokenizer",
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
    "PACK_ALGORITHMS",
    "MaskedTextBatch",
    "MaskedTextExample",
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
    "add_pack_algorithm",
    "algorithm_pack_path",
    "algorithm_pack_table",
    "advise_path",
    "batch_masked_text_examples",
    "iter_masked_text_batches",
    "plan_text_for_training",
    "prepare_masked_text_examples",
    "create_algorithm_pack",
    "delete_algorithm_pack",
    "get_algorithm_pack",
    "list_algorithm_packs",
    "remove_pack_algorithm",
]
