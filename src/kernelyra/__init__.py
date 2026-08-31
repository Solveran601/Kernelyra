"""Stable, side-effect-free public Kernelyra SDK surface."""

from .async_client import AsyncKernelyraClient
from .auto import AutoTrainer, TrainingPlan, TrainingResult, finetune, plan, train
from .client import DaemonClient, RemoteError
from .conversation_chunks import (
    ConversationChunk,
    ConversationChunker,
    ConversationMessage,
    iter_conversation_chunks,
    iter_conversation_messages,
    iter_jsonl_messages,
    iter_plain_text_messages,
    iter_telegram_messages,
    write_conversation_chunks,
)
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
from .model_guard import ModelGuard, assess_trend
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
from .native_core import NativeTensorArena, native_core_self_test, native_core_status
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
from .workspace import Kernelyra, RunHandle, Workspace, default_library_workspace

__version__ = "0.7.0b1"

Dataset = DatasetInfo
Run = RunInfo
KernelyraClient = DaemonClient

__all__ = [
    "AccessDeniedError",
    "ApprovalError",
    "AsyncKernelyraClient",
    "AutoTrainer",
    "analyze_inspection",
    "assess_trend",
    "BackendInfo",
    "ByteTokenizer",
    "ConfigurationError",
    "Config",
    "ConversationChunk",
    "ConversationChunker",
    "ConversationMessage",
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
    "ModelGuard",
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
    "native_core_self_test",
    "native_core_status",
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
    "default_library_workspace",
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
    "batch_masked_text_examples",
    "iter_masked_text_batches",
    "iter_conversation_chunks",
    "iter_conversation_messages",
    "iter_jsonl_messages",
    "iter_plain_text_messages",
    "iter_telegram_messages",
    "plan_text_for_training",
    "prepare_masked_text_examples",
    "write_conversation_chunks",
]
