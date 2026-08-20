from __future__ import annotations

import unittest
from typing import Any

from kernelyra.server import create_app
from tests.helpers import isolated_workspace

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
REQUEST_MODELS = (
    "AgentSessionRequest",
    "ApprovalRequest",
    "BatchPlanRequest",
    "CommandRequest",
    "MCPCommandRequest",
    "MCPDatasetRequest",
    "MCPExportRequest",
    "PathRequest",
    "RunRequest",
    "TokenRequest",
)

# This semantic snapshot intentionally ignores descriptions, titles, generated
# JSON-Schema dialect details and ordering that vary across FastAPI/Pydantic.
EXPECTED_OPERATIONS: tuple[tuple[Any, ...], ...] = (
    ("/api/v1/agent-sessions", "GET", (), False, (), ("200",)),
    ("/api/v1/agent-sessions/{session_id}", "DELETE", (("path", "session_id", True),), False, (), ("200", "422")),
    ("/api/v1/approvals", "POST", (), True, (("application/json", "ApprovalRequest"),), ("201", "422")),
    ("/api/v1/approvals/revoke", "POST", (), True, (("application/json", "TokenRequest"),), ("200", "422")),
    ("/api/v1/batch/plan", "POST", (), True, (("application/json", "BatchPlanRequest"),), ("200", "422")),
    ("/api/v1/capabilities", "GET", (), False, (), ("200",)),
    ("/api/v1/datasets", "GET", (("query", "limit", False), ("query", "offset", False)), False, (), ("200", "422")),
    ("/api/v1/datasets", "POST", (("query", "target", False),), True, (("multipart/form-data", "UploadDatasetBody"),), ("201", "422")),
    ("/api/v1/datasets/from-path", "POST", (), True, (("application/json", "PathRequest"),), ("201", "422")),
    ("/api/v1/datasets/{dataset_id}", "DELETE", (("path", "dataset_id", True),), False, (), ("204", "422")),
    ("/api/v1/datasets/{dataset_id}", "GET", (("path", "dataset_id", True),), False, (), ("200", "422")),
    ("/api/v1/events", "GET", (("query", "after", False),), False, (), ("200", "422")),
    ("/api/v1/hardware", "GET", (), False, (), ("200",)),
    ("/api/v1/health", "GET", (), False, (), ("200",)),
    ("/api/v1/logs", "GET", (("query", "limit", False),), False, (), ("200", "422")),
    ("/api/v1/mcp/capabilities", "GET", (), False, (), ("200",)),
    ("/api/v1/mcp/datasets", "GET", (("query", "limit", False), ("query", "offset", False)), False, (), ("200", "422")),
    ("/api/v1/mcp/datasets/from-path", "POST", (), True, (("application/json", "MCPDatasetRequest"),), ("201", "422")),
    ("/api/v1/mcp/datasets/{dataset_id}", "GET", (("path", "dataset_id", True),), False, (), ("200", "422")),
    ("/api/v1/mcp/hardware", "GET", (), False, (), ("200",)),
    ("/api/v1/mcp/logs", "GET", (("query", "limit", False),), False, (), ("200", "422")),
    ("/api/v1/mcp/paths/inspect", "POST", (), True, (("application/json", "PathRequest"),), ("200", "422")),
    ("/api/v1/mcp/runs", "GET", (("query", "limit", False), ("query", "offset", False)), False, (), ("200", "422")),
    ("/api/v1/mcp/runs", "POST", (), True, (("application/json", "RunRequest"),), ("201", "422")),
    ("/api/v1/mcp/runs/{run_id}", "GET", (("path", "run_id", True),), False, (), ("200", "422")),
    ("/api/v1/mcp/runs/{run_id}/command", "POST", (("path", "run_id", True),), True, (("application/json", "MCPCommandRequest"),), ("200", "422")),
    ("/api/v1/mcp/runs/{run_id}/export", "POST", (("path", "run_id", True),), True, (("application/json", "MCPExportRequest"),), ("200", "422")),
    ("/api/v1/mcp/runs/{run_id}/logs", "GET", (("path", "run_id", True), ("query", "limit", False)), False, (), ("200", "422")),
    ("/api/v1/mcp/runs/{run_id}/metrics", "GET", (("path", "run_id", True),), False, (), ("200", "422")),
    ("/api/v1/mcp/sessions", "POST", (), True, (("application/json", "AgentSessionRequest"),), ("201", "422")),
    ("/api/v1/model-formats", "GET", (), False, (), ("200",)),
    ("/api/v1/paths/inspect", "POST", (), True, (("application/json", "PathRequest"),), ("200", "422")),
    ("/api/v1/rebalance", "POST", (), False, (), ("200",)),
    ("/api/v1/runs", "GET", (("query", "limit", False), ("query", "offset", False), ("query", "status", False)), False, (), ("200", "422")),
    ("/api/v1/runs", "POST", (), True, (("application/json", "RunRequest"),), ("201", "422")),
    ("/api/v1/runs/{run_id}", "GET", (("path", "run_id", True),), False, (), ("200", "422")),
    ("/api/v1/runs/{run_id}/command", "POST", (("path", "run_id", True),), True, (("application/json", "CommandRequest"),), ("200", "422")),
    ("/api/v1/runs/{run_id}/export", "GET", (("path", "run_id", True),), False, (), ("200", "422")),
    ("/api/v1/runs/{run_id}/logs", "GET", (("path", "run_id", True), ("query", "limit", False)), False, (), ("200", "422")),
    ("/api/v1/runs/{run_id}/metrics", "GET", (("path", "run_id", True),), False, (), ("200", "422")),
    ("/api/v1/state", "GET", (), False, (), ("200",)),
)
EXPECTED_REQUEST_MODELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "AgentSessionRequest": (("client_id", "ttl_seconds"), ("client_id",)),
    "ApprovalRequest": (("action", "resource_id", "ttl_seconds"), ("action", "resource_id")),
    "BatchPlanRequest": (("batch_mode", "batch_size", "dataset", "profile", "ram"), ()),
    "CommandRequest": (("command",), ("command",)),
    "MCPCommandRequest": (("approval_token", "command"), ("command",)),
    "MCPDatasetRequest": (("approval_token", "path", "target"), ("approval_token", "path")),
    "MCPExportRequest": (("approval_token",), ("approval_token",)),
    "PathRequest": (("path", "target"), ("path",)),
    "RunRequest": (
        (
            "accept_batch_risk",
            "architecture",
            "backend",
            "base_run_id",
            "batch_mode",
            "batch_size",
            "cpu",
            "data_workers",
            "dataset",
            "degradation_margin",
            "degradation_patience",
            "early_stopping_patience",
            "evaluation_interval",
            "gpu",
            "hidden_layers",
            "learning_rate",
            "max_steps",
            "min_improvement",
            "mode",
            "model_format",
            "model_path",
            "name",
            "objective",
            "precision",
            "prefetch",
            "priority",
            "profile",
            "ram",
            "seed",
            "start",
            "target_patience",
            "target_score",
            "weight_decay",
        ),
        (),
    ),
    "TokenRequest": (("token",), ("token",)),
}


def _model_name(schema: dict[str, Any]) -> str:
    reference = schema.get("$ref", "")
    if not reference and len(schema.get("allOf", [])) == 1:
        reference = schema["allOf"][0].get("$ref", "")
    name = reference.rsplit("/", 1)[-1] if reference else "inline"
    return "UploadDatasetBody" if name.startswith("Body_upload_dataset") else name


def semantic_contract(schema: dict[str, Any]) -> tuple[tuple[tuple[Any, ...], ...], dict[str, Any]]:
    operations: list[tuple[Any, ...]] = []
    for path, path_item in schema["paths"].items():
        if not path.startswith("/api/v1/"):
            continue
        for method, operation in path_item.items():
            if method not in HTTP_METHODS:
                continue
            parameters = tuple(
                sorted(
                    (item.get("in"), item.get("name"), bool(item.get("required")))
                    for item in operation.get("parameters", [])
                )
            )
            request_body = operation.get("requestBody", {})
            body = tuple(
                sorted(
                    (content_type, _model_name(media.get("schema", {})))
                    for content_type, media in request_body.get("content", {}).items()
                )
            )
            responses = tuple(sorted(operation.get("responses", {})))
            operations.append(
                (path, method.upper(), parameters, bool(request_body.get("required")), body, responses)
            )

    schemas = schema.get("components", {}).get("schemas", {})
    models = {
        name: (
            tuple(sorted(schemas[name].get("properties", {}))),
            tuple(sorted(schemas[name].get("required", []))),
        )
        for name in REQUEST_MODELS
    }
    return tuple(sorted(operations)), models


class OpenAPISnapshotTests(unittest.TestCase):
    maxDiff = None

    def test_v1_semantic_contract(self) -> None:
        with isolated_workspace() as temporary:
            actual_operations, actual_models = semantic_contract(
                create_app(temporary / "project").openapi()
            )
            self.assertEqual(actual_operations, EXPECTED_OPERATIONS)
            self.assertEqual(actual_models, EXPECTED_REQUEST_MODELS)


if __name__ == "__main__":
    unittest.main()
