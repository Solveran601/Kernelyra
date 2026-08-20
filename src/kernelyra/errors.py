class KernelyraError(Exception):
    """Base error raised by the public SDK."""


class ConfigurationError(KernelyraError):
    pass


class DatasetError(KernelyraError):
    pass


class StorageError(KernelyraError):
    pass


class NotFoundError(KernelyraError):
    pass


class DatasetNotFoundError(DatasetError, NotFoundError):
    pass


class PayloadTooLargeError(KernelyraError):
    pass


class RateLimitError(KernelyraError):
    pass


class RunError(KernelyraError):
    pass


class RunNotFoundError(RunError, NotFoundError):
    pass


class RunStateError(RunError):
    """Raised when a lifecycle command is invalid for the current run state."""

    def __init__(self, run_id: str, command: str, status: str, allowed: tuple[str, ...]):
        self.run_id = run_id
        self.command = command
        self.status = status
        self.allowed = allowed
        expected = ", ".join(allowed) if allowed else "нет допустимых состояний"
        super().__init__(
            f"Команда '{command}' недопустима для запуска {run_id} в состоянии "
            f"'{status}'. Допустимо: {expected}"
        )


class AccessDeniedError(KernelyraError):
    pass


class ApprovalError(AccessDeniedError):
    pass


class DaemonUnavailableError(KernelyraError):
    pass


class WorkerError(RunError):
    pass


class WorkerProtocolError(WorkerError):
    pass


class WorkerCrashedError(WorkerError):
    pass


class WorkerTimeoutError(WorkerError):
    pass
