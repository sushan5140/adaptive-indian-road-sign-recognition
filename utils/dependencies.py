"""Clear errors for optional machine-learning dependencies."""


class DependencyUnavailableError(RuntimeError):
    """Raised when an optional runtime dependency is required but unavailable."""
