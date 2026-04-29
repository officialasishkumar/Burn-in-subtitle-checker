"""Project-specific exceptions with stable CLI exit codes."""


class BurnSubError(Exception):
    """Base class for user-facing processing errors."""

    exit_code = 4


class ConfigError(BurnSubError):
    """Invalid command line input or configuration."""

    exit_code = 2


class MissingDependencyError(BurnSubError):
    """Required executable, Python package, model, or language data is missing."""

    exit_code = 3


class ProcessingError(BurnSubError):
    """A processing stage failed after validation."""

    exit_code = 4
