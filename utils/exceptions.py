"""Custom exceptions used across the pipeline."""


class PipelineError(RuntimeError):
    """Base class for pipeline-specific failures."""


class ConfigurationError(PipelineError):
    """Raised when a configuration file is missing or invalid."""


class IngestionError(PipelineError):
    """Raised when imagery querying or download fails."""


class PreprocessingError(PipelineError):
    """Raised when geospatial preprocessing cannot complete."""
