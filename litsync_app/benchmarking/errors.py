class BenchmarkError(Exception):
    """Base class for deterministic benchmark failures."""


class SpecValidationError(BenchmarkError):
    pass


class SpecImmutabilityError(BenchmarkError):
    pass


class GoldValidationError(BenchmarkError):
    pass


class RunArtifactError(BenchmarkError):
    pass


class ComparisonError(BenchmarkError):
    pass


class PublicationError(BenchmarkError):
    pass
