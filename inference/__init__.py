"""Open-set inference: base classes, registered classes, and unknown signs.

Only the dependency-light decision policy and registration types are re-exported
here. :class:`inference.pipeline.OpenSetRecognizer` needs torch and must be
imported from its own module, so this package stays importable in environments
without a deep-learning stack.
"""

from inference.decision import (
    BaseEvidence,
    OpenSetDecision,
    OpenSetDecisionError,
    OpenSetThresholds,
    PrototypeEvidence,
    Strategy,
    Verdict,
    build_base_evidence,
    build_prototype_evidence,
    decide,
    decide_from_scores,
    softmax,
)
from inference.registration import (
    IncrementalRegistrar,
    ReferenceCoherence,
    RegistrationError,
    RegistrationPolicy,
    RegistrationResult,
    measure_coherence,
)

__all__ = [
    "BaseEvidence",
    "IncrementalRegistrar",
    "OpenSetDecision",
    "OpenSetDecisionError",
    "OpenSetThresholds",
    "PrototypeEvidence",
    "ReferenceCoherence",
    "RegistrationError",
    "RegistrationPolicy",
    "RegistrationResult",
    "Strategy",
    "Verdict",
    "build_base_evidence",
    "build_prototype_evidence",
    "decide",
    "decide_from_scores",
    "measure_coherence",
    "softmax",
]
