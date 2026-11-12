"""跨厂商统一故障码（UFC）目录与规范化。"""

from .models import (  # noqa: F401
    CHECKPOINT_POLICIES,
    KNOWN_ACTIONS,
    VERIFY_CHECKS,
    Confidence,
    Fault,
    FaultCode,
    FaultState,
    Recurrence,
    Scope,
    Severity,
    SignalRule,
    Target,
    Vendor,
    VendorEvent,
    VendorMapping,
)
from .normalizer import Normalizer, Translation, infer_target  # noqa: F401
from .taxonomy import Catalog, load_catalog  # noqa: F401

__all__ = [
    "Catalog",
    "Confidence",
    "Fault",
    "FaultCode",
    "FaultState",
    "Normalizer",
    "Recurrence",
    "Scope",
    "Severity",
    "SignalRule",
    "Target",
    "Translation",
    "Vendor",
    "VendorEvent",
    "VendorMapping",
    "infer_target",
    "load_catalog",
]
