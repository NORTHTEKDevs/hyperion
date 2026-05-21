"""Pure-VSA reasoning: no gradient descent, no MLP, no neural learning.

Composition by VSA algebra (bind / unbind / bundle), retrieval by codebook
cleanup, learning by outer-product memory accumulation only.
"""

from pure_vsa.memory import AssociativeMemory
from pure_vsa.composer import RuleComposer
from pure_vsa.reasoner import PureVSAReasoner
from pure_vsa.hyperion import HyperionReasoner, HyperionConfig, Example

__all__ = [
    # High-level API (recommended for new code).
    "HyperionReasoner",
    "HyperionConfig",
    "Example",
    # Low-level primitives (for research / extension).
    "AssociativeMemory",
    "RuleComposer",
    "PureVSAReasoner",
]
