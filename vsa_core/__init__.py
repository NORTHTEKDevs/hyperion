"""VSA core primitives.

Bipolar {-1,+1}^D hypervectors. Composition via:
  bind   = circular convolution (FFT)
  bundle = sum + sign
  permute = cyclic rotation
  cleanup = cosine argmax against a codebook
"""

from vsa_core.ops import bind, unbind, bundle, bundle_soft, permute, recursive_bind
from vsa_core.codebook import Codebook, fpe_positions
from vsa_core.cleanup import cleanup, similarity
from vsa_core.ste import sign_ste, soft_binarize

__all__ = [
    "bind",
    "unbind",
    "bundle",
    "bundle_soft",
    "permute",
    "recursive_bind",
    "Codebook",
    "fpe_positions",
    "cleanup",
    "similarity",
    "sign_ste",
    "soft_binarize",
]
