"""Shared library for the Pmx reconstruction scripts.

Pipeline-specific by design: this is what the four Pmx_reconstruction scripts
share with each other and with nothing else. Anything here that turns out to be
useful to HATF as well should be promoted to the top-level utils/ rather than
imported across pipelines.

Deliberately free of re-exports: import from the leaf module, so that pulling in
one piece never drags CAMB or colossus in by side effect.
"""

# =============================================
#                   VARIABLES
# =============================================

# The mass fractions definitions:
#
#   f_i          per-bin catalogue mass fraction, n_i M_i / rhobar_m
#   f_part       per-bin mass actually inside the membership spheres
#                (\tilde f_i in the notes)
#   f_u          the catalogue deficit, 1 - sum_i f_i
#   f_out        the measured deficit, 1 - sum_i f_part  (\tilde f_u)
#   f_smallhalo  component (i): the integral of M n(M) below M_r, written f_(i)
#                in the notes. NOT called f_i in code, because that name is
#                taken by the per-bin fraction above.