"""Work around Hawkmoth's broken rendering of variable-length array parameters.

Hawkmoth renders a parameter such as ``double out[static n]`` as
``double[static n] out``: its declarator fixup (:py:meth:`DocCursor._var_type_fixup
<hawkmoth.doccursor.DocCursor._var_type_fixup>`) moves array dimensions from the
type onto the parameter name for constant and incomplete arrays only.
Variable-length arrays (``[static n]``, ``[restrict n]``, ``[cnt]`` ...) fall
through, leaving the dimension in the type spelling, and Sphinx's C domain then
rejects the signature — the documented function is silently dropped.

This module patches ``_var_type_fixup`` to append any trailing array dimensions
that remained in the type spelling to the parameter name, e.g.
``double[static n] out`` becomes ``double out[static n]``.

Remove this module once Hawkmoth renders variable-length array parameters
correctly (upstream fix, Hawkmoth > 0.22).
"""

import re

from hawkmoth.doccursor import DocCursor

_orig_var_type_fixup = DocCursor._var_type_fixup
_array_suffix = re.compile(r"(\[[^\[\]]*\])+$")


def _var_type_fixup(cursor):
    """Wrap :py:func:`DocCursor._var_type_fixup` to fix array dimensions."""
    ttype, name = _orig_var_type_fixup(cursor)

    # The parenthesized group only matches trailing array dimensions; array
    # bounds are expressions that cannot contain nested brackets.
    ttype = ttype.rstrip()
    match = _array_suffix.search(ttype)
    if match is not None:
        ttype = ttype[: match.start()].rstrip()
        name += match.group(1)

    return ttype, name


DocCursor._var_type_fixup = staticmethod(_var_type_fixup)


def setup(app):
    """Register the extension with Sphinx."""
    return {"version": "0.1", "parallel_read_safe": True}
