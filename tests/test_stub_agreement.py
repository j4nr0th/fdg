"""Guard the hand-written stubs in ``python/fdg/_fdg.pyi`` against the C source.

The stub is maintained by hand, so a docstring fixed on either side alone drifts
from the other. These checks compare the runtime docstrings, which the extension
takes from the C source, against the stub text and fail when a convention the
moving-mesh assembly depends on differs between them.
"""

import re
from pathlib import Path

import pytest

STUB = Path(__file__).resolve().parents[1] / "python" / "fdg" / "_fdg.pyi"


def _normalised(text: str) -> str:
    """Collapse whitespace so line wrapping does not affect the comparison."""
    return re.sub(r"\s+", " ", text).strip()


def _stub_text() -> str:
    """Return the stub source, which the package ships alongside the extension."""
    return _normalised(STUB.read_text(encoding="utf-8"))


def _strip_markup(text: str) -> str:
    """Drop RST inline markup, which the two copies need not use identically."""
    return text.replace("``", "").replace("*", "")


@pytest.mark.parametrize(
    ("qualified", "fragments"),
    [
        (
            "compute_kform_interior_product_matrix",
            [
                # the rows are (order - 1)-forms of basis_left, not 0-forms
                "(order - 1)-form components provide the test",
                "order-form components provide the trial",
                # the pairing carries the determinant, which the assembly relies on
                "cancels the determinant",
            ],
        ),
        (
            "FunctionSpace.values_at_integration_nodes",
            [
                # transpose puts the basis axes first; the default is the reverse
                "axes indexing the bases come before",
                "integration-point axes come first",
            ],
        ),
        (
            "IntegrationSpace.nodes",
            ["tensor grid of axis", "rather than a one-dimensional array"],
        ),
    ],
)
def test_stub_matches_the_c_docstring(qualified: str, fragments: list[str]) -> None:
    """Every documented convention must read the same in the stub and the C source.

    A stub that describes the default where the code implements the flag, or calls
    a k-form space a space of 0-forms, is worse than no documentation: it sends a
    caller to assemble a matrix that is the wrong shape.
    """
    import fdg

    obj = fdg
    for part in qualified.split("."):
        obj = getattr(obj, part)

    runtime = _strip_markup(_normalised(obj.__doc__ or ""))
    stub = _strip_markup(_stub_text())

    for fragment in fragments:
        needle = _strip_markup(_normalised(fragment))
        assert needle in runtime, f"{qualified}: C source is missing {fragment!r}"
        assert needle in stub, f"{qualified}: stub is missing {fragment!r}"


def test_incidence_operator_orientation_is_documented() -> None:
    """The ``right`` flag is how the assembly obtains the derivative matrix.

    ``incidence_kform_operator(..., right=True)`` returns the ``(n_{k+1}, n_k)``
    matrix used to compose the two terms of the Lie derivative, so both the stub
    and the C source must say which way it runs.
    """
    from fdg import incidence_kform_operator

    runtime = _strip_markup(_normalised(incidence_kform_operator.__doc__ or ""))
    stub = _strip_markup(_stub_text())
    for fragment in ("right", "transposed operator"):
        needle = _strip_markup(fragment)
        assert needle in runtime
        assert needle in stub
