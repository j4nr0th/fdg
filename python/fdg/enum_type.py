"""Definitions for enum types used to match some string literals."""

from enum import StrEnum
from typing import Literal


class BasisType(StrEnum):
    """Type of basis that can be used for basis sets."""

    LEGENDRE = "legendre"
    BERNSTEIN = "bernstein"
    LAGRANGE_UNIFORM = "lagrange-uniform"
    LAGRNAGE_GAUSS = "lagrange-gauss"
    LAGRNAGE_GAUSS_LOBATTO = "lagrange-gauss-lobatto"
    LAGRANGE_CHEBYSHEV_GAUSS = "lagrange-chebyshev-gauss"


_BasisTypeHint = (
    BasisType
    | Literal[
        "legendre",
        "bernstein",
        "lagrange-uniform",
        "lagrange-gauss",
        "lagrange-gauss-lobatto",
        "lagrange-chebyshev-gauss",
    ]
)


class IntegrationMethod(StrEnum):
    """Methods of integration which are supported."""

    GAUSS = "gauss"
    GAUSS_LOBATTO = "gauss-lobatto"


_IntegrationMethodHint = (
    IntegrationMethod
    | Literal[
        "gauss",
        "gauss-lobatto",
    ]
)
