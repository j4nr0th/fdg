"""Package dedicated to interpolation using data defined on different topologies."""

# C module interface
from interplib._interp import DEFAULT_BASIS_REGISTRY as DEFAULT_BASIS_REGISTRY
from interplib._interp import DEFAULT_INTEGRATION_REGISTRY as DEFAULT_INTEGRATION_REGISTRY
from interplib._interp import BasisRegistry as BasisRegistry
from interplib._interp import BasisSpecs as BasisSpecs
from interplib._interp import CoordinateMap as CoordinateMap
from interplib._interp import CovectorBasis as CovectorBasis
from interplib._interp import DegreesOfFreedom as DegreesOfFreedom
from interplib._interp import FunctionSpace as FunctionSpace
from interplib._interp import IntegrationRegistry as IntegrationRegistry
from interplib._interp import IntegrationSpace as IntegrationSpace
from interplib._interp import IntegrationSpecs as IntegrationSpecs
from interplib._interp import KForm as KForm
from interplib._interp import KFormSpecs as KFormSpecs
from interplib._interp import SpaceMap as SpaceMap
from interplib._interp import compute_gradient_mass_matrix as compute_gradient_mass_matrix
from interplib._interp import (
    compute_kform_incidence_matrix as compute_kform_incidence_matrix,
)
from interplib._interp import (
    compute_kform_interior_product_matrix as compute_kform_interior_product_matrix,
)
from interplib._interp import compute_kform_mass_matrix as compute_kform_mass_matrix
from interplib._interp import compute_mass_matrix as compute_mass_matrix
from interplib._interp import incidence_kform_operator as incidence_kform_operator
from interplib._interp import incidence_matrix as incidence_matrix
from interplib._interp import incidence_operator as incidence_operator
from interplib._interp import (
    transform_contravariant_to_target as transform_contravariant_to_target,
)
from interplib._interp import (
    transform_covariant_to_target as transform_covariant_to_target,
)
from interplib._interp import (
    transform_kform_component_to_target as transform_kform_component_to_target,
)
from interplib._interp import transform_kform_to_target as transform_kform_to_target

# DoFs functions
from interplib.degrees_of_freedom import (
    compute_dual_degrees_of_freedom as compute_dual_degrees_of_freedom,
)
from interplib.degrees_of_freedom import reconstruct as reconstruct

# Domains
from interplib.domains import Line as Line
from interplib.domains import Quad as Quad

# Enum types
from interplib.enum_type import BasisType as BasisType
from interplib.enum_type import IntegrationMethod as IntegrationMethod

# Integration functions
from interplib.integration import integrate_callable as integrate_callable
from interplib.integration import projection_l2_dual as projection_l2_dual
from interplib.integration import projection_l2_primal as projection_l2_primal
