"""Package dedicated to interpolation using data defined on different topologies."""

# C module interface
from fdg._fdg import DEFAULT_BASIS_REGISTRY as DEFAULT_BASIS_REGISTRY
from fdg._fdg import DEFAULT_INTEGRATION_REGISTRY as DEFAULT_INTEGRATION_REGISTRY
from fdg._fdg import BasisRegistry as BasisRegistry
from fdg._fdg import BasisSpecs as BasisSpecs
from fdg._fdg import CoordinateMap as CoordinateMap
from fdg._fdg import CovectorBasis as CovectorBasis
from fdg._fdg import DegreesOfFreedom as DegreesOfFreedom
from fdg._fdg import FunctionSpace as FunctionSpace
from fdg._fdg import IntegrationRegistry as IntegrationRegistry
from fdg._fdg import IntegrationSpace as IntegrationSpace
from fdg._fdg import IntegrationSpecs as IntegrationSpecs
from fdg._fdg import KForm as KForm
from fdg._fdg import KFormSpecs as KFormSpecs
from fdg._fdg import SpaceMap as SpaceMap
from fdg._fdg import compute_gradient_mass_matrix as compute_gradient_mass_matrix
from fdg._fdg import (
    compute_kform_incidence_matrix as compute_kform_incidence_matrix,
)
from fdg._fdg import (
    compute_kform_interior_product_matrix as compute_kform_interior_product_matrix,
)
from fdg._fdg import compute_kform_mass_matrix as compute_kform_mass_matrix
from fdg._fdg import compute_mass_matrix as compute_mass_matrix
from fdg._fdg import incidence_kform_operator as incidence_kform_operator
from fdg._fdg import incidence_matrix as incidence_matrix
from fdg._fdg import incidence_operator as incidence_operator
from fdg._fdg import (
    transform_contravariant_to_target as transform_contravariant_to_target,
)
from fdg._fdg import (
    transform_covariant_to_target as transform_covariant_to_target,
)
from fdg._fdg import (
    transform_kform_component_to_target as transform_kform_component_to_target,
)
from fdg._fdg import transform_kform_to_target as transform_kform_to_target

# DoFs functions
from fdg.degrees_of_freedom import (
    compute_dual_degrees_of_freedom as compute_dual_degrees_of_freedom,
)
from fdg.degrees_of_freedom import reconstruct as reconstruct

# Domains
from fdg.domains import Line as Line
from fdg.domains import Quad as Quad

# Enum types
from fdg.enum_type import BasisType as BasisType
from fdg.enum_type import IntegrationMethod as IntegrationMethod

# Integration functions
from fdg.integration import integrate_callable as integrate_callable
from fdg.integration import projection_kform_l2_dual as projection_kform_l2_dual
from fdg.integration import projection_kform_l2_primal as projection_kform_l2_primal
from fdg.integration import projection_l2_dual as projection_l2_dual
from fdg.integration import projection_l2_primal as projection_l2_primal
