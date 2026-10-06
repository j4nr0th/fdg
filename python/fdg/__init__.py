"""Package dedicated to interpolation using data defined on different topologies."""

# C module interface
from fdg._fdg import DEFAULT_BASIS_REGISTRY as DEFAULT_BASIS_REGISTRY
from fdg._fdg import DEFAULT_INTEGRATION_REGISTRY as DEFAULT_INTEGRATION_REGISTRY
from fdg._fdg import BasisRegistry as BasisRegistry
from fdg._fdg import BasisSpecs as BasisSpecs
from fdg._fdg import CoordinateMap as CoordinateMap
from fdg._fdg import CovectorBasis as CovectorBasis
from fdg._fdg import DegreesOfFreedom as DegreesOfFreedom
from fdg._fdg import DirectDofMap as DirectDofMap
from fdg._fdg import ElementDoFs as ElementDoFs
from fdg._fdg import ElementKForms as ElementKForms
from fdg._fdg import FunctionSpace as FunctionSpace
from fdg._fdg import IntegrationRegistry as IntegrationRegistry
from fdg._fdg import IntegrationSpace as IntegrationSpace
from fdg._fdg import IntegrationSpecs as IntegrationSpecs
from fdg._fdg import KForm as KForm
from fdg._fdg import KFormSpecs as KFormSpecs
from fdg._fdg import Mesh as Mesh
from fdg._fdg import MeshGeometry as MeshGeometry
from fdg._fdg import MeshKFormSpecs as MeshKFormSpecs
from fdg._fdg import SampledSpaceMap as SampledSpaceMap
from fdg._fdg import SpaceMap as SpaceMap
from fdg._fdg import compute_gradient_mass_matrix as compute_gradient_mass_matrix
from fdg._fdg import compute_kform_boundary_load as compute_kform_boundary_load
from fdg._fdg import (
    compute_kform_boundary_mass_matrices as compute_kform_boundary_mass_matrices,
)
from fdg._fdg import (
    compute_kform_boundary_trace_moments as compute_kform_boundary_trace_moments,
)
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
    packed_kform_constraints_to_csr as packed_kform_constraints_to_csr,
)
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
from fdg._fdg import (
    transform_kform_to_target_sampled as transform_kform_to_target_sampled,
)
from fdg.boundary_conditions import BoundaryCondition as BoundaryCondition
from fdg.boundary_conditions import BoundaryData as BoundaryData
from fdg.boundary_conditions import BoundaryPair as BoundaryPair
from fdg.boundary_conditions import BoundaryPairGroup as BoundaryPairGroup
from fdg.boundary_conditions import (
    compute_kform_global_constraints as compute_kform_global_constraints,
)

# DoFs functions
from fdg.degrees_of_freedom import reconstruct as reconstruct

# Domains
from fdg.domains import Hypercube as Hypercube
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

# Moving meshes
from fdg.moving_mesh import MovingMesh as MovingMesh
from fdg.moving_mesh import advection_operator as advection_operator
from fdg.moving_mesh import lie_derivative_operator as lie_derivative_operator
from fdg.moving_mesh import (
    space_maps_from_geometry_dofs as space_maps_from_geometry_dofs,
)
from fdg.moving_mesh import stage_mass as stage_mass

# Hybridized solves
from fdg.solve import ElementBlockBuilder as ElementBlockBuilder
from fdg.solve import HybridizedSolution as HybridizedSolution
from fdg.solve import laplace_stiffness as laplace_stiffness
from fdg.solve import mixed_block as mixed_block
from fdg.solve import solve_hybridized as solve_hybridized

# Sparse assembly
from fdg.sparse import scatter_csc as scatter_csc

# Time marching
from fdg.time_marching import CollocationTableau as CollocationTableau
from fdg.time_marching import LinearSolver as LinearSolver
from fdg.time_marching import MarchResult as MarchResult
from fdg.time_marching import collocation_tableau as collocation_tableau
from fdg.time_marching import march as march
