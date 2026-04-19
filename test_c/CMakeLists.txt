add_library(test_common STATIC
        common/common.h
        common/common.c
)
target_link_libraries(test_common PUBLIC fdg m)

# INTEGRATION TESTS

add_executable(test_gll integration/test_gll.c)
target_link_libraries(test_gll PRIVATE test_common)
add_test(NAME gauss-legendre-lobato COMMAND test_gll)

add_executable(test_glg integration/test_glg.c)
target_link_libraries(test_glg PRIVATE test_common)
add_test(NAME gauss-legendre COMMAND test_glg)

add_executable(test_int_rule_registry integration/test_int_rule_registry.c)
target_link_libraries(test_int_rule_registry PRIVATE test_common)
add_test(NAME int_rule_registry COMMAND test_int_rule_registry)

# BASIS TESTS

add_executable(test_basis_set_registry basis/test_basis_set_registry.c)
target_link_libraries(test_basis_set_registry PRIVATE test_common)
add_test(NAME basis_set_registry COMMAND test_basis_set_registry)

add_executable(test_covector_basis basis/test_covector_basis.c)
target_link_libraries(test_covector_basis PRIVATE test_common)
add_test(NAME covector_basis COMMAND test_covector_basis)

# COMMON TESTS

add_executable(test_rw_lock common/test_rw_lock.c)
target_link_libraries(test_rw_lock PRIVATE test_common)
add_test(NAME rw_lock COMMAND test_rw_lock)

# POLYNOMIAL TESTS

add_executable(test_lagrange_transpose polynomials/test_lagrange_transpose.c)
target_link_libraries(test_lagrange_transpose PRIVATE test_common)
add_test(NAME lagrange_transpose COMMAND test_lagrange_transpose)

add_executable(test_lagrange_derivative_transpose polynomials/test_lagrange_derivative_transpose.c)
target_link_libraries(test_lagrange_derivative_transpose PRIVATE test_common)
add_test(NAME lagrange_derivative_transpose COMMAND test_lagrange_derivative_transpose)

# OPERATION TESTS

## QR Decompositions
add_executable(test_qr_decomposition operations/test_qr_decomposition.c)
target_link_libraries(test_qr_decomposition PRIVATE test_common)
add_test(NAME qr_decomposition COMMAND test_qr_decomposition)

add_executable(test_qr_decomposition_2 operations/test_qr_decomposition_2.c)
target_link_libraries(test_qr_decomposition_2 PRIVATE test_common)
add_test(NAME qr_decomposition_2 COMMAND test_qr_decomposition_2)

add_executable(test_qr_decomposition_3 operations/test_qr_decomposition_3.c)
target_link_libraries(test_qr_decomposition_3 PRIVATE test_common)
add_test(NAME qr_decomposition_3 COMMAND test_qr_decomposition_3)

add_executable(test_back_substitute operations/test_back_substitute.c)
target_link_libraries(test_back_substitute PRIVATE test_common)
add_test(NAME back_substitute COMMAND test_back_substitute)

## Boundary scaling
add_executable(test_boundary_scaling operations/test_boundary_scaling.c)
target_link_libraries(test_boundary_scaling PRIVATE test_common)
add_test(NAME boundary_scaling COMMAND test_boundary_scaling)

## Topological
add_library(test_topo_common STATIC operations/topo_common.c)
target_link_libraries(test_topo_common PUBLIC test_common)

add_executable(test_topo operations/test_topo.c)
target_link_libraries(test_topo PRIVATE test_topo_common)
add_test(NAME topo_base COMMAND test_topo)

add_executable(test_topo_1d operations/test_topo_1d.c)
target_link_libraries(test_topo_1d PRIVATE test_topo_common)
add_test(NAME topo_1d COMMAND test_topo_1d)

add_executable(test_topo_2d operations/test_topo_2d.c)
target_link_libraries(test_topo_2d PRIVATE test_topo_common)
add_test(NAME topo_2d COMMAND test_topo_2d)
