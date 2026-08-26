#ifndef FDG_PYTHON_CONSTRAINTS_H
#define FDG_PYTHON_CONSTRAINTS_H

#include "kform_objects.h"
#include "mappings.h"
#include "module.h"

FDG_INTERNAL
PyObject *compute_kform_boundary_constraints_impl(const interplib_module_state_t *state, kform_spec_object *test_spec,
                                                  kform_spec_object *element_spec, space_map_object *element_map,
                                                  const int8_t *orientation);

#endif // FDG_PYTHON_CONSTRAINTS_H
