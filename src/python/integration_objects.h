#ifndef FDG_INTEGRATION_RULE_OBJECT_H
#define FDG_INTEGRATION_RULE_OBJECT_H
#include "../integration/integration_rules.h"
#include "module.h"
#include <cutl/iterators/multidim_iteration.h>

typedef struct
{
    PyObject_HEAD;
    integration_rule_registry_t *registry;
} integration_registry_object;

typedef struct
{
    PyObject_HEAD;
    integration_spec_t spec;
} integration_specs_object;

FDG_INTERNAL
extern PyType_Spec integration_registry_type_spec;

FDG_INTERNAL
extern PyType_Spec integration_specs_type_spec;

FDG_INTERNAL
integration_registry_object *integration_registry_object_create(PyTypeObject *type);

FDG_INTERNAL
integration_specs_object *integration_specs_object_create(PyTypeObject *type, integration_spec_t spec);

typedef struct
{
    PyObject_VAR_HEAD;
    integration_spec_t specs[];
} integration_space_object;

FDG_INTERNAL
extern PyType_Spec integration_space_type_spec;

FDG_INTERNAL
multidim_iterator_t *integration_space_iterator(const integration_space_object *space);

FDG_INTERNAL
multidim_iterator_t *integration_specs_iterator(unsigned n_specs,
                                                const integration_spec_t FDG_ARRAY_ARG(specs, static n_specs));

FDG_INTERNAL
const integration_rule_t **python_integration_rules_get(unsigned n_rules,
                                                        const integration_spec_t specs[const static n_rules],
                                                        integration_rule_registry_t *registry);

FDG_INTERNAL
void python_integration_rules_release(unsigned n_rules, const integration_rule_t *rules[static n_rules],
                                      integration_rule_registry_t *registry);

#endif // FDG_INTEGRATION_RULE_OBJECT_H
