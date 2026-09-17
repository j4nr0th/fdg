#pragma once
#include <stdbool.h>

#include "../basis/basis_set.h"
#include <cutl/iterators/combination_iterator.h>

typedef struct
{
    unsigned ndim;
    const basis_set_t *basis;
} function_space_t;

typedef struct
{
    unsigned ndim;
    unsigned order;
    const basis_spec_t *basis;
} kform_spec_t;

static inline unsigned kform_spec_component_count(const kform_spec_t *kform_spec)
{
    return combination_total_count(kform_spec->ndim, kform_spec->order);
}

/**
 * @brief Decode the covector axes of one k-form component.
 *
 * @param spec K-form specification.
 * @param component Component index in `[0, C(spec->ndim, spec->order))`.
 * @param axes Output array of `spec->order` sorted covector axes.
 */
static inline void kform_component_axes(const kform_spec_t *spec, unsigned component,
                                        uint8_t axes[static spec->order == 0 ? 1 : spec->order])
{
    combination_set_to_index((uint8_t)spec->ndim, (uint8_t)spec->order, axes, component);
}

/**
 * @brief Count the local DoFs of one k-form component.
 *
 * An active wedge axis contributes `basis_order` functions; an inactive axis
 * contributes `basis_order + 1`. The returned product is local to the
 * component, not its offset in the flattened array.
 *
 * @param ndim Number of reference dimensions.
 * @param basis Basis specification of each reference dimension.
 * @param order Order of the k-form.
 * @param axes Sorted covector axes of the component.
 * @return Local DoF count for the specified component.
 */
static inline size_t kform_component_dof_count(unsigned ndim, const basis_spec_t basis[static ndim], unsigned order,
                                               const uint8_t axes[static order == 0 ? 1 : order])
{
    size_t dof_count = 1;
    for (unsigned idim = 0, iaxis = 0; idim < ndim; ++idim)
    {
        const bool active = iaxis < order && axes[iaxis] == idim;
        dof_count *= (size_t)basis[idim].order + (active ? 0 : 1);
        if (active)
            ++iaxis;
    }
    return dof_count;
}

/**
 * @brief Count the local DoFs of one k-form component.
 *
 * See #kform_component_dof_count for the counting rule.
 *
 * @param spec K-form specification.
 * @param component Component index in `[0, C(spec->ndim, spec->order))`.
 * @return Local DoF count for the specified component.
 */
static inline size_t kform_spec_component_dof_count(const kform_spec_t *spec, unsigned component)
{
    CUTL_ASSERT((size_t)component < kform_spec_component_count(spec), "Component index out of range");

    uint8_t axes[UINT8_MAX];
    kform_component_axes(spec, component, axes);
    return kform_component_dof_count(spec->ndim, spec->basis, spec->order, axes);
}

/**
 * @brief Compute cumulative offsets for all k-form components.
 *
 * On success, `offsets[c]` is the first flattened DoF of component `c`, and
 * `offsets[component_count]` is the total DoF count.
 *
 * @param spec Test-space specification.
 * @param offset_count Number of entries available in `offsets`.
 * @param offsets Output array with at least `component_count + 1` entries.
 */
static inline void kform_spec_component_offsets(const kform_spec_t *spec, size_t offset_count,
                                                size_t offsets[const static offset_count])
{

    const size_t component_count = kform_spec_component_count(spec);
    CUTL_ASSERT(offset_count >= component_count + 1, "Insufficient storage for component offsets");

    size_t offset = 0;
    offsets[0] = 0;
    for (size_t component = 0; component < component_count; ++component)
    {
        const size_t dof_count = kform_spec_component_dof_count(spec, (unsigned)component);
        offset += dof_count;
        offsets[component + 1] = offset;
    }
}

/**
 * @brief Count the total number of local DoFs across all components.
 *
 * @param spec K-form specification.
 * @return Sum of the component-local DoF counts.
 */
static inline size_t kform_spec_total_dofs(const kform_spec_t *spec)
{
    const size_t component_count = kform_spec_component_count(spec);
    size_t dof_count = 0;
    for (size_t component = 0; component < component_count; ++component)
        dof_count += kform_spec_component_dof_count(spec, (unsigned)component);
    return dof_count;
}

typedef struct
{
    unsigned ndim;
    unsigned order;
    const basis_set_t *basis;
} kform_space_t;

static inline unsigned kform_component_count(const kform_space_t *kform_space)
{
    return combination_total_count(kform_space->ndim, kform_space->order);
}
