#ifndef INTERPLIB_COVECTOR_BASIS_H
#define INTERPLIB_COVECTOR_BASIS_H
#include "../common/common_defines.h"

#include <stdint.h>

enum
{
    // The highest number of basis dimensions and rank, due to the number of bits
    // we can pack in an `unsigned` type. Since `unsigned` is 32 bits basically
    // everywhere, I do not imagine this ever being too few.
    COVECTOR_BASIS_MAX_DIM = sizeof(unsigned) * 8,
};

/**
 * Representation of the bundle of basis covectors.
 *
 * The basis consist of wedge products of individual covectors, such as
 * `dx_1 ^ dx_3 ^ dx_4`, `dx_2 ^ dx_3`. Due to the nature of how the wedge
 * product affects the sign, these basis have their sign, which can change
 * when applying wedge or Hodge to them.
 *
 */
typedef struct
{
    unsigned dimension : 31;
    unsigned sign : 1;
    unsigned basis_bits;
} covector_basis_t;

// When the dimension is zero, the basis is assumed to be zero.
static const covector_basis_t COVECTOR_BASIS_ZERO = {};

INTERPLIB_INTERNAL
int covector_basis_has_component(covector_basis_t basis, unsigned dim);

/**
 * Computes the exterior wedge product of two covector bases.
 *
 * The wedge product computes a new covector basis in which the dimensions
 * of the input bases are merged, considering their orientation. If the input
 * bases have overlapping dimensions, the resulting covector is zero.
 *
 * @param b1 A pointer to the first covector basis.
 * @param b2 A pointer to the second covector basis.
 * @return A new covector basis resulting from the wedge product of b1 and b2.
 *         The result contains the combined dimension, updated sign, and basis bits.
 *         If the dimensions of b1 and b2 overlap, the result is a zero basis.
 */
INTERPLIB_INTERNAL
covector_basis_t covector_basis_wedge(covector_basis_t b1, covector_basis_t b2);

/**
 * Computes the rank of the given covector basis.
 * The rank is determined by the number of covector basis.
 *
 * @param basis A pointer to the covector_basis_t structure representing
 *              the covector basis whose rank is to be computed.
 * @return The rank of the covector basis, which is the count of set bits
 *         in the basis_bits field of the provided structure.
 */
INTERPLIB_INTERNAL
unsigned covector_basis_rank(covector_basis_t basis);

/**
 * Constructs a covector basis representation with the specified parameters.
 *
 * @param dimension The dimension of the covector space. Must be less than COVECTOR_BASIS_MAX_DIM.
 * @param sign The sign of the covector basis. Negative values mean the basis have a negative sign.
 * @param rank The number of active components in the covector basis. Must not exceed the dimension.
 * @param indices Sorted indices specifying the indices of the components to be set in the basis.
 *           Each index must be within the bounds of the dimension and must be unique.
 *           Behavior is undefined if an index is repeated, out of bounds, or unsorted.
 * @return The constructed covector_basis_t structure representing the basis.
 */
INTERPLIB_INTERNAL
covector_basis_t covector_basis_create(unsigned dimension, int sign, unsigned rank,
                                       const unsigned INTERPLIB_ARRAY_ARG(indices, static rank));

/**
 * Constructs a covector basis representation with the specified parameters.
 *
 * @param dimension The dimension of the covector space. Must be less than COVECTOR_BASIS_MAX_DIM.
 * @param sign The sign of the covector basis. Negative values mean the basis have a negative sign.
 * @param rank The number of active components in the covector basis. Must not exceed the dimension.
 * @param indices Sorted indices specifying the indices of the components to be set in the basis.
 *           Each index must be within the bounds of the dimension and must be unique.
 *           Behavior is undefined if an index is repeated, out of bounds, or unsorted.
 * @return The constructed covector_basis_t structure representing the basis.
 */
INTERPLIB_INTERNAL
covector_basis_t covector_basis_create_u8(unsigned dimension, int sign, unsigned rank,
                                          const uint8_t INTERPLIB_ARRAY_ARG(indices, static rank));

/**
 * Computes the result of applying a contravector basis on the covector basis bundle.
 *
 * The function can only be called if the basis has the covariant component corresponding to
 * the contravariant basis that is to be applied. Otherwise, the result is by definition zero.
 *
 * @param basis Pointer to the covector basis structure to be transformed.
 * @param dim The index of the contravector basis.
 *
 * @return Resulting covector basis bundle.
 */
INTERPLIB_INTERNAL
covector_basis_t covector_basis_apply_contra_basis(covector_basis_t basis, unsigned dim);

typedef enum
{
    COVECTOR_BASIS_ORDER_BEFORE,   // B1 before B2
    COVECTOR_BASIS_EQUAL,          // B1 with B2
    COVECTOR_BASIS_ORDER_AFTER,    // B1 after B2
    COVECTOR_BASIS_NOT_COMPARABLE, // B1 and B2 are not of the same rank and/or dimension.
} covector_basis_order_relation_t;

/**
 * Determines the order relation between two covector bases.
 *
 * @param basis_1 Pointer to the first covector basis.
 * @param basis_2 Pointer to the second covector basis.
 * @return A value of type covector_basis_order_relation_t indicating the order
 *         relation between the two bases:
 *         - COVECTOR_BASIS_ORDER_BEFORE: The first basis is ordered before the second.
 *         - COVECTOR_BASIS_EQUAL: The two bases are equal.
 *         - COVECTOR_BASIS_ORDER_AFTER: The first basis is ordered after the second.
 *         - COVECTOR_BASIS_NOT_COMPARABLE: The bases are not comparable due to
 *           differing ranks or dimensions.
 */
INTERPLIB_INTERNAL
covector_basis_order_relation_t covector_basis_determine_order(covector_basis_t basis_1, covector_basis_t basis_2);

INTERPLIB_INTERNAL
int covector_basis_is_zero(covector_basis_t basis);

covector_basis_t covector_basis_hodge(covector_basis_t basis);

#endif // INTERPLIB_COVECTOR_BASIS_H
