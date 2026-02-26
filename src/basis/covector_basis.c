#include "covector_basis.h"

#include <stdarg.h>
#include <stdint.h>

static unsigned popcnt_uint(unsigned x)
{
    unsigned count = 0;
    // We just do popcount
#ifdef __GNUC__
    count = __builtin_popcount(x);
#else
    // I do not know what the compiler we are on, we just do it the regular way.
    while (x)
    {
        count += 1;
        x &= ~(x - 1);
    }
#endif
    return count;
}

int covector_basis_has_component(const covector_basis_t basis, const unsigned dim)
{
    ASSERT(dim < basis.dimension, "Dimension was out of bounds.");
    return (basis.basis_bits & 1 << dim) != 0;
}
covector_basis_t covector_basis_wedge(const covector_basis_t b1, const covector_basis_t b2)
{
    ASSERT(b1.dimension == b2.dimension, "Basis had different dimensions (%u and %u).", b1.dimension, b2.dimension);
    if (b1.basis_bits & b2.basis_bits)
    {
        // There is some overlap, which means the result is zero
        return COVECTOR_BASIS_ZERO;
    }

    unsigned sign = b1.sign ^ b2.sign;
    // Add bits from b2->basis_bits to resulting_bits, but each time we have a bit set after the destination, we flip
    // the sign
    unsigned remaining_bits = b2.basis_bits;
    while (remaining_bits)
    {
        // Extract the next bit
        const unsigned new_remainder = remaining_bits & (remaining_bits - 1);
        const unsigned new_bit = remaining_bits ^ new_remainder;
        remaining_bits = new_remainder;

        const unsigned higher_bits = b1.basis_bits & ~(new_bit - 1);
        // Flip the sign the same number of times as the number of higher bits, so if the number is odd.
        // we reverse the sign, otherwise we keep it
        if (popcnt_uint(higher_bits) & 1)
        {
            sign = !sign;
        }
    }

    return (covector_basis_t){
        .dimension = b1.dimension,                   // We copy the dimension
        .sign = sign,                                // Sign is the product
        .basis_bits = b1.basis_bits | b2.basis_bits, // Basis are merged
    };
}

unsigned covector_basis_rank(const covector_basis_t basis)
{
    return popcnt_uint(basis.basis_bits);
}

covector_basis_t covector_basis_create(const unsigned dimension, const int sign, const unsigned rank,
                                       const unsigned FDG_ARRAY_ARG(indices, static rank))
{
    ASSERT(dimension > 0, "Dimension must be positive.");
    ASSERT(rank <= dimension, "Rank was larger than dimension.");
    ASSERT(dimension < COVECTOR_BASIS_MAX_DIM, "Maximum dimension count of %u was exceeded!", COVECTOR_BASIS_MAX_DIM);
    covector_basis_t basis = {.dimension = dimension, .sign = sign < 0};
    for (unsigned i = 0; i < rank; ++i)
    {
        const unsigned idx = indices[i];
        ASSERT(i == 0 || idx > indices[i - 1], "Indices were not sorted in ascending order.");
        ASSERT(idx < dimension, "Index %u was out of bounds for dimension %u.", idx, dimension);
        const unsigned bit = (1u << idx);
        ASSERT(basis.basis_bits ^ bit, "Component %u was already specified.", idx);
        basis.basis_bits |= bit;
    }

    return basis;
}

covector_basis_t covector_basis_create_u8(const unsigned dimension, const int sign, const unsigned rank,
                                          const uint8_t FDG_ARRAY_ARG(indices, static rank))
{
    ASSERT(dimension > 0, "Dimension must be positive.");
    ASSERT(rank <= dimension, "Rank was larger than dimension.");
    ASSERT(dimension < COVECTOR_BASIS_MAX_DIM, "Maximum dimension count of %u was exceeded!", COVECTOR_BASIS_MAX_DIM);
    covector_basis_t basis = {.dimension = dimension, .sign = sign < 0};
    for (unsigned i = 0; i < rank; ++i)
    {
        const unsigned idx = indices[i];
        ASSERT(i == 0 || idx > indices[i - 1], "Indices were not sorted in ascending order.");
        ASSERT(idx < dimension, "Index %u was out of bounds for dimension %u.", idx, dimension);
        const unsigned bit = (1u << idx);
        ASSERT(basis.basis_bits ^ bit, "Component %u was already specified.", idx);
        basis.basis_bits |= bit;
    }

    return basis;
}

covector_basis_t covector_basis_apply_contra_basis(const covector_basis_t basis, const unsigned dim)
{
    ASSERT(covector_basis_has_component(basis, dim), "Basis did not have component %u.", dim);
    // Remove the bit with `dim`, but flip the sign the number of bits set before it!
    int sign = basis.sign;
    const unsigned basis_bits = basis.basis_bits ^ (1u << dim);
    const unsigned lower_bits = basis_bits & (basis_bits - 1);
    if (popcnt_uint(lower_bits) & 1)
        sign = !sign;
    return (covector_basis_t){.dimension = basis.dimension, .sign = sign, .basis_bits = basis_bits};
}

covector_basis_order_relation_t covector_basis_determine_order(const covector_basis_t basis_1,
                                                               const covector_basis_t basis_2)
{
    if (basis_1.dimension != basis_2.dimension)
        return COVECTOR_BASIS_NOT_COMPARABLE;

    const unsigned rank_1 = covector_basis_rank(basis_1);
    if (rank_1 != covector_basis_rank(basis_2))
        return COVECTOR_BASIS_NOT_COMPARABLE;

    // We can do a quick equality check now.
    if (basis_1.basis_bits == basis_2.basis_bits)
        return COVECTOR_BASIS_EQUAL;

    const unsigned unique_bits1 = basis_1.basis_bits & ~basis_2.basis_bits;
    const unsigned unique_bits2 = basis_2.basis_bits & ~basis_1.basis_bits;
    const unsigned first_bit_1 = unique_bits1 & ~(unique_bits1 - 1);
    const unsigned first_bit_2 = unique_bits2 & ~(unique_bits2 - 1);

    // If our rank is below half the dimension, the one that has lower bits is first, otherwise it is reversed.
    if ((first_bit_1 < first_bit_2) ^ (rank_1 > basis_1.dimension / 2))
    {
        return COVECTOR_BASIS_ORDER_BEFORE;
    }
    return COVECTOR_BASIS_ORDER_AFTER;
}

int covector_basis_is_zero(const covector_basis_t basis)
{
    return basis.dimension == 0;
}

covector_basis_t covector_basis_hodge(const covector_basis_t basis)
{
    if (basis.dimension == 0)
        return basis;

    unsigned hodge_bits = basis.basis_bits ^ ((1u << basis.dimension) - 1);
    covector_basis_t hodge_basis = {.dimension = basis.dimension, .sign = basis.sign, .basis_bits = hodge_bits};
    // We introduce the bits one by one and count the number of bits following it to get the flip count
    unsigned flip_count = 0;
    while (hodge_bits)
    {
        // Get the next lowest bit.
        const unsigned next_bit = hodge_bits & ~(hodge_bits - 1);
        // Remove it from the remaining bits.
        hodge_bits ^= next_bit;
        const unsigned higher_bits = basis.basis_bits & ~(next_bit - 1);
        if (!higher_bits)
            break;
        // Update flip count based on how many higher bits there are in the original
        flip_count += popcnt_uint(higher_bits);
    }
    // Flip sign if we had an odd number of flips
    if (flip_count & 1)
        hodge_basis.sign = !hodge_basis.sign;

    return hodge_basis;
}
