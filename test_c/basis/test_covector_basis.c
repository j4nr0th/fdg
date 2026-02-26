#include "../../src/basis/covector_basis.h"
#include "../common/common.h"

static void print_basis(const covector_basis_t *basis)
{
    printf("%c ", basis->sign ? '-' : '+');
    int triggered = 0;
    for (unsigned i = 0; i < basis->dimension; ++i)
    {
        if (basis->basis_bits & (1 << i))
        {
            if (triggered)
            {
                printf(" ^ ");
            }
            triggered = 1;
            printf("dx_%d", i);
        }
    }
    if (!triggered)
        printf("1");
}

void check_basis_equal(const covector_basis_t *b1, const covector_basis_t *b2)
{
    TEST_ASSERTION(b1->dimension == b2->dimension, "Dimensions do not match");
    TEST_ASSERTION(b1->sign == b2->sign, "Sign does not match");
    TEST_ASSERTION(b1->basis_bits == b2->basis_bits, "Basis does not match");
}

void check_hodge(const covector_basis_t b)
{
    printf("Basis: ");
    print_basis(&b);
    printf("\n");
    const covector_basis_t hodge = covector_basis_hodge(b);
    printf("Hodge: ");
    print_basis(&hodge);
    printf("\n");
    const covector_basis_t wedge = covector_basis_wedge(b, hodge);
    printf("Wedge: ");
    print_basis(&wedge);
    printf("\n");
    TEST_ASSERTION(covector_basis_rank(wedge) == b.dimension,
                   "Wedge of basis and its Hodge should have rank equal it its dimension");
    TEST_ASSERTION(wedge.sign == 0, "Wedge of basis and its Hodge should have the same sign as the basis");
}

int main(void)
{
    typedef covector_basis_t basis;
    // Basic checks in 3d
    {
        const basis basis_1 = covector_basis_create(3, 0, 1, (unsigned[]){0});
        check_hodge(basis_1);
        const basis basis_2 = covector_basis_create(3, 0, 1, (unsigned[]){1});
        check_hodge(basis_2);

        // b1 ^ b2
        const basis basis_3 = covector_basis_wedge(basis_1, basis_2);
        check_hodge(basis_3);
        printf("b1 ^ b2: ");
        print_basis(&basis_3);
        printf("\n");
        const basis expected_basis_3 = covector_basis_create(3, 0, 2, (unsigned[]){0, 1});
        check_basis_equal(&basis_3, &expected_basis_3);

        // b2 ^ b1 = - b1 ^ b2
        const basis basis_4 = covector_basis_wedge(basis_2, basis_1);
        check_hodge(basis_4);
        printf("Basis 2 wedge Basis 1: ");
        print_basis(&basis_4);
        printf("\n");
        const basis expected_basis_4 = covector_basis_create(3, -1, 2, (unsigned[]){0, 1});
        check_basis_equal(&basis_4, &expected_basis_4);
    }

    // Some more involved checks in 5d
    {
        const basis b0 = covector_basis_create(5, 0, 1, (unsigned[]){0});
        check_hodge(b0);
        const basis b1 = covector_basis_create(5, 0, 1, (unsigned[]){1});
        check_hodge(b1);
        const basis b4 = covector_basis_create(5, 0, 1, (unsigned[]){4});
        check_hodge(b4);

        // b1 ^ b4 ^ b0
        const basis b140 = covector_basis_wedge(covector_basis_wedge(b1, b4), b0);
        check_hodge(b140);
        const basis expected_b140 = covector_basis_create(5, 0, 3, (unsigned[]){0, 1, 4});
        printf("b1 ^ b4 ^ b0: ");
        print_basis(&b140);
        printf("\n");
        check_basis_equal(&b140, &expected_b140);

        // b4 ^ b1 ^ b0
        const basis b410 = covector_basis_wedge(covector_basis_wedge(b4, b1), b0);
        check_hodge(b410);
        const basis expected_b410 = covector_basis_create(5, -1, 3, (unsigned[]){0, 1, 4});
        printf("b4 ^ b1 ^ b0: ");
        print_basis(&b410);
        printf("\n");
        check_basis_equal(&b410, &expected_b410);
    }

    // Fun in 7d
    {
        // (b1 ^ b3) ^ (b6 ^ b0 ^ b2) = - b0 ^ b1 ^ b2 ^ b3 ^ b6
        const basis b13 = covector_basis_create(7, 0, 2, (unsigned[]){1, 3});
        print_basis(&b13);
        const basis b602 = covector_basis_create(7, 0, 3, (unsigned[]){0, 2, 6});
        print_basis(&b602);
        const basis expected_result = covector_basis_create(7, -1, 5, (unsigned[]){0, 1, 2, 3, 6});
        const basis computed = covector_basis_wedge(b13, b602);
        print_basis(&computed);
        printf("(b1 ^ b3) ^ (b6 ^ b0 ^ b2): ");
        print_basis(&computed);
        printf("\n");
        printf("Expected: ");
        print_basis(&expected_result);
        printf("\n");
        check_basis_equal(&computed, &expected_result);
    }

    return 0;
}
