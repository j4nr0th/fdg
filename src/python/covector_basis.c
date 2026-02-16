#include "covector_basis.h"

#include "cutl/iterators/combination_iterator.h"

static PyObject *covector_basis_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_Format(PyExc_TypeError, "Constructor takes no keyword arguments (got %R).", kwds);
        return NULL;
    }
    if (PyTuple_GET_SIZE(args) == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Constructor requires at least one argument.");
        return NULL;
    }
    // Check that all arguments are numbers
    const unsigned n_basis = PyTuple_GET_SIZE(args) - 1;
    for (unsigned i = 0; i <= n_basis; ++i)
    {
        if (!PyNumber_Check(PyTuple_GET_ITEM(args, i)))
        {
            PyErr_Format(PyExc_TypeError, "Argument %i was not a number, but was instead %R.", i,
                         PyTuple_GET_ITEM(args, i));
            return NULL;
        }
    }
    const Py_ssize_t n_dims = PyNumber_AsSsize_t(PyTuple_GET_ITEM(args, 0), PyExc_OverflowError);
    if (PyErr_Occurred())
        return NULL;

    if (n_dims < 0 || n_dims >= COVECTOR_BASIS_MAX_DIM)
    {
        PyErr_Format(PyExc_ValueError, "Expected number of dimensions in range [0, %u), but got %zd.",
                     COVECTOR_BASIS_MAX_DIM, n_dims);
        return NULL;
    }

    unsigned cnt = 0;
    unsigned present[COVECTOR_BASIS_MAX_DIM] = {};
    for (unsigned i = 0; i < n_basis; ++i)
    {
        const Py_ssize_t val = PyNumber_AsSsize_t(PyTuple_GET_ITEM(args, i + 1), PyExc_OverflowError);
        if (PyErr_Occurred())
            return NULL;
        if (val < 0 || val >= n_dims)
        {
            PyErr_Format(PyExc_ValueError, "Expected basis index in range [0, %zd), but got %zd.", n_dims, val);
            return NULL;
        }
        if (cnt > 0 && present[cnt - 1] >= val)
        {
            PyErr_Format(PyExc_ValueError, "Basis indices must be in strictly increasing order, but got %zd <= %u.",
                         val, present[cnt - 1]);
            return NULL;
        }
        present[cnt] = val;
        cnt += 1;
    }

    return (PyObject *)covector_basis_object_create(type, covector_basis_make(n_dims, 0, cnt, present));
}

static PyObject *covector_basis_get_ndim(PyObject *self, void *Py_UNUSED(closure))
{
    const covector_basis_object *const this = (covector_basis_object *)self;
    return PyLong_FromUnsignedLong(this->basis.dimension);
}

static PyObject *covector_basis_get_rank(PyObject *self, void *Py_UNUSED(closure))
{
    const covector_basis_object *const this = (covector_basis_object *)self;
    return PyLong_FromUnsignedLong(covector_basis_rank(this->basis));
}

static PyObject *covector_basis_get_sign(PyObject *self, void *Py_UNUSED(closure))
{
    const covector_basis_object *const this = (covector_basis_object *)self;
    return PyLong_FromLong(this->basis.sign ? -1 : +1);
}

static PyObject *covector_basis_get_index(PyObject *self, void *Py_UNUSED(closure))
{
    const covector_basis_object *const this = (covector_basis_object *)self;
    // Unpack indices of non-zero bits into an array
    uint8_t indices[COVECTOR_BASIS_MAX_DIM] = {};
    const unsigned r = covector_basis_rank(this->basis);
    for (unsigned i = 0, j = 0; i < this->basis.dimension && j < r; ++i)
    {

        if (covector_basis_has_component(this->basis, i))
        {
            indices[j] = i;
            j += 1;
        }
    }
    // Now we can call the function
    const unsigned idx = combination_get_index(this->basis.dimension, r, indices);
    return PyLong_FromUnsignedLong(idx);
}

static PyObject *covector_basis_xor(PyObject *self, PyObject *other)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return NULL;
    }

    if (!PyObject_TypeCheck(other, state->covector_basis_type) || !PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot xor two basis sets with different types (got %s and %s).",
                     Py_TYPE(other)->tp_name, Py_TYPE(self)->tp_name);
        return NULL;
    }

    const covector_basis_object *const this = (covector_basis_object *)self;
    const covector_basis_object *const that = (covector_basis_object *)other;
    // Check if either is zero
    if (covector_basis_is_zero(this->basis) || covector_basis_is_zero(that->basis))
    {
        return (PyObject *)covector_basis_object_create(state->covector_basis_type, COVECTOR_BASIS_ZERO);
    }

    // Check if these match
    if (this->basis.dimension != that->basis.dimension)
    {
        PyErr_Format(PyExc_ValueError, "Cannot xor two basis sets with different dimensions (%u != %u).",
                     this->basis.dimension, that->basis.dimension);
        return NULL;
    }

    const covector_basis_object *out =
        covector_basis_object_create(state->covector_basis_type, covector_basis_wedge(this->basis, that->basis));
    return (PyObject *)out;
}

static PyObject *covector_basis_negative(PyObject *self)
{
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot negate a basis of type %s.", Py_TYPE(self)->tp_name);
        return NULL;
    }
    const covector_basis_object *const this = (covector_basis_object *)self;
    return (PyObject *)covector_basis_object_create(state->covector_basis_type,
                                                    (covector_basis_t){.dimension = this->basis.dimension,
                                                                       .sign = !this->basis.sign,
                                                                       .basis_bits = this->basis.basis_bits});
}

static PyObject *covector_basis_invert(PyObject *self)
{
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot invert a basis of type %s.", Py_TYPE(self)->tp_name);
        return NULL;
    }
    const covector_basis_object *const this = (covector_basis_object *)self;
    return (PyObject *)covector_basis_object_create(state->covector_basis_type, covector_basis_hodge(this->basis));
}

static PyObject *covector_basis_richcompare(PyObject *self, PyObject *other, const int op)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (!PyObject_TypeCheck(self, state->covector_basis_type) || !PyObject_TypeCheck(other, state->covector_basis_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const covector_basis_object *const this = (covector_basis_object *)self;
    const covector_basis_object *const that = (covector_basis_object *)other;
    if (this->basis.sign != that->basis.sign)
    {
        if (op == Py_NE)
            Py_RETURN_TRUE;
        if (op == Py_EQ)
            Py_RETURN_FALSE;
        PyErr_SetString(PyExc_ValueError,
                        "Can not compare two basis with different signs for relation other than equal/not equal.");
        return NULL;
    }

    // If both are zero, it is simple.
    if (covector_basis_is_zero(this->basis) && covector_basis_is_zero(that->basis))
    {
        return PyBool_FromLong(op == Py_EQ);
    }
    // Get relation in terms of ordering
    const covector_basis_order_relation_t relation = covector_basis_determine_order(this->basis, that->basis);
    // PySys_FormatStdout("Relation status %u (%S vs %S)\n", relation, self, other);
    switch (relation)
    {
    case COVECTOR_BASIS_NOT_COMPARABLE:
        PyErr_Format(PyExc_ValueError,
                     "Cannot compare two bases of different dimensions and/or ranks ((%u, %u) != (%u, %u)).",
                     this->basis.dimension, covector_basis_rank(this->basis), that->basis.dimension,
                     covector_basis_rank(that->basis));
        return NULL;

    case COVECTOR_BASIS_EQUAL:
        if (op == Py_EQ || op == Py_GE || op == Py_LE)
        {
            Py_RETURN_TRUE;
        }
        break;
    case COVECTOR_BASIS_ORDER_AFTER:
        if (op == Py_GT || op == Py_GE || op == Py_NE)
        {
            Py_RETURN_TRUE;
        }
        break;
    case COVECTOR_BASIS_ORDER_BEFORE:
        if (op == Py_LT || op == Py_LE || op == Py_NE)
        {
            Py_RETURN_TRUE;
        }
        break;
    }

    Py_RETURN_FALSE;
}

static int covector_basis_bool(PyObject *self)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot convert a basis of type %s to a boolean.", Py_TYPE(self)->tp_name);
        return -1;
    }
    const covector_basis_object *const this = (covector_basis_object *)self;
    return !covector_basis_is_zero(this->basis);
}

static PyObject *covector_basis_str(PyObject *self)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return NULL;
    }
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot convert a basis of type %s to a string.", Py_TYPE(self)->tp_name);
        return NULL;
    }
    enum
    {
        CHARS_PER_BASIS = 8,
        BUFFER_SIZE = CHARS_PER_BASIS * COVECTOR_BASIS_MAX_DIM + 1
    };
    char buffer[BUFFER_SIZE];
    unsigned pos = 0;
    int status;
    const covector_basis_object *const this = (covector_basis_object *)self;

#define PRINT_TO_BUFFER(fmt, ...)                                                                                      \
    {                                                                                                                  \
        status = snprintf(buffer + pos, BUFFER_SIZE - pos, fmt __VA_OPT__(, )##__VA_ARGS__);                           \
        if (status < 0)                                                                                                \
        {                                                                                                              \
            PyErr_Format(PyExc_RuntimeError, "Failed to print to buffer: %s", strerror(errno));                        \
            return NULL;                                                                                               \
        }                                                                                                              \
        pos += status;                                                                                                 \
    }

    int set = 0;
    for (unsigned i = 0; i < this->basis.dimension; ++i)
    {
        if (this->basis.basis_bits & (1 << i))
        {
            if (set)
            {
                PRINT_TO_BUFFER(" ^ dx_%u", i);
            }
            else
            {
                PRINT_TO_BUFFER("%s dx_%u", this->basis.sign ? "-" : "+", i);
            }
            set = 1;
        }
    }
    if (!set)
    {
        if (this->basis.dimension == 0)
        {
            PRINT_TO_BUFFER("0");
        }
        else if (this->basis.sign)
        {
            PRINT_TO_BUFFER("-1");
        }
        else
        {
            PRINT_TO_BUFFER("+1");
        }
    }

    buffer[BUFFER_SIZE - 1] = 0;
    return PyUnicode_FromString(buffer);
}

static PyObject *covector_basis_repr(PyObject *self)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return NULL;
    }
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot convert a basis of type %s to a string.", Py_TYPE(self)->tp_name);
        return NULL;
    }
    enum
    {
        CHARS_PER_BASIS = 4,
        BUFFER_SIZE = CHARS_PER_BASIS * COVECTOR_BASIS_MAX_DIM + 1
    };
    char buffer[BUFFER_SIZE];
    unsigned pos = 0;
    int status;
    const covector_basis_object *const this = (covector_basis_object *)self;
    buffer[0] = 0; // in case we are dealing with zero
    for (unsigned i = 0; i < this->basis.dimension; ++i)
    {
        if (covector_basis_has_component(this->basis, i))
        {
            PRINT_TO_BUFFER(", %u", i);
        }
    }
    return PyUnicode_FromFormat("%cCovectorBasis(%u%s)", this->basis.sign ? '+' : '-', this->basis.dimension, buffer);
}

#undef PRINT_TO_BUFFER

static Py_hash_t covector_basis_hash(PyObject *self)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Cannot hash an object of type %s.", Py_TYPE(self)->tp_name);
        return -1;
    }
    const covector_basis_object *const this = (covector_basis_object *)self;
    // Reinterpret the basis value as the hash value
    const union {
        covector_basis_t basis;
        Py_hash_t hash;
    } hash = {.basis = {
                  .dimension = this->basis.dimension,
                  .sign = this->basis.sign,
                  .basis_bits = ~this->basis.basis_bits, // Flip the basis bits, since this way we are almost always
                                                         // sure we do not return -1
              }};
    return hash.hash;
}

static int ensure_basis_and_state(PyObject *self, PyTypeObject *defining_class, covector_basis_object **p_this,
                                  const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->covector_basis_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s object, but got a %s.", state->covector_basis_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_this = (covector_basis_object *)self;
    return 0;
}

PyDoc_STRVAR(covector_basis_normalize_docstring,
             "normalize() -> tuple[int, CovectorBasis]\nNormalize the basis by splitting the sign.\n");

static PyObject *covector_basis_normalize(PyObject *self, PyTypeObject *defining_class,
                                          PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                          const PyObject *kwnames)
{
    if (nargs || (kwnames && PyTuple_GET_SIZE(kwnames)))
    {
        PyErr_SetString(PyExc_TypeError, "normalize() takes no arguments.");
        return NULL;
    }

    const interplib_module_state_t *state;
    covector_basis_object *this;
    if (ensure_basis_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    covector_basis_object *const norm =
        covector_basis_object_create(state->covector_basis_type, (covector_basis_t){
                                                                     .dimension = this->basis.dimension,
                                                                     .sign = 0,
                                                                     .basis_bits = this->basis.basis_bits,
                                                                 });
    if (!norm)
        return NULL;

    return cpyutl_output_create_check(CPYOUT_TYPE_TUPLE,
                                      (const cpyutl_output_t[]){
                                          {.type = CPYOUT_TYPE_PYINT, .value_int = this->basis.sign ? -1 : +1},
                                          {.type = CPYOUT_TYPE_PYOBJ, .value_obj = (PyObject *)norm},
                                          {},
                                      });
}

PyDoc_STRVAR(covector_basis_docstring,
             "CovectorBasis(n: int, /, *idx: int)\n"
             "Type used to specify covector basis bundle.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "n : int\n"
             "    Dimension of the space basis bundle is in.\n"
             "\n"
             "*idx : int\n"
             "    Indices of basis present in the bundle. Should be sorted and non-repeating.\n");

static int covector_basis_contains(PyObject *self, PyObject *item)
{
    const covector_basis_object *const this = (covector_basis_object *)self;
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;

    // First, check if this is an integer or basis
    if (PyObject_TypeCheck(item, state->covector_basis_type))
    {
        // It is a basis
        const covector_basis_object *const that = (covector_basis_object *)item;
        if (this->basis.dimension != that->basis.dimension)
        {
            return 0;
        }

        // Check if it is a subset of the self
        for (unsigned i = 0; i < this->basis.dimension; ++i)
        {
            if (!covector_basis_has_component(this->basis, i) && covector_basis_has_component(that->basis, i))
            {
                return 0;
            }
        }
        return 1;
    }
    // Try and convert to an integer
    const Py_ssize_t num = PyNumber_AsSsize_t(item, PyExc_OverflowError);
    if (PyErr_Occurred())
        return -1;
    if (num < 0 || num >= this->basis.dimension)
    {
        PyErr_Format(PyExc_ValueError, "Index %zd is out of bounds for basis of dimension %u.", num,
                     this->basis.dimension);
        return -1;
    }

    return covector_basis_has_component(this->basis, num);
}

PyType_Spec covector_basis_type_spec = {
    .name = "interplib._interp.CovectorBasis",
    .basicsize = sizeof(covector_basis_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_new, covector_basis_new},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "ndim",
                     .get = covector_basis_get_ndim,
                     .doc = "int : Number of dimensions of the space the basis are in.",
                 },
                 {
                     .name = "rank",
                     .get = covector_basis_get_rank,
                     .doc = "int : Number of basis contained.",
                 },
                 {
                     .name = "sign",
                     .get = covector_basis_get_sign,
                     .doc = "int : The sign of the basis.",
                 },
                 {
                     .name = "index",
                     .get = covector_basis_get_index,
                     .doc = "int : Index of the basis for the k-form.",
                 },
                 {},
             }},
            {Py_nb_xor, covector_basis_xor},
            {Py_nb_negative, covector_basis_negative},
            {Py_nb_invert, covector_basis_invert},
            {Py_tp_richcompare, covector_basis_richcompare},
            {Py_nb_bool, covector_basis_bool},
            {Py_tp_doc, (void *)covector_basis_docstring},
            {Py_tp_str, covector_basis_str},
            {Py_tp_hash, covector_basis_hash},
            {Py_tp_repr, covector_basis_repr},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "normalize",
                     .ml_meth = (void *)covector_basis_normalize,
                     .ml_flags = METH_FASTCALL | METH_METHOD | METH_KEYWORDS,
                     .ml_doc = covector_basis_normalize_docstring,
                 },
                 {},
             }},
            {Py_sq_contains, covector_basis_contains},
            {},
        },
};

covector_basis_object *covector_basis_object_create(PyTypeObject *type, const covector_basis_t basis)
{
    covector_basis_object *const this = (covector_basis_object *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;
    // When we init we need to write, and this will upset the compiler. After this, this is all const.
    *(covector_basis_t *)&this->basis = basis;
    return this;
}

unsigned kform_basis_get_num_dofs(const unsigned ndim, const basis_spec_t basis[static ndim], const unsigned order,
                                  const uint8_t components[static order])
{
    unsigned dofs = 1;
    for (unsigned idim = 0, icomponent = 0; idim < ndim; ++idim)
    {
        unsigned n;
        if (icomponent != order && idim == components[icomponent])
        {
            n = basis[idim].order;
            icomponent += 1;
        }
        else
        {
            n = basis[idim].order + 1;
        }
        dofs *= n;
    }
    return dofs;
}

void kform_basis_set_iterator(const unsigned ndim, const basis_spec_t basis[static ndim], const unsigned order,
                              const uint8_t components[static order], multidim_iterator_t *iter)
{
    // Can't put the check here, since the iterator is not initialized on the first run.
    for (unsigned idim = 0, icomponent = 0; idim < ndim; ++idim)
    {
        unsigned n;
        if (icomponent != order && idim == components[icomponent])
        {
            n = basis[idim].order;
            icomponent += 1;
        }
        else
        {
            n = basis[idim].order + 1;
        }
        multidim_iterator_init_dim(iter, idim, n);
    }
}
