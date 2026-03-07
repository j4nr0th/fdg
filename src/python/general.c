#include "general.h"

typedef struct
{
    unsigned idim;
    unsigned level;
    unsigned index;
    long int offset;
} loop_state_t;

static void scale_array_boundary_iterative(const unsigned ndim, const npy_intp dims[static ndim],
                                           const npy_intp strides[static ndim], loop_state_t work_stack[ndim - 1],
                                           double *restrict out_array)
{

    unsigned spos = 1;
    work_stack[0] = (loop_state_t){.idim = ndim, .level = 0, .index = 0, .offset = 0};

    while (spos > 0)
    {
        const loop_state_t state = work_stack[spos - 1];
        const unsigned idim = state.idim;
        const unsigned level = state.level;
        const long int offset = state.offset;
        const unsigned index = state.index;

        if (index == dims[idim - 1])
        {
            // Pop off the stack
            spos -= 1;
        }
        else
        {
            work_stack[spos - 1].index += 1;

            const unsigned new_level = level + (index == 0 || index + 1 == dims[idim - 1]);
            const long new_offset = offset + index * strides[idim - 1];
            if (idim > 2)
            {
                // Push on a new frame (level depends on if we are first/last
                work_stack[spos] = (loop_state_t){.idim = idim - 1, .level = new_level, .offset = new_offset};
                spos += 1;
                ASSERT(spos < ndim, "I miscounted loop stack space needed.");
            }
            else if (new_level > 0)
            {
                const long int dim = dims[0];
                const long int stride = strides[0];
                out_array[new_offset] /= (new_level + 1.0);
                for (long int i = 1; i < dim - 1 && new_level > 1; ++i)
                {
                    out_array[new_offset + i * stride] /= new_level;
                }
                out_array[new_offset + (dim - 1) * stride] /= (new_level + 1.0);
            }
        }
    }
}

static PyObject *fdg_scale_array_boundary(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                          PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    PyObject *py_in;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_in},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyArrayObject *const array = (PyArrayObject *)PyArray_FROMANY(py_in, NPY_DOUBLE, 0, 0,
                                                                  NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS |
                                                                      NPY_ARRAY_WRITEABLE | NPY_ARRAY_ENSURECOPY);
    if (!array)
        return NULL;

    // Get the dimensions
    const unsigned ndims = (unsigned)PyArray_NDIM(array);
    const npy_intp *const dims = PyArray_DIMS(array);

    // Allocate counters
    loop_state_t *work_stack;
    npy_intp *strides;
    void *const mem =
        cutl_alloc_group(&PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                                                {sizeof(*work_stack) * (ndims - 1), .p_ptr = (void **)&work_stack},
                                                {sizeof(*strides) * (ndims), .p_ptr = (void **)&strides},
                                                {},
                                            });
    if (!mem)
    {
        Py_DECREF(array);
        return NULL;
    }

    // Compute correct strides (numpy only knows them in terms of bytes, but we want element strides)
    strides[ndims - 1] = 1;
    for (unsigned i = 1; i < ndims; ++i)
        strides[ndims - i - 1] = strides[ndims - i] * dims[ndims - i];

    // Scale the values
    scale_array_boundary_iterative(ndims, dims, strides, work_stack, PyArray_DATA(array));

    // Release the memory
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);
    return (PyObject *)array;
}

PyDoc_STRVAR(fdg_scale_array_boundary_docstring,
             "_scale_array_boundry(arr: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray[numpy.double]\n"
             "Scale the array based on how many N-dimensional boundaries an entry appears.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "arr : array_like\n"
             "    Array to scale.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Scaled array.\n");

PyMethodDef general_methods[] = {
    {
        .ml_name = "_scale_array_boundary",
        .ml_meth = (void *)fdg_scale_array_boundary,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = fdg_scale_array_boundary_docstring,
    },
    {}, // Sentinel
};
