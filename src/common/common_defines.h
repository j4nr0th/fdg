#ifndef COMMON_DEFINES_H
#define COMMON_DEFINES_H

#ifdef __GNUC__
/**
 * @brief Mark a symbol as hidden from the shared library symbol table.
 *
 * Functions annotated with this macro are internal to the library and are
 * not part of its public ABI. When the library is built as a static library
 * they are still usable by code that links the archive directly, such as the
 * Python bindings.
 */
#define FDG_INTERNAL __attribute__((visibility("hidden")))
/**
 * @brief Mark a symbol as exported with default visibility.
 *
 * This is the default visibility anyway and the macro is provided for
 * symmetry with FDG_INTERNAL.
 */
#define FDG_EXTERNAL __attribute__((visibility("default")))
/**
 * @brief Annotate an array parameter with its size.
 *
 * The macro expands to a sized array declarator when compiling with GCC or
 * Clang, and to a plain pointer otherwise. This lets the compiler check
 * bounds of array parameters declared with it.
 *
 * @param arr Name of the array parameter.
 * @param sz Size expression for the array, which may reference other
 *           parameters of the function.
 */
#define FDG_ARRAY_ARG(arr, sz) arr[sz]

#define FDG_EXPECT_CONDITION(x) (__builtin_expect(x, 1))

#endif

#ifndef FDG_INTERNAL
#define FDG_INTERNAL
#endif

#ifndef FDG_EXTERNAL
#define FDG_EXTERNAL
#endif

#ifndef FDG_ARRAY_ARG
#define FDG_ARRAY_ARG(arr, sz) *arr
#endif

#endif // COMMON_DEFINES_H
