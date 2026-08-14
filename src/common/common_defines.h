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
#ifdef _DEBUG
/**
 * @brief Halt the program in a debugger-detectable way.
 *
 * Used by ASSERT when a condition fails. In Debug builds with GCC this
 * expands to a trap instruction; otherwise it expands to exit(EXIT_FAILURE).
 */
#define FDG_BREAK __builtin_trap()
#endif
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

#ifndef ASSERT
#ifdef FDG_ASSERTS
/**
 * @brief Test a condition and report a failure to stderr.
 *
 * Evaluates the condition exactly once. If it is false, a message with the
 * file, line, function and condition is printed to stderr and the program
 * terminates via FDG_BREAK.
 *
 * @note ASSERT is only active when built with FDG_ASSERTS defined. Otherwise
 * the macro expands to a no-op that does not evaluate the condition at all.
 *
 * @param condition Condition that must hold.
 * @param message Format string describing the expected condition, followed by
 *                any additional format arguments.
 */
#include <stdio.h>
#include <stdlib.h>
#ifndef FDG_BREAK
#define FDG_BREAK exit(EXIT_FAILURE)
#endif
#define ASSERT(condition, message, ...)                                                                                \
    ((condition) ? (void)0                                                                                             \
                 : (fprintf(stderr, "%s:%d: %s: Assertion '%s' failed - " message "\n", __FILE__, __LINE__, __func__,  \
                            #condition __VA_OPT__(, ) __VA_ARGS__),                                                    \
                    FDG_BREAK))
#else
#ifndef ASSERT
#define ASSERT(condition, message) 0
#endif
#endif
#endif

#ifdef __GNUC__
/**
 * @brief Inform the compiler that a condition holds, for optimization purposes.
 *
 * If the condition does not actually hold, the behavior of the program is
 * undefined. Unlike ASSERT this macro never evaluates or checks the condition
 * at runtime; it is purely an optimization hint.
 *
 * @param condition Condition that is assumed to hold.
 * @param message Message describing the assumption, used in diagnostics.
 */
#define ASSUME(condition, message) __attribute__((assume(condition)))
#endif

#ifndef ASSUME
#define ASSUME(condition, message) ASSUME(condition, message)
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
