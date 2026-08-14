//
// Created by jan on 29.9.2024.
//

#ifndef ERROR_H
#define ERROR_H
#include "common_defines.h"

/**
 * @brief Status and error codes returned by functions in this library.
 *
 * All functions that can fail return one of these values. A value of
 * FDG_SUCCESS means the operation completed successfully; any other value
 * is an error and describes why the operation failed.
 */
typedef enum
{
    FDG_SUCCESS = 0,                // The operation completed successfully.
    FDG_ERROR_NOT_IN_DOMAIN,        // An argument was not inside the domain of the function.
    FDG_ERROR_NOT_INCREASING,       // An input sequence was not monotonically increasing.
    FDG_ERROR_FAILED_ALLOCATION,    // Memory allocation failed.
    FDG_ERROR_BAD_SYSTEM,           // A system of equations could not be solved.
    FDG_ERROR_INVALID_ENUM,         // An enum argument had a value that was out of bounds.
    FDG_ERROR_NOT_IN_REGISTRY,      // An object was not found in the registry.
    FDG_ERROR_GEOID_OUT_OF_RANGE,   // A geo ID was not within the allowed range.
    FDG_ERROR_GEOID_NOT_VALID,      // A geo ID was not valid.
    FDG_ERROR_SURFACE_NOT_CLOSED,   // A surface did not have a closed boundary.
    FDG_ERROR_MATRIX_DIMS_MISMATCH, // Matrix dimensions do not match.

    FDG_ERROR_COUNT, // Total number of entries in this enum. Not a valid error code itself.
} fdg_result_t;

/**
 * @brief Get the symbolic name of a result code.
 *
 * The returned string is the identifier of the enum member, such as
 * "FDG_SUCCESS" or "FDG_ERROR_BAD_SYSTEM".
 *
 * @param error Result code to get the name for.
 * @return Statically allocated, null-terminated string with the name of the
 *         result code, or "UNKNOWN" if the value is outside of the valid range
 *         [0, FDG_ERROR_COUNT).
 */
FDG_INTERNAL
const char *fdg_error_str(fdg_result_t error);

/**
 * @brief Get a human-readable description of a result code.
 *
 * @param error Result code to get the message for.
 * @return Statically allocated, null-terminated string with a sentence
 *         describing what the result code means, or "UNKNOWN" if the value is
 *         outside of the valid range [0, FDG_ERROR_COUNT).
 */
FDG_INTERNAL
const char *fdg_error_msg(fdg_result_t error);

#endif // ERROR_H
