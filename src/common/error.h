//
// Created by jan on 29.9.2024.
//

#ifndef ERROR_H
#define ERROR_H
#include "common_defines.h"

typedef enum
{
    FDG_SUCCESS = 0,
    FDG_ERROR_NOT_IN_DOMAIN,
    FDG_ERROR_NOT_INCREASING,
    FDG_ERROR_FAILED_ALLOCATION,
    FDG_ERROR_BAD_SYSTEM,
    FDG_ERROR_INVALID_ENUM,
    FDG_ERROR_NOT_IN_REGISTRY,
    FDG_ERROR_GEOID_OUT_OF_RANGE,
    FDG_ERROR_GEOID_NOT_VALID,
    FDG_ERROR_SURFACE_NOT_CLOSED,
    FDG_ERROR_OBJECT_CONNECTED_TWICE,
    FDG_ERROR_MATRIX_DIMS_MISMATCH,

    FDG_ERROR_COUNT,
} interp_result_t;

FDG_INTERNAL
const char *interp_error_str(interp_result_t error);

FDG_INTERNAL
const char *interp_error_msg(interp_result_t error);

#endif // ERROR_H
