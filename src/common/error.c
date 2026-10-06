//
// Created by jan on 29.9.2024.
//

#include "error.h"

#define ERROR_ENUM_ENTRY(entry, msg) [(entry)] = {#entry, (msg)}

static const struct
{
    const char *str, *msg;
} error_messages[FDG_ERROR_COUNT] = {
    ERROR_ENUM_ENTRY(FDG_SUCCESS, "Success"),
    ERROR_ENUM_ENTRY(FDG_ERROR_NOT_IN_DOMAIN, "Argument was not inside the domain."),
    ERROR_ENUM_ENTRY(FDG_ERROR_FAILED_ALLOCATION, "Could not allocate desired amount of memory."),
};

const char *fdg_error_str(fdg_result_t error)
{
    if (error < 0 || error >= FDG_ERROR_COUNT)
        return "UNKNOWN";
    return error_messages[error].str;
}

const char *fdg_error_msg(fdg_result_t error)
{
    if (error < 0 || error >= FDG_ERROR_COUNT)
        return "UNKNOWN";
    return error_messages[error].msg;
}
