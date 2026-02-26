#include "rw_lock.h"
#include <threads.h>

FDG_INTERNAL
interp_result_t rw_lock_init(rw_lock_t *this)
{
    *this = (rw_lock_t){};
    if (mtx_init(&this->lock, mtx_plain) != thrd_success)
    {
        return FDG_ERROR_FAILED_ALLOCATION;
    }
    if (cnd_init(&this->cond_read) != thrd_success)
    {
        mtx_destroy(&this->lock);
        return FDG_ERROR_FAILED_ALLOCATION;
    }
    if (cnd_init(&this->cond_write) != thrd_success)
    {
        cnd_destroy(&this->cond_read);
        mtx_destroy(&this->lock);
        return FDG_ERROR_FAILED_ALLOCATION;
    }
    return FDG_SUCCESS;
}

FDG_INTERNAL
void rw_lock_destroy(rw_lock_t *this)
{
    cnd_destroy(&this->cond_write);
    cnd_destroy(&this->cond_read);
    mtx_destroy(&this->lock);
    *this = (rw_lock_t){};
}

FDG_INTERNAL
void rw_lock_acquire_read(rw_lock_t *this)
{
    mtx_lock(&this->lock);
    while (this->writers > 0)
    {
        cnd_wait(&this->cond_read, &this->lock);
    }
    this->readers += 1;
    mtx_unlock(&this->lock);
}

FDG_INTERNAL
void rw_lock_acquire_write(rw_lock_t *this)
{
    mtx_lock(&this->lock);
    this->writers += 1;
    if (this->readers > 0 || this->writers > 1)
    {
        cnd_wait(&this->cond_write, &this->lock);
    }
    mtx_unlock(&this->lock);
}

FDG_INTERNAL
void rw_lock_release_read(rw_lock_t *this)
{
    mtx_lock(&this->lock);
    this->readers -= 1;
    if (this->readers == 0)
    {
        cnd_signal(&this->cond_write);
    }
    mtx_unlock(&this->lock);
}

FDG_INTERNAL
void rw_lock_release_write(rw_lock_t *this)
{
    mtx_lock(&this->lock);
    this->writers -= 1;
    if (this->writers == 0)
    {
        cnd_broadcast(&this->cond_read);
    }
    else
    {
        cnd_signal(&this->cond_write);
    }
    mtx_unlock(&this->lock);
}
