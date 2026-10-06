#ifndef FDG_RW_LOCK_H
#define FDG_RW_LOCK_H

#include "error.h"
#include <threads.h>

/**
 * @brief Reader-writer lock with blocking acquire and release.
 *
 * Multiple readers may hold the lock simultaneously, but a writer blocks
 * until all readers have released it, and no new readers may acquire the
 * lock while a writer is waiting or holding it. Writers are mutually
 * exclusive among themselves.
 *
 * A lock is used through the init/acquire/release/destroy functions in this
 * header. The struct may be stored by value anywhere in memory, but the
 * fields must not be accessed directly; use the provided functions instead.
 */
typedef struct rw_lock_t rw_lock_t;
struct rw_lock_t
{
    mtx_t lock;       // Mutex protecting the counters and condition variables.
    cnd_t cond_read;  // Condition variable for readers blocked behind a writer.
    cnd_t cond_write; // Condition variable for writers blocked behind readers or other writers.
    unsigned readers; // Number of readers currently holding the lock.
    unsigned writers; // Number of writers holding or waiting for the lock.
};

/**
 * @brief Initialize a reader-writer lock.
 *
 * The lock starts out unlocked, with no readers and no writers. There are no
 * prior requirements on the contents of the memory pointed to by `this`.
 *
 * @param this Pointer to the lock to initialize.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if the
 *         underlying mutex or condition variables could not be created.
 *         On failure, the lock is left in an uninitialized state and must not
 *         be used; it may, however, be re-initialized with another call.
 */
FDG_INTERNAL
fdg_result_t rw_lock_init(rw_lock_t *this);

/**
 * @brief Destroy a reader-writer lock.
 *
 * The lock must have been previously initialized with rw_lock_init and must
 * not be held by any readers or writers when this function is called. After
 * this call the lock is zeroed out and must not be used again unless it is
 * re-initialized.
 *
 * @param this Pointer to the lock to destroy.
 */
FDG_INTERNAL
void rw_lock_destroy(rw_lock_t *this);

/**
 * @brief Acquire the lock for reading.
 *
 * Blocks until no writer holds or waits for the lock, then increments the
 * reader count. Multiple readers may hold the lock concurrently. Any number
 * of read acquisitions must be matched by an equal number of
 * rw_lock_release_read calls.
 *
 * @param this Pointer to an initialized lock.
 */
FDG_INTERNAL
void rw_lock_acquire_read(rw_lock_t *this);

/**
 * @brief Acquire the lock for writing.
 *
 * Blocks until no readers hold the lock and no other writer holds or waits
 * for it, then grants exclusive access to the calling thread. Writers are
 * fair with respect to readers: once a writer is waiting, no further readers
 * acquire the lock ahead of it.
 *
 * @param this Pointer to an initialized lock.
 */
FDG_INTERNAL
void rw_lock_acquire_write(rw_lock_t *this);

/**
 * @brief Release a previously acquired read lock.
 *
 * The calling thread must hold a read lock acquired with rw_lock_acquire_read.
 * If this is the last reader, one waiting writer is woken up.
 *
 * @param this Pointer to an initialized lock.
 */
FDG_INTERNAL
void rw_lock_release_read(rw_lock_t *this);

/**
 * @brief Release a previously acquired write lock.
 *
 * The calling thread must hold a write lock acquired with rw_lock_acquire_write.
 * If no writers remain, all waiting readers are woken up; otherwise the next
 * waiting writer is signaled.
 *
 * @param this Pointer to an initialized lock.
 */
FDG_INTERNAL
void rw_lock_release_write(rw_lock_t *this);

#endif // FDG_RW_LOCK_H
