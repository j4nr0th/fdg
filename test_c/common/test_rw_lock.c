//
// Created by jan on 2025-09-12.
//

#include "../../src/common/rw_lock.h"
#include "common.h"

enum
{
    THREAD_COUNT = 500,
    WRITER_ITERATIONS = 100,
};

typedef struct
{
    unsigned shared_data;
    unsigned writer_active;
    rw_lock_t lock;
} test_data_t;

int reader(void *arg)
{
    test_data_t *const data = (test_data_t *)arg;
    rw_lock_acquire_read(&data->lock);
    // Readers should not run when writer is active
    TEST_ASSERTION(data->writer_active == 0, "Writer is active");
    const unsigned expected = data->shared_data;
    thrd_sleep(&(struct timespec){.tv_sec = 0, .tv_nsec = 10000000}, NULL); // simulate reading
    TEST_ASSERTION(expected == data->shared_data, "Shared data is not correct");
    rw_lock_release_read(&data->lock);
    return 0;
}

int writer(void *arg)
{
    test_data_t *const data = (test_data_t *)arg;
    rw_lock_acquire_write(&data->lock);
    // Writer should have exclusive access
    TEST_ASSERTION(data->writer_active == 0, "Writer is active");
    data->writer_active = 1;
    for (unsigned i = 0; i < WRITER_ITERATIONS; ++i)
    {
        data->shared_data += 1;
        thrd_sleep(&(struct timespec){.tv_sec = 0, .tv_nsec = 1}, NULL); // simulate reading
    }
    data->writer_active = 0;
    rw_lock_release_write(&data->lock);
    return 0;
}

int main()
{
    test_data_t data = {.shared_data = 0, .writer_active = 0, .lock = {}};

    TEST_FDG_RESULT(rw_lock_init(&data.lock));

    thrd_t readers[THREAD_COUNT];
    thrd_t writers[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; ++i)
    {
        TEST_ASSERTION(thrd_create(readers + i, reader, &data) == thrd_success, "Failed to create reader thread");
        TEST_ASSERTION(thrd_create(writers + i, writer, &data) == thrd_success, "Failed to create writer thread");
    }

    for (int i = 0; i < THREAD_COUNT; ++i)
    {
        TEST_ASSERTION(thrd_join(readers[i], NULL) == thrd_success, "Failed to join reader thread");
        TEST_ASSERTION(thrd_join(writers[i], NULL) == thrd_success, "Failed to join writer thread");
    }

    TEST_ASSERTION(data.shared_data == THREAD_COUNT * WRITER_ITERATIONS, "Shared data is not correct");

    return 0;
}
