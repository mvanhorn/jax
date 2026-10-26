/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_CPU_THREAD_POOL_H_
#define JAXLIB_CPU_THREAD_POOL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/functional/function_ref.h"
#include "Eigen/ThreadPool"
#include "xla/backends/cpu/runtime/work_queue.h"
#include "xla/ffi/api/ffi.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/math/math_util.h"

namespace jax {

// Bridges `xla::ffi::ThreadPool` to `Eigen::ThreadPoolInterface` so XLA:CPU's
// work-stealing `xla::cpu::Worker::Parallelize` can schedule chunks onto the
// intra-op pool.
class FfiThreadPoolAdapter final : public Eigen::ThreadPoolInterface {
 public:
  explicit FfiThreadPoolAdapter(xla::ffi::ThreadPool thread_pool)
      : thread_pool_(thread_pool) {}

  void Schedule(std::function<void()> fn) override {
    thread_pool_.Schedule(std::move(fn));
  }

  int NumThreads() const override {
    return static_cast<int>(thread_pool_.num_threads());
  }

  int CurrentThreadId() const override { return -1; }

 private:
  xla::ffi::ThreadPool thread_pool_;
};

// Divides `count` items into chunks of at most `chunk_size` and invokes
// `fn(start, chunk_count)` on each chunk using XLA:CPU's work-stealing
// `Worker::Parallelize`.
inline xla::ffi::Error ParallelFor(
    xla::ffi::ThreadPool thread_pool, int64_t count, int64_t chunk_size,
    absl::FunctionRef<void(int64_t, int64_t)> fn) {
  if (count <= 0) {
    return xla::ffi::Error::Success();
  }
  if (chunk_size >= count) {
    fn(0, count);
    return xla::ffi::Error::Success();
  }
  const size_t num_chunks =
      static_cast<size_t>(tsl::MathUtil::CeilOfRatio(count, chunk_size));
  const size_t num_workers = std::min<size_t>(
      num_chunks,
      static_cast<size_t>(std::max<int64_t>(1, thread_pool.num_threads())));
  auto pool = std::make_shared<FfiThreadPoolAdapter>(thread_pool);
  Eigen::ThreadPoolInterface* pool_ptr = pool.get();
  tsl::BlockUntilReady(xla::cpu::Worker::Parallelize(
      pool_ptr, num_workers, num_chunks,
      [&, pool = std::move(pool)](size_t chunk_idx) {
        const int64_t start = static_cast<int64_t>(chunk_idx) * chunk_size;
        const int64_t chunk_count =
            std::min<int64_t>(chunk_size, count - start);
        fn(start, chunk_count);
      }));
  return xla::ffi::Error::Success();
}

}  // namespace jax

#endif  // JAXLIB_CPU_THREAD_POOL_H_
