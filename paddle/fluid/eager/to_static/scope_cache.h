// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "paddle/fluid/framework/scope.h"

using paddle::framework::Scope;

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace egr::jit {

class ScopeCache {
 public:
  ScopeCache() = default;
  ScopeCache(const ScopeCache&) = delete;
  ScopeCache(ScopeCache&&) = delete;
  ScopeCache& operator=(const ScopeCache&) = delete;

  static ScopeCache& Instance() {
    static ScopeCache cache;
    return cache;
  }

  Scope* Get(const int64_t key) {
    std::lock_guard<std::mutex> lock(mutex_);
    // if (cache_.find(key) == cache_.end()) {
    //   cache_.insert({key, {}});
    // }
    auto& cached_scopes = cache_[key];
    for (auto& scope : cached_scopes) {
      if (scope->CanReused()) {
        return scope.get();
      }
    }
    cached_scopes.emplace_back(std::make_unique<Scope>());
    return cached_scopes.back().get();
  }

 private:
  std::unordered_map<int64_t, std::vector<std::unique_ptr<Scope>>> cache_;
  std::mutex mutex_;
};
}  // namespace egr::jit
