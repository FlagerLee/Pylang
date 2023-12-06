/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYLANG_BUILTINS_H
#define PYLANG_BUILTINS_H

#include <cstdint>
#include <cstdio>

struct Unknown {
  void *ptr;
  unsigned int type_id;
};

struct Tuple {
  Unknown *array;
  int size;
};

// builtin functions
//extern "C" void print(Tuple *t, const char *sep = " ", const char *end = "\n", bool flush = false);
extern "C" void print(Tuple *t);
extern "C" std::size_t hash_int(int64_t i);
extern "C" std::size_t hash_double(double d);
extern "C" uint64_t hash_array(uint64_t key, const void *src, ssize_t src_sz);

#endif // PYLANG_BUILTINS_H
