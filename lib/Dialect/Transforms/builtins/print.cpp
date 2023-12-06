/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fmt/format.h>
#include "builtins.h"

void print(Tuple *tu, const char *sep, const char *end, bool flush) {
  Tuple t = *tu;
  if(t.size == 0) {
    // print() should print end string
    fmt::print("{}", end);
    return;
  }
  for(int i = 0; i < t.size; i ++) {
    if(i != 0) fmt::print("{}", sep);
    Unknown u = t.array[i];
    switch (u.type_id) {
    case 1:
      // Integer
      // TODO: change limited integer to unlimited integer
      fmt::print("{}", *((int*)u.ptr));
      break;
    case 2:
      // Double
      fmt::print("{}", *((double*)u.ptr));
      break;
    case 3:
      fmt::print("{}", *((bool*)u.ptr));
      break;
    case 4:
      fmt::print("{}", (char*)u.ptr);
      break;
    default:
      fmt::print("Unknown type currently. type id = {}\n", u.type_id);
    }
  }
  fmt::print("{}", end);
  if(flush)
    fflush(stdout); // TODO: change stdout to specific stream
}

void print(Tuple *tu) {
  Tuple t = *tu;
  if(t.size == 0) {
    // print() should print end string
    fmt::print("\n");
    return;
  }
  for(int i = 0; i < t.size; i ++) {
    if(i != 0) fmt::print(" ");
    Unknown u = t.array[i];
    switch (u.type_id) {
    case 1:
      // Integer
      // TODO: change limited integer to unlimited integer
      fmt::print("{}", *((int*)u.ptr));
      break;
    case 2:
      // Double
      fmt::print("{}", *((double*)u.ptr));
      break;
    case 3:
      fmt::print("{}", *((bool*)u.ptr));
      break;
    case 4:
      fmt::print("{}", *((char**)u.ptr));
      break;
    default:
      fmt::print("Unknown type currently. type id = {}\n", u.type_id);
    }
  }
  fmt::print("{}", "\n");
}
