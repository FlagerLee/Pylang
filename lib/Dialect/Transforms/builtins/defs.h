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

#ifndef PYLANG_DEFS_H
#define PYLANG_DEFS_H

#include "config.h"

//===----------------------------------------------------------------------===//
// These definitions are copied from cpython/Include/cpython/pyhash.h
//===----------------------------------------------------------------------===//


#if PTR_BIT_WIDTH >= 8
#  define _PyHASH_BITS 61
#else
#  define _PyHASH_BITS 31
#endif

#define _PyHASH_MODULUS (((size_t)1 << _PyHASH_BITS) - 1)
#define _PyHASH_INF 314159

//===----------------------------------------------------------------------===//
// end copy from cpython/Include/cpython/pyhash.h
//===----------------------------------------------------------------------===//

#endif // PYLANG_DEFS_H
