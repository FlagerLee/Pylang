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

#include "builtins.h"
#include "defs.h"
#include <cmath>
#include <cstdio>
#include <cstring>

using std::size_t;

size_t hash_int(int64_t i) {
  if (i < 0)
    return (size_t)(-(-i % _PyHASH_MODULUS));
  else
    return (size_t)(i % _PyHASH_MODULUS);
}

//===----------------------------------------------------------------------===//
// Functions below are modified from cpython/Python/pyhash.c
//===----------------------------------------------------------------------===//
size_t hash_double(double v) {
  int e, sign;
  double m;
  size_t x, y;

  if (std::isinf(v))
    return v > 0 ? _PyHASH_INF : -_PyHASH_INF;
  if (std::isnan(v))
    return 0;

  m = frexp(v, &e);

  sign = 1;
  if (m < 0) {
    sign = -1;
    m = -m;
  }

  /* process 28 bits at a time;  this should work well both for binary
     and hexadecimal floating point. */
  x = 0;
  while (m) {
    x = ((x << 28) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - 28);
    m *= 268435456.0; /* 2**28 */
    e -= 28;
    y = (size_t)m; /* pull out integer part */
    m -= y;
    x += y;
    if (x >= _PyHASH_MODULUS)
      x -= _PyHASH_MODULUS;
  }

  /* adjust for the exponent;  first reduce it modulo _PyHASH_BITS */
  e = e >= 0 ? e % _PyHASH_BITS : _PyHASH_BITS - 1 - ((-1 - e) % _PyHASH_BITS);
  x = ((x << e) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - e);

  x = x * sign;
  return (size_t)x;
}

/* **************************************************************************
 <MIT License>
 Copyright (c) 2013  Marek Majkowski <marek@popcount.org>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 </MIT License>

 Original location:
    https://github.com/majek/csiphash/

 Solution inspired by code from:
    Samuel Neves (supercop/crypto_auth/siphash24/little)
    djb (supercop/crypto_auth/siphash24/little2)
    Jean-Philippe Aumasson (https://131002.net/siphash/siphash24.c)

 Modified for Python by Christian Heimes:
    - C89 / MSVC compatibility
    - _rotl64() on Windows
    - letoh64() fallback
*/

/* byte swap little endian to host endian
 * Endian conversion not only ensures that the hash function returns the same
 * value on all platforms. It is also required to for a good dispersion of
 * the hash values' least significant bits.
 */
#if PY_LITTLE_ENDIAN
#define _le64toh(x) ((uint64_t)(x))
#elif defined(__APPLE__)
#define _le64toh(x) OSSwapLittleToHostInt64(x)
#elif defined(HAVE_LE64TOH)
#include <endian.h>
#define _le64toh(x) le64toh(x)
#else
#define _le64toh(x)                                                            \
  (((uint64_t)(x) << 56) | (((uint64_t)(x) << 40) & 0xff000000000000ULL) |     \
   (((uint64_t)(x) << 24) & 0xff0000000000ULL) |                               \
   (((uint64_t)(x) << 8) & 0xff00000000ULL) |                                  \
   (((uint64_t)(x) >> 8) & 0xff000000ULL) |                                    \
   (((uint64_t)(x) >> 24) & 0xff0000ULL) |                                     \
   (((uint64_t)(x) >> 40) & 0xff00ULL) | ((uint64_t)(x) >> 56))
#endif

#ifdef _MSC_VER
#define ROTATE(x, b) _rotl64(x, b)
#else
#define ROTATE(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))
#endif

#define HALF_ROUND(a, b, c, d, s, t)                                           \
  a += b;                                                                      \
  c += d;                                                                      \
  b = ROTATE(b, s) ^ a;                                                        \
  d = ROTATE(d, t) ^ c;                                                        \
  a = ROTATE(a, 32);

#define DOUBLE_ROUND(v0, v1, v2, v3)                                           \
  HALF_ROUND(v0, v1, v2, v3, 13, 16);                                          \
  HALF_ROUND(v2, v1, v0, v3, 17, 21);                                          \
  HALF_ROUND(v0, v1, v2, v3, 13, 16);                                          \
  HALF_ROUND(v2, v1, v0, v3, 17, 21);

static uint64_t siphash24(uint64_t k0, uint64_t k1, const void *src,
                          ssize_t src_sz) {
  uint64_t b = (uint64_t)src_sz << 56;
  const uint8_t *in = (const uint8_t *)src;

  uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
  uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
  uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
  uint64_t v3 = k1 ^ 0x7465646279746573ULL;

  uint64_t t;
  uint8_t *pt;

  while (src_sz >= 8) {
    uint64_t mi;
    memcpy(&mi, in, sizeof(mi));
    mi = _le64toh(mi);
    in += sizeof(mi);
    src_sz -= sizeof(mi);
    v3 ^= mi;
    DOUBLE_ROUND(v0, v1, v2, v3);
    v0 ^= mi;
  }

  t = 0;
  pt = (uint8_t *)&t;
  switch (src_sz) {
  case 7:
    pt[6] = in[6]; /* fall through */
    __attribute__((fallthrough));
  case 6:
    pt[5] = in[5]; /* fall through */
    __attribute__((fallthrough));
  case 5:
    pt[4] = in[4]; /* fall through */
    __attribute__((fallthrough));
  case 4:
    memcpy(pt, in, sizeof(uint32_t));
    break;
  case 3:
    pt[2] = in[2]; /* fall through */
    __attribute__((fallthrough));
  case 2:
    pt[1] = in[1]; /* fall through */
    __attribute__((fallthrough));
  case 1:
    pt[0] = in[0]; /* fall through */
  }
  b |= _le64toh(t);

  v3 ^= b;
  DOUBLE_ROUND(v0, v1, v2, v3);
  v0 ^= b;
  v2 ^= 0xff;
  DOUBLE_ROUND(v0, v1, v2, v3);
  DOUBLE_ROUND(v0, v1, v2, v3);

  /* modified */
  t = (v0 ^ v1) ^ (v2 ^ v3);
  return t;
}
// TODO: Add randomize hash key. This is defined in cpython function _Py_HashRandomization_Init
uint64_t hash_array(uint64_t key, const void *src, ssize_t src_sz) {
  return siphash24(key, 0, src, src_sz);
}
