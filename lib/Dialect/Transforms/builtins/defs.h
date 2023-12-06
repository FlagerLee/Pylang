//
// Created by flagerlee on 11/29/23.
//

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
