//
// Created by flagerlee on 11/28/23.
//

#ifndef PYLANG_BUILTIN_H
#define PYLANG_BUILTIN_H

#include <cstdint>
#include <cstdio>

extern "C" std::size_t hash_double(double d);
extern "C" std::size_t hash_int(int64_t i);
extern "C" uint64_t hash_array(uint64_t key, const void *src, ssize_t src_sz);

#endif // PYLANG_BUILTIN_H
