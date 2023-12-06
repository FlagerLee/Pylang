//
// Created by flagerlee on 11/28/23.
//

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
