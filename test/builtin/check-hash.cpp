//
// Created by flagerlee on 11/28/23.
//

#include "builtin.h"
#include <cstring>
#include <gtest/gtest.h>

TEST(Check_Hash, check_hash_double) {
  //EXPECT_EQ(hash_double(0.0), 0);
  //EXPECT_EQ(hash_double(1.0), 1);
  EXPECT_EQ(hash_double(1.1), 230584300921369601);
}

TEST(Check_Hash, check_hash_int) {
  EXPECT_EQ(hash_int(0), 0);
}

TEST(Check_Hash, check_hash_array) {
  EXPECT_EQ(hash_array(0, "hello", 5), 10142490492830962361);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}