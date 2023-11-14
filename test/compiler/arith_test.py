#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

i1 = 17
i2 = 3
f1 = 15.7
f2 = 2.9
b1 = True
b2 = False

# Add test
print(i1 + i2)
print(i1 + f1)
print(i1 + b1)
print(f1 + f2)
print(f1 + i1)
print(f1 + b1)
print(b1 + b2)
print(b1 + f1)
print(b1 + i1)
print()

# Sub test
print(i1 - i2)
print(i1 - f1)
print(i1 - b1)
print(f1 - i1)
print(f1 - f2)
print(f1 - b1)
print(b1 - i1)
print(b1 - f1)
print(b1 - b2)
print()

# Mul test
print(i1 * i2)
print(i1 * f1)
print(i1 * b1)
print(f1 * i1)
print(f1 * f2)
print(f1 * b1)
print(b1 * i1)
print(b1 * f1)
print(b1 * b2)
print()

# Div test
print(i1 / i2)
print(i1 / f1)
print(i1 / b1)
print(f1 / i1)
print(f1 / f2)
print(f1 / b1)
print(b1 / i1)
print(b1 / f1)
print(b2 / b1)
print()

# Modulo test
print(i1 % i2)
print(i1 % f1)
print(i1 % b1)
print(f1 % i2)
print(f1 % f2)
print(f1 % b1)
print(b1 % i1)
print(b1 % f1)
print(b2 % b1)
print()

# pow test
print(i1 ** i2)
print(i1 ** f1)
print(i1 ** b1)
print(f1 ** i1)
print(f1 ** f2)
print(f1 ** b1)
print(b1 ** i1)
print(b1 ** f1)
print(b1 ** b2)
print()

# lshift test
print(i1 << i2)
print(i1 << b1)
print(b1 << i1)
print(b1 << b2)
print()

# rshift test
print(i1 >> i2)
print(i1 >> b1)
print(b1 >> i1)
print(b1 >> b2)
print()

# bitor test
print(i1 | i2)
print(i1 | b1)
print(b1 | i1)
print(b1 | b2)
print()

# bitxor test
print(i1 ^ i2)
print(i1 ^ b1)
print(b1 ^ i1)
print(b1 ^ b2)
print()

# bitand test
print(i1 & i2)
print(i1 & b1)
print(b1 & i1)
print(b1 & b2)
print()

# floordiv test
print(i1 // i2)
print(i1 // f1)
print(i1 // b1)
print(f1 // i1)
print(f1 // f2)
print(f1 // b1)
print(b1 // i1)
print(b1 // f1)
print(b2 // b1)
print()

# usub test
print(-i1)
print(-f1)
print(-b1)
print()

# uadd test
print(+i1)
print(+f1)
print(+b1)
print()

# invert test
print(~i1)
print(~b1)
print()

# not test
print(not i1)
print(not f1)
print(not b1)
print(not b2)
print()

# compare test
print(2 < 3)
print(2 == 3)
print(2 != 3)
print(2 > 3)
