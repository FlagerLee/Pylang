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

#ifndef PYLANG_PYLANGTYPES_H
#define PYLANG_PYLANGTYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "pylang/IR/PylangTypes.h.inc"

#endif // PYLANG_PYLANGTYPES_H
