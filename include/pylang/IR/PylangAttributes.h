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

#ifndef PYLANG_PYLANGATTRIBUTES_H
#define PYLANG_PYLANGATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// Dialect Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "pylang/IR/PylangAttributes.h.inc"

#endif // PYLANG_PYLANGATTRIBUTES_H
