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

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pylang/IR/PylangDialect.h"
#include "pylang/IR/PylangTypes.h"
#define GET_TYPEDEF_CLASSES
#include "pylang/IR/PylangTypes.cpp.inc"

using namespace mlir;
using namespace mlir::pylang;

void PylangDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylang/IR/PylangTypes.cpp.inc"
    >();
}