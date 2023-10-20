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

#include "pylang/IR/PylangDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "pylang/IR/PylangOps.h"

using namespace mlir;
using namespace mlir::pylang;

#include "pylang/IR/PylangDialect.cpp.inc"

void PylangDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "pylang/IR/PylangOps.cpp.inc"
      >();
}