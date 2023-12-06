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

#include "pylang/Transforms/PylangBuiltinFunctionsUtils.h"
#include "config.h"
#include "mlir/Parser/Parser.h"
#include <map>
#include <unistd.h>

namespace mlir::pylang {

// tool functions
namespace {
std::map<std::string, std::string> builtin_definitions = {
    // print function
    {"print", "llvm.func external @print(%arg0: !llvm.ptr) "
              "attributes{func_wrapper = #pylang.FNARGS}"},
    // hash function
    {"hash_int", "llvm.func external @hash_int(%arg: i64) -> i64"},
    {"hash_double", "llvm.func external @hash_double(%arg: f64) -> i64"}};
} // namespace

LogicalResult createPylangBuiltinFunctionSymbolRef(PatternRewriter &rewriter,
                                                   ModuleOp module,
                                                   Location loc,
                                                   StringRef name) {
  if (module.lookupSymbol(name)) {
    emitError(loc) << "Builtin function '" << name << "' already exists";
    return failure();
  }
  auto func_def_iter = builtin_definitions.find(name.str());
  if (func_def_iter == builtin_definitions.end()) {
    emitError(loc) << "Function " << name << " is not a builtin function";
    return failure();
  }
  OwningOpRef<Operation *> func_res = parseSourceString(
      func_def_iter->second, ParserConfig(module.getContext()));
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.insert(func_res.get()->clone());
  return success();
}
} // namespace mlir::pylang