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

#include "config.h"
#include "mlir/Parser/Parser.h"
#include "pylang/Transforms/PylangBuiltinFunctionsUtils.h"
#include <unistd.h>

namespace mlir::pylang {

// tool functions
namespace {
// merge source module to target module
void combineModuleAndGetSymbol(OpBuilder &combinedModuleBuilder,
                               ModuleOp &target, ModuleOp &source) {
  combinedModuleBuilder.setInsertionPointToStart(target.getBody());
  for (auto &op : source.getOps()) {
    auto symbol_op = dyn_cast<SymbolOpInterface>(op);
    if (!symbol_op)
      continue;
    StringRef sym_name = symbol_op.getName();
    if (target.lookupSymbol(sym_name))
      continue;
    combinedModuleBuilder.insert(op.clone());
  }
}
} // namespace

LogicalResult createPylangBuiltinFunctionSymbolRef(PatternRewriter &rewriter,
                                                   ModuleOp module,
                                                   Location loc,
                                                   StringRef name) {
  if (module.lookupSymbol(name)) {
    emitError(loc) << "Builtin function '" << name << "' already exists";
    return failure();
  }
  std::string function_file_path = get_builtin_function_file_path(name.str());
  if(access(function_file_path.c_str(), F_OK) == -1) {
    emitError(loc) << "Builtin function file " << function_file_path << " does not exists or unavailable";
    return failure();
  }
  OwningOpRef<Operation *> mod_ref =
      parseSourceFile(function_file_path,
                      ParserConfig(module.getContext()));
  auto mod = llvm::dyn_cast<ModuleOp>(mod_ref.get());
  OpBuilder::InsertionGuard guard(rewriter);
  combineModuleAndGetSymbol(rewriter, module, mod);
  return success();
}
} // namespace mlir::pylang