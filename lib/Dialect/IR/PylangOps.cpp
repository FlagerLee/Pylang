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

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "pylang/IR/PylangDialect.h"
#include "pylang/IR/PylangOps.h"
#include "pylang/IR/PylangTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#define GET_OP_CLASSES
#include "pylang/IR/PylangOps.cpp.inc"

using namespace mlir;

//===-------------------------------------------------------------------===//
// ConstantOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::ConstantOp::verify() {
  auto type = getType();
  auto val = getValue();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType, pylang::BoolType,
                 pylang::StringType>(type))
    return emitOpError("type must be one of pylang.int, pylang.float, "
                       "pylang.bool and pylang.string");
  if (!llvm::isa<IntegerAttr, FloatAttr, BoolAttr, StringAttr>(getValue()))
    return emitOpError(
        "value must be an integer, float, bool, or string attribute");
  if (llvm::isa<pylang::IntegerType>(type) &&
      !llvm::isa<mlir::IntegerAttr>(val))
    return emitOpError() << "pylang type " << type
                         << "should be initialized by mlir::IntegerAttr";
  if (llvm::isa<pylang::FloatType>(type) && !llvm::isa<mlir::FloatAttr>(val))
    return emitOpError() << "pylang type " << type
                         << "should be initialized by mlir::FloatAttr";
  if (llvm::isa<pylang::BoolType>(type) && !llvm::isa<mlir::BoolAttr>(val))
    return emitOpError() << "pylang type " << type
                         << "should be initialized by mlir::BoolAttr";
  if (llvm::isa<pylang::StringType>(type) && !llvm::isa<mlir::StringAttr>(val))
    return emitOpError() << "pylang type " << type
                         << "should be initialized by mlir::StringAttr";
  return success();
}

OpFoldResult pylang::ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

//===-------------------------------------------------------------------===//
// FuncOp
//===-------------------------------------------------------------------===//

ParseResult pylang::FuncOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void pylang::FuncOp::print(::mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===-------------------------------------------------------------------===//
// CallOp
//===-------------------------------------------------------------------===//

CallInterfaceCallable pylang::CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void pylang::CallOp::setCalleeFromCallable(
    ::mlir::CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

Operation::operand_range pylang::CallOp::getArgOperands() {
  return getOperands().drop_front();
}

MutableOperandRange pylang::CallOp::getArgOperandsMutable() {
  return MutableOperandRange(*this, 1, getOperands().size());
}

LogicalResult
pylang::CallOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != 0 && fnType.getNumResults() != 1)
    return emitOpError("number of results should be 0 or 1");

  if (fnType.getNumResults() != 1)
    return emitOpError("incorrect number of results for callee");

  if (fnType.getNumResults() == 1)
    if (getResult().getType() != fnType.getResult(0)) {
      auto diag = emitOpError("result type mismatch");
      diag.attachNote() << "      op result types: " << getResult().getType();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

//===-------------------------------------------------------------------===//
// ReturnOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::ReturnOp::verify() {
  unsigned numOperands = getNumOperands();
  if (numOperands != 0 && numOperands != 1)
    return failure();
  return success();
}
