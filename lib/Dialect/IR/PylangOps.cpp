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
// TupleOp
//===-------------------------------------------------------------------===//

ParseResult pylang::TupleOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

void pylang::TupleOp::print(::mlir::OpAsmPrinter &p) {
  Operation *op = *this;
  p << " " << op->getOperands();
  p.printOptionalAttrDict(op->getAttrs());
  p << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    p << resultType;
    return;
  }

  // Otherwise, print a functional type.
  p.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

LogicalResult pylang::TupleOp::verify() {
  if (!isa<pylang::TupleType>(getResult().getType()))
    return emitOpError("TupleOp should return tuple type");
  return success();
}

//===-------------------------------------------------------------------===//
// CastOp
//===-------------------------------------------------------------------===//

ParseResult pylang::CastOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  Type type, res_type;
  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type) || parser.parseKeyword("to") ||
      parser.parseType(res_type))
    return failure();

  if (parser.resolveOperand(operand, type, result.operands))
    return failure();
  result.addTypes(res_type);

  return success();
}

void pylang::CastOp::print(OpAsmPrinter &p) {
  p << " " << getOperand();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getOperand().getType() << " to " << getResult().getType();
}

LogicalResult pylang::CastOp::verify() {
  if (getOperand().getType() == getResult().getType())
    return emitOpError() << "operand type(" << getOperand().getType()
                         << ") shouldn't equal to result type("
                         << getResult().getType() << ")";
  return success();
}

bool pylang::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type input = inputs[0];
  Type output = outputs[0];
  bool input_is_unknown = isa<pylang::UnknownType>(input);
  bool output_is_unknown = isa<pylang::UnknownType>(output);
  if (input_is_unknown ^ output_is_unknown)
    return true;
  // cast target type can only be (without unknown) bool, int, float, string
  // TODO: add cast target byte, etc.
  if (!isa<pylang::BoolType, pylang::IntegerType, pylang::FloatType,
           pylang::StringType>(output))
    return false;
  if (isa<pylang::NoneType>(input))
    return isa<pylang::BoolType>(output);
  if (isa<pylang::StringType>(input))
    return isa<pylang::IntegerType, pylang::BoolType>(output);
  if (isa<pylang::BoolType, pylang::IntegerType, pylang::FloatType>(input))
    return true;
  return isa<pylang::StringType>(output);
}

//===-------------------------------------------------------------------===//
// FuncOp
//===-------------------------------------------------------------------===//

ParseResult pylang::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
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

void pylang::FuncOp::print(OpAsmPrinter &p) {
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

//===-------------------------------------------------------------------===//
// AddOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::AddOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(res_type))
    return emitOpError("AddOp supports int and float only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// ConcatOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::ConcatOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::StringType, pylang::ListType, pylang::TupleType>(
          res_type))
    return emitOpError("ConcatOp can only concatenate string, list and tuple, "
                       "current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// UnknownAddOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::UnknownAddOp::verify() {
  operand_type_range types = getOperandTypes();
  Type res_type = getType();
  if (types.size() != 2)
    return emitOpError("incorrect number of operands");

  if (!llvm::isa<pylang::UnknownType>(types[0]) &&
      !llvm::isa<pylang::UnknownType>(types[1]))
    return emitOpError(
        "UAddOp requires at least one unknown types as its operands");
  if (!llvm::isa<pylang::UnknownType>(res_type))
    return emitOpError("UAddOp result type must be unknown, current type is ")
           << res_type;

  return success();
}

//===-------------------------------------------------------------------===//
// SubOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::SubOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(res_type))
    return emitOpError("SubOp supports int and float only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// UnknownSubOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::UnknownSubOp::verify() {
  operand_type_range types = getOperandTypes();
  Type res_type = getType();
  if (types.size() != 2)
    return emitOpError("incorrect number of operands");

  if (!llvm::isa<pylang::UnknownType>(types[0]) &&
      !llvm::isa<pylang::UnknownType>(types[1]))
    return emitOpError(
        "USubOp requires at least one unknown types as its operands");
  if (!llvm::isa<pylang::UnknownType>(res_type))
    return emitOpError("USubOp result type must be unknown, current type is ")
           << res_type;

  return success();
}

//===-------------------------------------------------------------------===//
// MulOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::MulOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(res_type))
    return emitOpError("MulOp supports int and float only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// DivOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::DivOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(res_type))
    return emitOpError("DivOp supports int and float only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// ModOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::ModOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(res_type))
    return emitOpError("ModOp supports int and float only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// PowOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::PowOp::verify() {
  operand_type_range types = getOperandTypes();
  Type res_type = getType();
  if (types.size() != 2)
    return emitOpError("incorrect number of operands");

  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(types[0]) &&
      !llvm::isa<pylang::IntegerType, pylang::FloatType>(types[1]))
    return emitOpError(
        "PowOp requires at least one unknown types as its operands");
  if (llvm::isa<pylang::IntegerType>(types[0]) &&
      llvm::isa<pylang::IntegerType>(types[1]) &&
      !llvm::isa<pylang::IntegerType>(res_type))
    return emitOpError("In PowOp, int ** int should return int, current return "
                       "type is ")
           << res_type;
  if (!llvm::isa<pylang::FloatType>(res_type))
    return emitOpError("PowOp result type must be unknown, current type is ")
           << res_type;

  return success();
}

//===-------------------------------------------------------------------===//
// LShiftOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::LShiftOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType>(res_type))
    return emitOpError("LShiftOp supports int only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// RShiftOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::RShiftOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType>(res_type))
    return emitOpError("RShiftOp supports int only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// BitOrOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::BitOrOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType>(res_type))
    return emitOpError("BitOr supports int only, current type is ") << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// BitXorOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::BitXorOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType>(res_type))
    return emitOpError("BitXorOp supports int only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// BitAndOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::BitAndOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType>(res_type))
    return emitOpError("BitAndOp supports int only, current type is ")
           << res_type;
  return success();
}

//===-------------------------------------------------------------------===//
// FloorDivOp
//===-------------------------------------------------------------------===//

LogicalResult pylang::FloorDivOp::verify() {
  Type res_type = getResult().getType();
  if (!llvm::isa<pylang::IntegerType, pylang::FloatType>(res_type))
    return emitOpError("FloorDivOp supports int and float only, "
                       "current type is ")
           << res_type;
  return success();
}
