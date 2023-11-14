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

#include "pylang/Transforms/PylangConversion.h"
#include "pylang/IR/PylangDialect.h"
#include "pylang/IR/PylangOps.h"
#include "pylang/IR/PylangTypes.h"
#include "pylang/Transforms/Passes.h"
#include "pylang/Transforms/PylangBuiltinFunctionsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

#include <atomic>
#include <fmt/format.h>
#include <string>

using namespace mlir;
using pylang::PylangTypeConverter;

MLIRContext *pylang::PylangTypeConverter::getContext() const { return context; }

pylang::PylangTypeConverter::PylangTypeConverter(MLIRContext *context)
    : context(context) {
  addConversion([](mlir::Type type) { return type; });

  /// convert pylang.int to int or llvm.array<?xi64>
  addConversion([&](pylang::IntegerType type) {
    unsigned width = type.getWidth();
    return mlir::IntegerType::get(this->context, width,
                                  mlir::IntegerType::Signless);
  });

  /// convert pylang.float to float64
  addConversion([&](pylang::FloatType type) {
    return mlir::Float64Type::get(this->context);
  });

  /// convert pylang.bool to bool
  addConversion([&](pylang::BoolType type) {
    return mlir::IntegerType::get(this->context, 1);
  });

  /// convert pylang.string to llvm.ptr<i8>
  addConversion([&](pylang::StringType type) {
    return LLVM::LLVMPointerType::get(mlir::IntegerType::get(this->context, 8));
  });

  addConversion([this](pylang::NoneType type) {
    return LLVM::LLVMVoidType::get(this->context);
  });

  addConversion([this](pylang::UnknownType type) {
    auto ctx = this->context;
    return LLVM::LLVMStructType::getLiteral(
        ctx,
        {LLVM::LLVMPointerType::get(ctx), mlir::IntegerType::get(ctx, 32)});
  });

  addConversion([this](pylang::ListType type) {
    auto ctx = this->context;
    auto unknown_struct = LLVM::LLVMStructType::getLiteral(
        ctx,
        {LLVM::LLVMPointerType::get(ctx), mlir::IntegerType::get(ctx, 32)});
    auto entry = LLVM::LLVMStructType::getLiteral(
        ctx,
        {LLVM::LLVMPointerType::get(unknown_struct),
         mlir::IntegerType::get(ctx, 32), mlir::IntegerType::get(ctx, 32)});
    return LLVM::LLVMPointerType::get(entry);
  });

  addConversion([this](pylang::TupleType type) {
    auto ctx = this->context;
    auto unknown_struct = LLVM::LLVMStructType::getLiteral(
        ctx,
        {LLVM::LLVMPointerType::get(ctx), mlir::IntegerType::get(ctx, 32)});
    auto entry = LLVM::LLVMStructType::getLiteral(
        ctx, {LLVM::LLVMPointerType::get(unknown_struct),
              mlir::IntegerType::get(ctx, 32)});
    return LLVM::LLVMPointerType::get(entry);
  });
}

// tool functions
namespace {
std::atomic_uint global_counter = 0;

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                              StringRef value, ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }
  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

Value createTuple(Location loc, OpBuilder &builder, ValueRange args,
                  const PylangTypeConverter &converter) {
  auto ctx = builder.getContext();
  unsigned size = args.size();
  Value array_size = builder.create<arith::ConstantIntOp>(loc, size, 32);
  Type unknown_type = converter.convertType(pylang::UnknownType::get(ctx));
  Type unknown_type_ptr = LLVM::LLVMPointerType::get(unknown_type);
  Value unknown_array =
      builder.create<LLVM::AllocaOp>(loc, unknown_type_ptr, array_size, 8);
  for (unsigned i = 0; i < size; i++) {
    Value index = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value ptr = builder.create<LLVM::GEPOp>(
        loc, unknown_type_ptr, unknown_array, ValueRange{index}, true);
    builder.create<LLVM::StoreOp>(loc, args[i], ptr);
  }

  // create tuple struct
  Value tuple_struct = builder.create<LLVM::UndefOp>(
      loc, LLVM::LLVMStructType::getLiteral(
               ctx, {unknown_type_ptr, mlir::IntegerType::get(ctx, 32)}));
  tuple_struct = builder.create<LLVM::InsertValueOp>(
      loc, tuple_struct, unknown_array, mlir::DenseI64ArrayAttr::get(ctx, {0}));
  tuple_struct = builder.create<LLVM::InsertValueOp>(
      loc, tuple_struct, array_size, mlir::DenseI64ArrayAttr::get(ctx, {1}));

  // store tuple into stack
  Type tuple_type = converter.convertType(pylang::TupleType::get(ctx));
  Value tuple = builder.create<LLVM::AllocaOp>(
      loc, tuple_type, builder.create<arith::ConstantIntOp>(loc, 1, 32), 8);
  builder.create<LLVM::StoreOp>(loc, tuple_struct, tuple);
  return tuple;
}

LogicalResult convertToUnknown(Location loc, OpBuilder &builder, const Type &ty,
                               const Value &val, Value &casted) {
  int type_id = isa_and_nonnull<pylang::IntegerType>(ty)  ? 1
                : isa_and_nonnull<pylang::FloatType>(ty)  ? 2
                : isa_and_nonnull<pylang::BoolType>(ty)   ? 3
                : isa_and_nonnull<pylang::StringType>(ty) ? 4
                                                          : -1;
  if (type_id == -1) {
    emitError(loc) << "Unsupported type casting : " << ty << " cast to Unknown";
    return failure();
  }

  auto ctx = builder.getContext();
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  // create unknown struct
  Value unknown_struct = builder.create<LLVM::UndefOp>(
      loc,
      LLVM::LLVMStructType::getLiteral(ctx, {LLVM::LLVMPointerType::get(ctx),
                                             mlir::IntegerType::get(ctx, 32)}));

  // store value into stack
  Value value_storage = builder.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(val.getType()), one, 8);
  builder.create<LLVM::StoreOp>(loc, val, value_storage);
  value_storage = builder.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(ctx), value_storage);

  // fill unknown struct
  Value type_id_value = builder.create<arith::ConstantIntOp>(loc, type_id, 32);
  unknown_struct = builder.create<LLVM::InsertValueOp>(
      loc, unknown_struct, value_storage,
      mlir::DenseI64ArrayAttr::get(ctx, {0}));
  unknown_struct = builder.create<LLVM::InsertValueOp>(
      loc, unknown_struct, type_id_value,
      mlir::DenseI64ArrayAttr::get(ctx, {1}));
  casted = unknown_struct;
  return success();
}
} // namespace

// lowering operations
namespace {

struct ConstantOpLowering : public OpConversionPattern<pylang::ConstantOp> {
  using OpConversionPattern<pylang::ConstantOp>::OpConversionPattern;

  explicit ConstantOpLowering(MLIRContext *context, TypeConverter &converter)
      : OpConversionPattern(converter, context) {}

  LogicalResult
  matchAndRewrite(pylang::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    const auto &converter = *getTypeConverter<PylangTypeConverter>();

    auto attr = op->getAttr("value");
    auto res = op->getResult(0);
    auto res_type = res.getType();
    Value constValue;

    if (isa<pylang::IntegerType>(res_type)) {
      auto converted_type = converter.convertType(res_type);
      auto int_attr = dyn_cast<IntegerAttr>(attr);
      APInt value = int_attr.getValue();
      IntegerAttr typed_attr;
      if (value.getBitWidth() != converted_type.getIntOrFloatBitWidth())
        typed_attr = IntegerAttr::get(
            converted_type, APInt(converted_type.getIntOrFloatBitWidth(),
                                  value.getNumWords(), value.getRawData()));
      else
        typed_attr = int_attr;
      constValue =
          rewriter.create<arith::ConstantOp>(loc, converted_type, typed_attr);
    }
    if (isa<pylang::FloatType>(res_type)) {
      auto typed_attr = dyn_cast<FloatAttr>(attr);
      constValue = rewriter.create<arith::ConstantOp>(
          loc, converter.convertType(res_type), typed_attr);
    }
    if (isa<pylang::BoolType>(res_type)) {
      auto typed_attr = dyn_cast<BoolAttr>(attr);
      constValue = rewriter.create<arith::ConstantOp>(
          loc, converter.convertType(res_type), typed_attr);
    }
    if (isa<pylang::StringType>(res_type)) {
      auto typed_attr = dyn_cast<StringAttr>(attr);
      const uint global_id = global_counter++;
      auto parentModule = op->getParentOfType<ModuleOp>();
      constValue = getOrCreateGlobalString(
          loc, rewriter, StringRef(fmt::format("global_string_{}", global_id)),
          typed_attr, parentModule);
    }

    rewriter.replaceOp(op, constValue);
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<pylang::FuncOp> {
  using OpConversionPattern<pylang::FuncOp>::OpConversionPattern;

  explicit FuncOpLowering(MLIRContext *context, TypeConverter &converter)
      : OpConversionPattern(converter, context) {}

  LogicalResult
  matchAndRewrite(pylang::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    const auto &converter = *getTypeConverter<PylangTypeConverter>();

    auto result_types = op->getResultTypes();
    SmallVector<Type> arg_types{};

    if (failed(converter.convertTypes(op.getArgumentTypes(), arg_types)))
      emitError(loc) << "Cannot convert type";

    LLVM::LLVMFunctionType func_type;
    if (result_types.size() == 0)
      func_type =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), arg_types);
    else if (result_types.size() == 1)
      func_type = LLVM::LLVMFunctionType::get(
          converter.convertType(result_types[0]), arg_types);
    auto func = rewriter.create<LLVM::LLVMFuncOp>(loc, op.getName(), func_type);

    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());

    rewriter.replaceOp(op, func);
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<pylang::CallOp> {
  using OpConversionPattern<pylang::CallOp>::OpConversionPattern;

  explicit CallOpLowering(MLIRContext *context, TypeConverter &converter)
      : OpConversionPattern(converter, context) {}

  LogicalResult
  matchAndRewrite(pylang::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    const auto &converter = *getTypeConverter<PylangTypeConverter>();

    // verify call op return value number
    auto result_types = op->getResultTypes();
    if (result_types.size() != 1 && result_types.size() != 0) {
      emitError(loc, "Call op must have 0 or 1 return value");
      return failure();
    }

    // get callee symbol
    auto parent_module = op->getParentOfType<ModuleOp>();
    auto symbol_name = op.getCallee();
    FlatSymbolRefAttr symbol_ref;
    if (parent_module.lookupSymbol(symbol_name)) {
      symbol_ref = op.getCalleeAttr();
    } else {
      if (failed(pylang::createPylangBuiltinFunctionSymbolRef(
              rewriter, parent_module, loc, symbol_name))) {
        emitError(loc) << "Error when calling function '" << symbol_name << "'";
        return failure();
      } else
        symbol_ref = op.getCalleeAttr();
    }

    // create call
    LLVM::CallOp call;
    auto func_wrapper_attr =
        parent_module.lookupSymbol(symbol_name)->getAttr("func_wrapper");
    if (isa_and_nonnull<pylang::FUNCTION_ARGS_WRAPPERAttr>(func_wrapper_attr)) {
      // wrap arguments in a tuple
      auto args = adaptor.getOperands();
      auto types = op->getOperandTypes();
      assert(args.size() == types.size());
      std::vector<Value> casted_args;
      for (unsigned i = 0; i < args.size(); i++) {
        Type ty = types[i];
        if (llvm::isa<pylang::UnknownType>(ty))
          casted_args.emplace_back(args[i]);
        else {
          Value casted;
          if (failed(
                  convertToUnknown(loc, rewriter, types[i], args[i], casted))) {
            emitError(loc) << "Error when casting type '" << args[i].getType()
                           << "'";
            return failure();
          }
          casted_args.emplace_back(casted);
        }
      }
      call = result_types.size() == 0
                 ? rewriter.create<LLVM::CallOp>(
                       loc, TypeRange{}, symbol_ref,
                       createTuple(loc, rewriter, casted_args, converter))
                 : rewriter.create<LLVM::CallOp>(
                       loc, converter.convertType(result_types[0]), symbol_ref,
                       createTuple(loc, rewriter, casted_args, converter));
    } else {
      // no wrapper
      call = rewriter.create<LLVM::CallOp>(
          loc, converter.convertType(result_types[0]), symbol_ref,
          adaptor.getOperands());
    }
    rewriter.replaceOp(op, call);
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<pylang::ReturnOp> {
  using OpConversionPattern<pylang::ReturnOp>::OpConversionPattern;

  explicit ReturnOpLowering(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto ret = rewriter.create<LLVM::ReturnOp>(loc, adaptor.getOperands());

    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<pylang::CastOp> {
  using OpConversionPattern<pylang::CastOp>::OpConversionPattern;

  explicit CastOpLowering(MLIRContext *context, TypeConverter &converter)
      : OpConversionPattern<pylang::CastOp>(converter, context) {}

  LogicalResult
  matchAndRewrite(pylang::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    const auto &converter = *getTypeConverter<PylangTypeConverter>();

    Type from_ty = op.getValue().getType(), to_ty = op.getResult().getType();
    Value input = adaptor.getValue();
    Value res = input;
    // float -> *
    if (isa<pylang::FloatType>(from_ty)) {
      // float -> int
      if (isa<pylang::IntegerType>(to_ty)) {
        res = rewriter.create<arith::FPToSIOp>(
            loc, converter.convertType(to_ty), input);
      }
      // float -> bool
      else if (isa<pylang::BoolType>(to_ty)) {
        Value fzero = rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(converter.convertType(from_ty), 0.0));
        res = rewriter.create<arith::CmpFOp>(loc, converter.convertType(to_ty),
                                             arith::CmpFPredicate::ONE, input,
                                             fzero);
      }
    }
    // int -> *
    else if (isa<pylang::IntegerType>(from_ty)) {
      // int -> float
      if (isa<pylang::FloatType>(to_ty)) {
        res = rewriter.create<arith::SIToFPOp>(
            loc, converter.convertType(to_ty), input);
      }
      // int -> bool
      else if (isa<pylang::BoolType>(to_ty)) {
        Value zero = rewriter.create<arith::ConstantOp>(
            loc, IntegerAttr::get(converter.convertType(from_ty), 0));
        res = rewriter.create<arith::CmpIOp>(loc, converter.convertType(to_ty),
                                             arith::CmpIPredicate::ne, input,
                                             zero);
      }
    }
    // bool -> *
    else if (isa<pylang::BoolType>(from_ty)) {
      // bool -> float
      if (isa<pylang::FloatType>(to_ty)) {
        res = rewriter.create<arith::UIToFPOp>(
            loc, converter.convertType(to_ty), input);
      }
      // bool -> int
      else if (isa<pylang::IntegerType>(to_ty)) {
        res = rewriter.create<arith::ExtUIOp>(loc, converter.convertType(to_ty),
                                              input);
      }
    }
    if (res == input)
      return emitError(loc)
             << "Unsupported type cast: from " << from_ty << " to " << to_ty;

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct AddOpLowering : public OpConversionPattern<pylang::AddOp> {
  using OpConversionPattern<pylang::AddOp>::OpConversionPattern;

  explicit AddOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res;
    if (isa<pylang::IntegerType>(op.getResult().getType()))
      res = rewriter.create<arith::AddIOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    else
      res = rewriter.create<arith::AddFOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct SubOpLowering : public OpConversionPattern<pylang::SubOp> {
  using OpConversionPattern<pylang::SubOp>::OpConversionPattern;

  explicit SubOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res;
    if (isa<pylang::IntegerType>(op.getResult().getType()))
      res = rewriter.create<arith::SubIOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    else
      res = rewriter.create<arith::SubFOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MulOpLowering : public OpConversionPattern<pylang::MulOp> {
  using OpConversionPattern<pylang::MulOp>::OpConversionPattern;

  explicit MulOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res;
    if (isa<pylang::IntegerType>(op.getResult().getType()))
      res = rewriter.create<arith::MulIOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    else
      res = rewriter.create<arith::MulFOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct DivOpLowering : public OpConversionPattern<pylang::DivOp> {
  using OpConversionPattern<pylang::DivOp>::OpConversionPattern;

  explicit DivOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res =
        rewriter.create<arith::DivFOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ModOpLowering : public OpConversionPattern<pylang::ModOp> {
  using OpConversionPattern<pylang::ModOp>::OpConversionPattern;

  explicit ModOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::ModOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // a % b equals (a mod b + b) mod b
    Value res;
    if (isa<pylang::IntegerType>(op.getResult().getType())) {
      Value mod = rewriter.create<arith::RemSIOp>(loc, adaptor.getLhs(),
                                                  adaptor.getRhs());
      Value add = rewriter.create<arith::AddIOp>(loc, mod, adaptor.getRhs());
      res = rewriter.create<arith::RemSIOp>(loc, add,
                                            adaptor.getRhs());
    }
    else {
      Value mod = rewriter.create<arith::RemFOp>(loc, adaptor.getLhs(),
                                                  adaptor.getRhs());
      Value add = rewriter.create<arith::AddFOp>(loc, mod, adaptor.getRhs());
      res = rewriter.create<arith::RemFOp>(loc, add,
                                            adaptor.getRhs());
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct PowOpLowering : public OpConversionPattern<pylang::PowOp> {
  using OpConversionPattern<pylang::PowOp>::OpConversionPattern;

  explicit PowOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res;
    if (isa<pylang::FloatType>(op.getRhs().getType()))
      res = rewriter.create<math::PowFOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    else if (isa<pylang::IntegerType>(op.getLhs().getType()))
      res = rewriter.create<math::IPowIOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    else
      res = rewriter.create<math::FPowIOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct LShiftOpLowering : public OpConversionPattern<pylang::LShiftOp> {
  using OpConversionPattern<pylang::LShiftOp>::OpConversionPattern;

  explicit LShiftOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::LShiftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res = rewriter.create<arith::ShLIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct RShiftOpLowering : public OpConversionPattern<pylang::RShiftOp> {
  using OpConversionPattern<pylang::RShiftOp>::OpConversionPattern;

  explicit RShiftOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::RShiftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res = rewriter.create<arith::ShRSIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BitOrOpLowering : public OpConversionPattern<pylang::BitOrOp> {
  using OpConversionPattern<pylang::BitOrOp>::OpConversionPattern;

  explicit BitOrOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::BitOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res = rewriter.create<arith::OrIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BitXorOpLowering : public OpConversionPattern<pylang::BitXorOp> {
  using OpConversionPattern<pylang::BitXorOp>::OpConversionPattern;

  explicit BitXorOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::BitXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res = rewriter.create<arith::XOrIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BitAndOpLowering : public OpConversionPattern<pylang::BitAndOp> {
  using OpConversionPattern<pylang::BitAndOp>::OpConversionPattern;

  explicit BitAndOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::BitAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res = rewriter.create<arith::AndIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct FloorDivOpLowering : public OpConversionPattern<pylang::FloorDivOp> {
  using OpConversionPattern<pylang::FloorDivOp>::OpConversionPattern;

  explicit FloorDivOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::FloorDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // a // b equals to (a - a mod b) / b
    Value res;
    if (isa<pylang::FloatType>(op.getRhs().getType())) {
      Value rem = rewriter.create<arith::RemFOp>(loc, adaptor.getLhs(), adaptor.getRhs());
      Value sub = rewriter.create<arith::SubFOp>(loc, adaptor.getLhs(), rem);
      res = rewriter.create<arith::DivFOp>(loc, sub, adaptor.getRhs());
    }
    else {
      Value rem = rewriter.create<arith::RemSIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
      Value sub = rewriter.create<arith::SubIOp>(loc, adaptor.getLhs(), rem);
      res = rewriter.create<arith::DivSIOp>(loc, sub, adaptor.getRhs());
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct CmpOpLowering : public OpConversionPattern<pylang::CmpOp> {
  using OpConversionPattern<pylang::CmpOp>::OpConversionPattern;

  explicit CmpOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(pylang::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    Value res;
    if(isa<pylang::IntegerType>(op.getLhs().getType())) {
      arith::CmpIPredicate predicate;
      switch (op.getPredicate()) {
      case pylang::CmpPredicate::EQ:
        predicate = arith::CmpIPredicate::eq;
        break;
      case pylang::CmpPredicate::NEQ:
        predicate = arith::CmpIPredicate::ne;
        break;
      case pylang::CmpPredicate::LT:
        predicate = arith::CmpIPredicate::slt;
        break;
      case pylang::CmpPredicate::LTE:
        predicate = arith::CmpIPredicate::sle;
        break;
      case pylang::CmpPredicate::GT:
        predicate = arith::CmpIPredicate::sgt;
        break;
      case pylang::CmpPredicate::GTE:
        predicate = arith::CmpIPredicate::sge;
        break;
      }
      res = rewriter.create<arith::CmpIOp>(loc, predicate, adaptor.getLhs(), adaptor.getRhs());
    }
    else {
      arith::CmpFPredicate predicate;
      switch (op.getPredicate()) {
      case pylang::CmpPredicate::EQ:
        predicate = arith::CmpFPredicate::OEQ;
        break;
      case pylang::CmpPredicate::NEQ:
        predicate = arith::CmpFPredicate::ONE;
        break;
      case pylang::CmpPredicate::LT:
        predicate = arith::CmpFPredicate::OLT;
        break;
      case pylang::CmpPredicate::LTE:
        predicate = arith::CmpFPredicate::OLE;
        break;
      case pylang::CmpPredicate::GT:
        predicate = arith::CmpFPredicate::OGT;
        break;
      case pylang::CmpPredicate::GTE:
        predicate = arith::CmpFPredicate::OGE;
        break;
      }
      res = rewriter.create<arith::CmpFOp>(loc, predicate, adaptor.getLhs(), adaptor.getRhs());
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};
} // namespace

static void
populateLowerPylangConversionPatterns(RewritePatternSet &patterns,
                                      PylangTypeConverter &&converter) {
  patterns.add<ConstantOpLowering>(patterns.getContext(), converter);
  patterns.add<FuncOpLowering>(patterns.getContext(), converter);
  patterns.add<CallOpLowering>(patterns.getContext(), converter);
  patterns.add<ReturnOpLowering>(patterns.getContext());
  patterns.add<CastOpLowering>(patterns.getContext(), converter);
  // lower binary op
  patterns.add<AddOpLowering>(patterns.getContext());
  patterns.add<SubOpLowering>(patterns.getContext());
  patterns.add<MulOpLowering>(patterns.getContext());
  patterns.add<DivOpLowering>(patterns.getContext());
  patterns.add<ModOpLowering>(patterns.getContext());
  patterns.add<PowOpLowering>(patterns.getContext());
  patterns.add<LShiftOpLowering>(patterns.getContext());
  patterns.add<RShiftOpLowering>(patterns.getContext());
  patterns.add<BitOrOpLowering>(patterns.getContext());
  patterns.add<BitXorOpLowering>(patterns.getContext());
  patterns.add<BitAndOpLowering>(patterns.getContext());
  patterns.add<FloorDivOpLowering>(patterns.getContext());
  patterns.add<CmpOpLowering>(patterns.getContext());
}

namespace {
class LowerPylangPass
    : public PassWrapper<LowerPylangPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPylangPass);

  LowerPylangPass() = default;

  LowerPylangPass(const LowerPylangPass &) {}

  StringRef getArgument() const final { return "lower-pylang"; }

  StringRef getDescription() const final { return "Lower Pylang Dialect"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, LLVM::LLVMDialect, math::MathDialect>();
    target.addIllegalDialect<pylang::PylangDialect>();

    RewritePatternSet patterns(context);
    populateLowerPylangConversionPatterns(patterns,
                                          PylangTypeConverter(context));

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<pylang::PylangDialect, arith::ArithDialect, func::FuncDialect,
                memref::MemRefDialect, LLVM::LLVMDialect, math::MathDialect>();
  }
};
} // namespace

void pylang::registerLowerPylangPass() { PassRegistration<LowerPylangPass>(); }
std::unique_ptr<Pass> pylang::createLowerToLLVMPass() {
  return std::make_unique<LowerPylangPass>();
}