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

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PylangCompiler.h"
#include "pylang/IR/PylangDialect.h"
#include "pylang/IR/PylangOps.h"
#include "pylang/Transforms/Passes.h"
#include "pylang/Transforms/PylangConversion.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

namespace py = pybind11;

//===-------------------------------------------------------------------===//
// Python Bindings
//===-------------------------------------------------------------------===//

PYBIND11_MODULE(PylangCompiler, m) {
  py::class_<Compiler>(m, "Compiler")
      .def(py::init<>())
      .def("createConstantInt", &Compiler::createConstantInt, py::arg("value"),
           py::arg("loc"), "Create pylang.constant with python int")
      .def("createConstantFloat", &Compiler::createConstantFloat,
           py::arg("value"), py::arg("loc"),
           "Create pylang.constant with python float")
      .def("createConstantBool", &Compiler::createConstantBool,
           py::arg("value"), py::arg("loc"),
           "Create pylang.constant with python bool")
      .def("createConstantString", &Compiler::createConstantString,
           py::arg("value"), py::arg("loc"),
           "Create pylang.constant with python string")
      .def("createFunction", &Compiler::createFunction,
           py::arg("function_name"), py::arg("input_type"),
           py::arg("function_attribute"), py::arg("loc"), "Create pylang.func")
      .def("createCall", &Compiler::createCall, py::arg("callee_name"),
           py::arg("ssa_operands"), py::arg("loc"), "Create pylang.call")
      .def("createReturn", &Compiler::createReturn, py::arg("ssa"),
           py::arg("loc"), "Create pylang.return")
      .def("lowerToLLVM", &Compiler::lowerToLLVM, "Lower pylang to llvm")
      .def("dump", &Compiler::dump)
      .def("emitLLVMIR", &Compiler::emitLLVMIR)
      .def("runJIT", &Compiler::runJIT);
  py::class_<LocationAdaptor, FileLineColLocAdaptor>(m, "FileLineColLocAdaptor")
      .def(py::init<string, unsigned, unsigned>());
}

//===-------------------------------------------------------------------===//
// Compiler Function Implementation
//===-------------------------------------------------------------------===//

Compiler::Compiler() : has_main(false) {
  // Init context and register dialects
  ctx = new MLIRContext();
  ctx->getOrLoadDialect<pylang::PylangDialect>();

  builder = new OpBuilder(ctx);
  mod = ModuleOp::create(builder->getUnknownLoc());
  builder->setInsertionPointToEnd(mod.getBody());

  // Initialize builtin functions
  ssa2value_map.emplace_back();
  symbol_map.emplace_back();
  ssa_counter.emplace(0);
  auto &builtin_map = symbol_map[0];
  // print
  builtin_map.insert(
      std::make_pair("print", std::make_pair(SymbolRefAttr::get(ctx, "print"),
                                             pylang::NoneType::get(ctx))));
  // input
  builtin_map.insert(
      std::make_pair("input", std::make_pair(SymbolRefAttr::get(ctx, "input"),
                                             pylang::StringType::get(ctx))));

  // init pass
  pm = new PassManager(mod->getName());
  addPass();
}

Compiler::~Compiler() {
  delete pm;
  delete builder;
  delete ctx;
}

//===-------------------------------------------------------------------===//
// IR creator function
//===-------------------------------------------------------------------===//

unsigned Compiler::createConstantInt(int value, LocationAdaptor *loc_adaptor) {
  Value val = builder->create<pylang::ConstantOp>(
      loc_adaptor->getLoc(ctx), pylang::IntegerType::get(ctx, 32),
      IntegerAttr::get(IntegerType::get(ctx, 32), value));
  return insertSSAValue(val);
}

unsigned Compiler::createConstantFloat(float value,
                                       LocationAdaptor *loc_adaptor) {
  Value val = builder->create<pylang::ConstantOp>(
      loc_adaptor->getLoc(ctx), pylang::FloatType::get(ctx),
      FloatAttr::get(Float64Type::get(ctx), value));
  return insertSSAValue(val);
}

unsigned Compiler::createConstantBool(bool value,
                                      LocationAdaptor *loc_adaptor) {
  Value val = builder->create<pylang::ConstantOp>(loc_adaptor->getLoc(ctx),
                                                  pylang::BoolType::get(ctx),
                                                  BoolAttr::get(ctx, value));
  return insertSSAValue(val);
}

unsigned Compiler::createConstantString(const string &value,
                                        LocationAdaptor *loc_adaptor) {
  Value val = builder->create<pylang::ConstantOp>(loc_adaptor->getLoc(ctx),
                                                  pylang::StringType::get(ctx),
                                                  StringAttr::get(ctx, value + '\0'));
  return insertSSAValue(val);
}

void Compiler::createFunction(
    const string &name, const std::vector<MlirType> &input_types,
    std::optional<std::vector<MlirAttribute>> function_attributes,
    LocationAdaptor *loc_adaptor) {
  // unwrap input type and function attributes
  std::vector<NamedAttribute> attrs = {};
  /*
  if (function_attributes != std::nullopt)
    for (auto attr : function_attributes.value())
      attrs.emplace_back(dyn_cast<NamedAttribute>(unwrap(attr)));
      */
  std::vector<Type> types;
  for (auto ty : input_types)
    types.emplace_back(unwrap(ty));
  pylang::FuncOp func;
  if (name == "main" && !has_main) {
    // create main function
    assert(input_types.empty());
    func = builder->create<pylang::FuncOp>(
        loc_adaptor->getLoc(ctx), name, FunctionType::get(ctx, {}, {}), attrs);
    builder->setInsertionPointToStart(&func.front());
    createReturn(std::nullopt, loc_adaptor);
    has_main = true;
  } else {
    func = builder->create<pylang::FuncOp>(loc_adaptor->getLoc(ctx), name,
                                           FunctionType::get(ctx, types, {}),
                                           attrs);
    if (region_name.empty())
      symbol_map.back().insert(
          std::make_pair(name, std::make_pair(SymbolRefAttr::get(ctx, name),
                                              pylang::ListType::get(ctx))));
    else {
      unsigned region_size = region_name.size();
      string confused_name = (name == "main" && has_main) ? "fake_main" : name;
      for (unsigned sz = region_size; sz > 0; sz--)
        confused_name =
            fmt::format("{}_{}", confused_name, region_name[sz - 1]);
      auto &sym_map = symbol_map[symbol_map.size() - 1];
      auto it = sym_map.find(name);
      if (it == sym_map.end())
        sym_map.insert(std::make_pair(
            name, std::make_pair(SymbolRefAttr::get(ctx, confused_name),
                                 pylang::ListType::get(ctx))));
      ssa_counter.emplace(0);
      ssa2value_map.emplace_back();
      symbol_map.emplace_back();
      region_name.push_back(name);
    }
  }
  builder->setInsertionPointToStart(&func.front());
}

unsigned Compiler::createCall(const string &name,
                              const std::vector<unsigned int> &ssa_operands,
                              LocationAdaptor *loc_adaptor) {
  // find callee
  FlatSymbolRefAttr callee;
  Type ret_type;
  bool found = false;
  for (unsigned i = symbol_map.size(); i > 0; i--) {
    auto it = symbol_map.back().find(name);
    if (it != symbol_map[i - 1].end()) {
      callee = it->second.first;
      ret_type = it->second.second;
      found = true;
      break;
    }
  }
  if (!found) {
    emitError(loc_adaptor->getLoc(ctx))
        << "Symbol '" << name << "' not found, terminated\n";
    exit(1);
  }
  std::vector<Value> operands;
  auto &value_map = ssa2value_map.back();
  for (auto ssa : ssa_operands) {
    auto it = value_map.find(ssa);
    if (it == value_map.end()) {
      emitError(loc_adaptor->getLoc(ctx))
          << "Cannot find ssa value %" << ssa << "\n";
      exit(1);
    }
    operands.push_back(it->second);
  }
  if (llvm::isa<pylang::NoneType>(ret_type)) {
    builder->create<pylang::CallOp>(loc_adaptor->getLoc(ctx), TypeRange{},
                                    callee, operands);
    return 0;
  } else {
    auto value = builder->create<pylang::CallOp>(loc_adaptor->getLoc(ctx),
                                                 ret_type, callee, operands);
    return insertSSAValue(value.getResult());
  }
}

void Compiler::createReturn(std::optional<const unsigned> ssa,
                            LocationAdaptor *loc_adaptor) {
  if (ssa == std::nullopt) {
    builder->create<pylang::ReturnOp>(loc_adaptor->getLoc(ctx));
    return;
  }
  auto it = ssa2value_map.back().find(*ssa);
  if (it == ssa2value_map.back().end()) {
    emitError(loc_adaptor->getLoc(ctx))
        << "Cannot find ssa value %" << *ssa << "\n";
    exit(1);
  }
  builder->create<pylang::ReturnOp>(loc_adaptor->getLoc(ctx), it->second);
}

//===-------------------------------------------------------------------===//
// pass manage function
//===-------------------------------------------------------------------===//

void Compiler::addPass() {
  pm->addPass(pylang::createLowerToLLVMPass());
  pm->addPass(createConvertFuncToLLVMPass());
}

bool Compiler::lowerToLLVM() {
  auto res = succeeded(pm->run(mod));
  return res;
}

bool Compiler::emitLLVMIR() {
  mlir::registerBuiltinDialectTranslation(*mod->getContext());
  mlir::registerLLVMDialectTranslation(*mod->getContext());
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
  if(!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return false;
  }
  llvmModule->dump();
  return true;
}

void Compiler::runJIT() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerBuiltinDialectTranslation(*mod->getContext());
  mlir::registerLLVMDialectTranslation(*mod->getContext());
  auto maybeEngine = mlir::ExecutionEngine::create(mod);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
  }
}

unsigned Compiler::insertSSAValue(Value val) {
  auto &value_map = ssa2value_map.back();
  unsigned ssa = ssa_counter.top()++;
  value_map.insert(std::make_pair(ssa, val));
  return ssa;
}
