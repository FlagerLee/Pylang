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

#ifndef PYLANG_PYLANGCOMPILER_H
#define PYLANG_PYLANGCOMPILER_H

#include <atomic>
#include <map>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

#include "mlir/CAPI/Support.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/IR/Module.h"

#include "pylang/IR/Pylang.h"

using std::string;
using namespace mlir;

//===-------------------------------------------------------------------===//
// Class Definitions
//===-------------------------------------------------------------------===//

class Compiler;
class LocationAdaptor;
class FileLineColLocAdaptor;

class Compiler {
public:
  explicit Compiler();
  ~Compiler();

  //===-------------------------------------------------------------------===//
  // IR creator function
  //===-------------------------------------------------------------------===//

  // TODO: current function can only accept 32-bit integer. Use string to pass
  // unlimited precision integer.
  unsigned createConstantInt(int, LocationAdaptor *);
  unsigned createConstantFloat(float, LocationAdaptor *);
  unsigned createConstantBool(bool, LocationAdaptor *);
  unsigned createConstantString(const string &, LocationAdaptor *);
  unsigned createTuple(const std::vector<unsigned> &, LocationAdaptor *);

  // createFunction does not accept return type because return type must be
  // tuple
  void createFunction(const string &, const std::vector<MlirType> &,
                      std::optional<std::vector<MlirAttribute>>,
                      LocationAdaptor *);
  unsigned createCall(const string &, const std::vector<unsigned> &,
                      LocationAdaptor *);

  unsigned createAdd(unsigned, unsigned, LocationAdaptor *);
  unsigned createSub(unsigned, unsigned, LocationAdaptor *);
  unsigned createMul(unsigned, unsigned, LocationAdaptor *);
  unsigned createDiv(unsigned, unsigned, LocationAdaptor *);
  unsigned createMod(unsigned, unsigned, LocationAdaptor *);
  unsigned createPow(unsigned, unsigned, LocationAdaptor *);
  unsigned createLShift(unsigned, unsigned, LocationAdaptor *);
  unsigned createRShift(unsigned, unsigned, LocationAdaptor *);
  unsigned createBitOr(unsigned, unsigned, LocationAdaptor *);
  unsigned createBitXor(unsigned, unsigned, LocationAdaptor *);
  unsigned createBitAnd(unsigned, unsigned, LocationAdaptor *);
  unsigned createFloorDiv(unsigned, unsigned, LocationAdaptor *);
  unsigned createAnd(std::vector<unsigned>, LocationAdaptor *);
  unsigned createOr(std::vector<unsigned>, LocationAdaptor *);
  unsigned createCmp(std::vector<unsigned>, std::vector<mlir::pylang::CmpPredicate>, LocationAdaptor *);
  unsigned createInvert(unsigned, LocationAdaptor *);
  unsigned createNot(unsigned, LocationAdaptor *);
  unsigned createUAdd(unsigned, LocationAdaptor *);
  unsigned createUSub(unsigned, LocationAdaptor *);


  void createReturn(std::optional<const unsigned>, LocationAdaptor *);

  //===-------------------------------------------------------------------===//
  // pass manage function
  //===-------------------------------------------------------------------===//
  void addPass();
  bool lowerToLLVM();
  bool emitLLVMIR();
  void runJIT();
  void dump() { mod.dump(); }

private:
  bool has_main;

  MLIRContext *ctx;
  OpBuilder *builder;
  ModuleOp mod;

  std::vector<std::map<unsigned, Value>> ssa2value_map;
  std::vector<std::map<string, std::pair<FlatSymbolRefAttr, Type>>> symbol_map;
  std::stack<std::atomic<unsigned>> ssa_counter;
  std::vector<string> region_name;

  PassManager *pm;

  //===-------------------------------------------------------------------===//
  // tool function
  //===-------------------------------------------------------------------===//
  unsigned insertSSAValue(Value val);
  Value getValueBySSA(unsigned ssa, Location loc);
};

class LocationAdaptor {
public:
  virtual Location getLoc(MLIRContext *ctx) = 0;
  virtual ~LocationAdaptor() = default;
};

class FileLineColLocAdaptor : LocationAdaptor {
public:
  FileLineColLocAdaptor(string file_name, unsigned line, unsigned column)
      : file_name(std::move(file_name)), line(line), column(column) {}
  ~FileLineColLocAdaptor() override = default;
  Location getLoc(MLIRContext *ctx) override {
    return FileLineColLoc::get(ctx, file_name, line, column);
  }

  string file_name;
  unsigned line, column;
};

#endif // PYLANG_PYLANGCOMPILER_H
