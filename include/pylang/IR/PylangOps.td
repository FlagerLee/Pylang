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

#ifndef PYLANG_OPS
#define PYLANG_OPS

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "pylang/IR/PylangDialect.td"
include "pylang/IR/PylangTypes.td"
include "pylang/IR/PylangAttributes.td"

class Pylang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pylang_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Pylang Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Pylang_Op<"constant", [ConstantLike, Pure]> {
    let summary = "Python Integer, Float, String and Boolean type constant";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs BaseType:$result);

    let hasFolder = 1;
    let assemblyFormat = "attr-dict $value `:` type($result)";
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Pylang_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";

    let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
      "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs), [{
        buildWithEntryBlock($_builder, $_state, name, type, attrs, type.getInputs());
      }]>
    ];
    let extraClassDeclaration = [{
      //===------------------------------------------------------------------===//
      // FunctionOpInterface Methods
      //===------------------------------------------------------------------===//

      /// Returns the argument types of this function.
      ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

      /// Returns the result types of this function.
      ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

      Region *getCallableRegion() { return &getBody(); }
    }];
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Pylang_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface>,
    MemRefsNormalizable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "pylang call operation";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<BaseType>:$operands);
  let results = (outs Optional<BaseType>:$result);
  let extraClassDeclaration = [{
    /// Returns the callee function type.
    FunctionType getCalleeFunctionType();
  }];
  let assemblyFormat = [{
      $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Pylang_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "pylang return operation";
  let arguments = (ins Variadic<Tuple>:$input);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CaptureOp
// Capture value from outer region(function)
//===----------------------------------------------------------------------===//


#endif // PYLANG_OPS