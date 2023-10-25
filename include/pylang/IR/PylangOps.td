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
include "mlir/Interfaces/CastInterfaces.td"
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

def ConstantOp : Pylang_Op<"Constant", [ConstantLike, Pure]> {
    let summary = "Python Integer, Float, String and Boolean type constant";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs BaseType:$result);

    let hasFolder = 1;
    let assemblyFormat = "attr-dict $value `:` type($result)";
    let hasVerifier = 1;
}

def TupleOp : Pylang_Op<"Tuple", []> {
    let summary = "Python tuple creator";
    
    let arguments = (ins Variadic<BaseType>:$operands);
    let results = (outs Tuple:$results);

    let builders = [
      OpBuilder<(ins "ValueRange":$operands)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Pylang_Op<"Cast", [DeclareOpInterfaceMethods<CastOpInterface>]> {
    let summary = "cast type to another type";

    let arguments = (ins BaseType:$value);
    let results = (outs BaseType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Pylang_Op<"Func", [FunctionOpInterface, IsolatedFromAbove]> {
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

def CallOp : Pylang_Op<"Call",
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

def ReturnOp : Pylang_Op<"Return", [Pure, HasParent<"FuncOp">, Terminator]> {
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

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

def AddOp : Pylang_Op<"Add", [Commutative]> {
  let summary = "pylang add operation";
  let description = [{
    Pylang add op supports int + int, float + float.
    If adds like float + int is required, int should be casted to float first.
    If * + bool is required, bool should be casted to int or float.
    string + string, list + list, tuple + tuple belongs to concat operation.
    If an object defines __add__ method, then it should call __add__ instead of using AddOp
  }];
  let arguments = (ins AnyTypeOf<[Int, Float, Bool]>:$lhs, AnyTypeOf<[Int, Float, Bool]>:$rhs);
  let results = (outs AnyTypeOf<[Int, Float]>:$result);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ConcatOp : Pylang_Op<"Concat", []> {
  let summary = "pylang concatenate operation";
  let description = [{
    Pylang concat op supports these concatenates:
    1. string + string
    2. list + list
    3. tuple + tuple
  }];
  let arguments = (ins AnyTypeOf<[String, List, Tuple]>:$lhs, AnyTypeOf<[String, List, Tuple]>:$rhs);
  let results = (outs AnyTypeOf<[String, List, Tuple]>:$result);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def UnknownAddOp : Pylang_Op<"UAdd", []> {
  let summary = "pylang add unknown type operation";
  let description = [{
    Pylang unknown add op is used when the type of at least one of the
    two arguments is 'Unknown'.
  }];
  let arguments = (ins AnyTypeOf<[BaseType, Unknown]>:$lhs, AnyTypeOf<[BaseType, Unknown]>:$rhs);
  let results = (outs Unknown:$result);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}



#endif // PYLANG_OPS