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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "pylang/IR/PylangDialect.td"
include "pylang/IR/PylangTypes.td"
include "pylang/IR/PylangAttributes.td"

class Pylang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pylang_Dialect, mnemonic, traits>;

class Pylang_Binary_Op<string mnemonic, list<Trait> traits = []> :
    Pylang_Op<mnemonic, traits> {
      let arguments = (ins AnyType:$lhs, AnyType:$rhs);
      let results = (outs AnyType:$result);
      let hasVerifier = 1;
    }

class Pylang_Binary_Same_Type_Op<string mnemonic, list<Trait> traits = []> :
    Pylang_Binary_Op<mnemonic, traits # [SameOperandsAndResultType]> {
      let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
    }

class Pylang_Binary_Different_Type_Op<string mnemonic, list<Trait> traits = []> :
    Pylang_Binary_Op<mnemonic, traits> {
      let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
    }

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
//
//                           Binary Operations
//
//===----------------------------------------------------------------------===//

def AddOp : Pylang_Binary_Same_Type_Op<"Add", [Commutative]> {
  let summary = "pylang add operation";
  let description = [{
    Pylang add op supports int + int, float + float.
    If adds like float + int is required, int should be casted to float first.
    If ? + bool is required, bool should be casted to int or float.
    string + string, list + list, tuple + tuple belongs to concat operation.
    If an object defines __add__ method, then it should call __add__ instead of using AddOp
  }];
}

def ConcatOp : Pylang_Binary_Same_Type_Op<"Concat", []> {
  let summary = "pylang concatenate operation";
  let description = [{
    Pylang concat op supports these concatenates:
    1. string + string
    2. list + list
    3. tuple + tuple
  }];
}

def UnknownAddOp : Pylang_Binary_Different_Type_Op<"UAdd", []> {
  let summary = "pylang add unknown type operation";
  let description = [{
    Pylang unknown add op is used when the type of at least one of the
    two arguments is 'Unknown'.
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, pylang::UnknownType::get($_builder.getContext()), lhs, rhs);
    }]>
  ];
}

def SubOp : Pylang_Binary_Same_Type_Op<"Sub", []> {
  let summary = "pylang subtraction operation";
  let description = [{
      Pylang sub op supports int - int, float - float.
      If subtractions like float - int is required, int should be casted to float first.
      If ? - bool is required, bool should be casted to int or float.
      If an object defines __sub__ method, then it should call __sub__ instead of using SubOp
  }];
}

def UnknownSubOp : Pylang_Binary_Different_Type_Op<"USub", []> {
  let summary = "pylang subtraction unknown type operation";
  let description = [{
    Pylang unknown sub op is used when the type of at least one of the
    two arguments is 'Unknown'.
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, pylang::UnknownType::get($_builder.getContext()), lhs, rhs);
    }]>
  ];
}

def MulOp : Pylang_Binary_Same_Type_Op<"Mul", []> {
  let summary = "pylang multiplication operation";
  let description = [{
      Pylang mul op supports int * int, float * float.
      If multiplications like float * int is required, int should be casted to float first.
      If ? * bool is required, bool should be casted to int or float.
      If an object defines __mul__ method, then it should call __mul__ instead of using MulOp
  }];
}

def DivOp : Pylang_Binary_Same_Type_Op<"Div", []> {
  let summary = "pylang division operation";
  let description = [{
      Pylang division op supports float / float.
      If divisions like float / int is required, int should be casted to float first.
      If ? / bool is required, bool should be casted to float.
      If an object defines __truediv__ method, then it should call __truediv__ instead of using DivOp
  }];
}

def ModOp : Pylang_Binary_Same_Type_Op<"Mod", []> {
  let summary = "pylang mod operation";
  let description = [{
      Pylang mod op supports int % int, float % float.
      If mods like float % int is required, int should be casted to float first.
      If ? % bool is required, bool should be casted to int or float.
      If an object defines __mod__ method, then it should call __mod__ instead of using ModOp
  }];
}

def PowOp : Pylang_Binary_Different_Type_Op<"Pow", []> {
  let summary = "pylang power operation";
  let description = [{
      Pylang pow op supports int/float ** int/float.
      If ? ** bool is required, bool should be casted to int.
      If an object defines __pow__ method, then it should call __pow__ instead of using PowOp
  }];
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      if(::llvm::isa<pylang::IntegerType>(lhs.getType()) && ::llvm::isa<pylang::IntegerType>(rhs.getType()))
        build($_builder, $_state, lhs.getType(), lhs, rhs);
      else if(::llvm::isa<pylang::FloatType>(lhs.getType()))
        build($_builder, $_state, lhs.getType(), lhs, rhs);
      else
        build($_builder, $_state, rhs.getType(), lhs, rhs);
    }]>
  ];
}

def LShiftOp : Pylang_Binary_Same_Type_Op<"LShift", []> {
  let summary = "pylang left shift operation";
  let description = [{
      Pylang lshift op supports int << int.
      If mods like int << bool is required, bool should be casted to int first.
      If an object defines __lshift__ method, then it should call __lshift__ instead of using LShiftOp
  }];
}

def RShiftOp : Pylang_Binary_Same_Type_Op<"RShift", []> {
  let summary = "pylang right shift operation";
  let description = [{
      Pylang rshift op supports int >> int.
      If mods like int >> bool is required, bool should be casted to int first.
      If an object defines __rshift__ method, then it should call __rshift__ instead of using RShiftOp
  }];
}

def BitOrOp : Pylang_Binary_Same_Type_Op<"BitOr", []> {
  let summary = "pylang bit or operation";
  let description = [{
      Pylang bitor op supports int | int.
      If bitors like int | bool is required, bool should be casted to int first.
      If an object defines __or__ method, then it should call __or__ instead of using BitOrOp
  }];
}

def BitXorOp : Pylang_Binary_Same_Type_Op<"BitXor", []> {
  let summary = "pylang bit xor operation";
  let description = [{
      Pylang bitxor op supports int ^ int.
      If bitxors like int ^ bool is required, bool should be casted to int first.
      If an object defines __xor__ method, then it should call __xor__ instead of using BitXorOp
  }];
}

def BitAndOp : Pylang_Binary_Same_Type_Op<"BitAnd", []> {
  let summary = "pylang bit and operation";
  let description = [{
      Pylang bitand op supports int & int.
      If bitands like int & bool is required, bool should be casted to int first.
      If an object defines __and__ method, then it should call __and__ instead of using BitAndOp
  }];
}

def FloorDivOp : Pylang_Binary_Same_Type_Op<"FloorDiv", []> {
  let summary = "pylang floor div operation";
  let description = [{
      Pylang floor div op supports int // int, float // float.
      If floor divs like float // int is required, int should be casted to float first.
      If ? // bool is required, bool should be casted to int or float.
      If an object defines __floordiv__ method, then it should call __floordiv__ instead of using FloorDivOp
  }];
}

#endif // PYLANG_OPS