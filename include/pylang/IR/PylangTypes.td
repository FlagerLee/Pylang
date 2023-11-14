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

#ifndef PYLANG_TYPES
#define PYLANG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "pylang/IR/PylangDialect.td"

class Pylang_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Pylang_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// all type has its id
// None: 0
// Int: 1
// Float: 2
// Bool: 3
// String: 4
// List: 5
// Tuple: 6
// Set: 7
// Dict: 8
// Unknown is not a real type


//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

def None : Pylang_Type<"None", "none"> {
    let description = [{
        this type should be lower to llvm.void
    }];
    let summary = "none";
}

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

def Int : Pylang_Type<"Integer", "int"> {
    let description = [{
        this type should be lower to one of these:
        1. IntegerType
        2. llvm.array
    }];
    let summary = "int";
    let parameters = (ins "unsigned":$width);
    let builders = [
            TypeBuilder<(ins "unsigned":$width)>
    ];

    let assemblyFormat = "$width";
}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

def Float : Pylang_Type<"Float", "float"> {
    let description = [{
        this type should be lower to Float64Type
    }];
    let summary = "float";
}

//===----------------------------------------------------------------------===//
// BooleanType
//===----------------------------------------------------------------------===//

def Bool : Pylang_Type<"Bool", "bool"> {
    let description = [{
        this type should be lower to IntegerType<1>
    }];
    let summary = "bool";
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def String : Pylang_Type<"String", "string"> {
    let description = [{
        this type should be lower to llvm.ptr<i8>
    }];
    let summary = "string";
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

def List : Pylang_Type<"List", "list"> {
    let description = [{
        this type should be lower to llvm.ptr<Entry>
        Entry is a struct:
        {
          Unknown *data;
          unsigned size;
          unsigned alloc_size;
        }
        data is an array points to objects.
    }];
    let summary = "list";
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

def Tuple : Pylang_Type<"Tuple", "tuple"> {
    let description = [{
        this type should be lower to llvm.ptr<Entry>
    }];
    let summary = "tuple";
}

//===----------------------------------------------------------------------===//
// SetType
//===----------------------------------------------------------------------===//

def Set : Pylang_Type<"Set", "set"> {
    let summary = "set";
}

//===----------------------------------------------------------------------===//
// DictType
//===----------------------------------------------------------------------===//

def Dict : Pylang_Type<"Dict", "dict"> {
    let summary = "dict";
}

//===----------------------------------------------------------------------===//
// UnknownType
//===----------------------------------------------------------------------===//

def Unknown : Pylang_Type<"Unknown", "unknown"> {
    let description = [{
        this type is used to represent those unpredictable type during compilation.
        this type should be lower to struct:
        {
          void *data;
          unsigned type;
        }
        type represents concrete type of this unknown type(which can only be known
        during runtime).
    }];
    let summary = "unknown";
}

#endif // PYLANG_TYPES