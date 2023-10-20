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

#ifndef PYLANG_ATTRIBUTES
#define PYLANG_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "pylang/IR/PylangDialect.td"

class Pylang_Attr<string name, string attrMnemonic, list<Trait> traits = []> : AttrDef<Pylang_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Function Wrapper
//===----------------------------------------------------------------------===//

def FUNCTION_ARGS_WRAPPER : Pylang_Attr<"FUNCTION_ARGS_WRAPPER", "FNARGS"> {
    let description = [{
        This attribute marks that a function accepts input *args.
        Functions who have this attributes should wrap its 
        arguments into a tuple.
    }];
}

def FUNCTION_KWARGS_WRAPPER : Pylang_Attr<"FUNCTION_KWARGS_WRAPPER", "FNKWARGS"> {
    let description = [{
        This attribute marks that a function accepts input *kwargs.
        Functions who have this attributes should wrap its
        arguments into a dict
    }];
}

#endif // PYLANG_ATTRIBUTES