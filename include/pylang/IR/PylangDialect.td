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

#ifndef PYLANG_BASE
#define PYLANG_BASE

include "mlir/IR/OpBase.td"

def Pylang_Dialect : Dialect {
    let name = "pylang";
    let cppNamespace = "::mlir::pylang";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
    private:
        // Register Python Attributes
        void registerAttributes();
        // Register Python Types
        void registerTypes();

    public:
    }];
}

#endif // PYLANG_BASE