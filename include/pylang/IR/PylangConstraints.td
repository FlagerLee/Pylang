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

include "pylang/IR/PylangTypes.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/DialectBase.td"


def PylangBaseType : AnyTypeOf<
        [None, Int, Float, Bool, String, List, Tuple, Set, Dict],
        "python base type">;
def Basic3Type : AnyTypeOf<[Int, Float, Bool]>;
def NumberType : AnyTypeOf<[Int, Float]>;
def Basic2Type : AnyTypeOf<[Int, Bool]>;