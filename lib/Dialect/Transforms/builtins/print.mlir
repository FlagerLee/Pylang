// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// print function accepts a tuple as input.
// tuple type should be defined as a llvm.ptr<tuple_entry>
//
// tuple_entry should be defined as:
// struct tuple_entry {
//     Unknown *data,
//     unsigned size
// }
//
// Unknown should be defined as:
// struct Unknown {
//     void *data,
//     unsigned type
// }

module {
llvm.mlir.global internal constant @none("None\00")
llvm.mlir.global internal constant @int_fmt("%d\00")
llvm.mlir.global internal constant @float_fmt("%lf\00")
llvm.mlir.global internal constant @true("True\00")
llvm.mlir.global internal constant @false("False\00")
llvm.mlir.global internal constant @string_fmt("%s\00")
llvm.mlir.global internal constant @ws(" \00")
llvm.mlir.global internal constant @nl("\n\00")
llvm.mlir.global internal constant @no_type("Error: no support for current type, typeid = %d\n\00")

llvm.func @printf(%arg: !llvm.ptr<i8>, ...) -> i32

llvm.func internal @print_none(%0: i32) {
    %addr = llvm.mlir.addressof @none : !llvm.ptr<array<5 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<5 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>) -> i32
    llvm.return
}

llvm.func internal @print_int(%0: i32, %1: i32) {
    %addr = llvm.mlir.addressof @int_fmt : !llvm.ptr<array<3 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<3 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>, i32) -> i32
    llvm.return
}

llvm.func internal @print_float(%0: i32, %1: f64) {
    %addr = llvm.mlir.addressof @float_fmt : !llvm.ptr<array<4 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<4 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>, f64) -> i32
    llvm.return
}

llvm.func internal @print_bool(%0: i32, %1: i1) {
    %zero = llvm.mlir.constant(0 : i1) : i1
    %is_true = llvm.icmp "eq" %zero, %1 : i1
    llvm.cond_br %is_true, ^false, ^true
^true:
    %addr_t = llvm.mlir.addressof @true : !llvm.ptr<array<5 x i8>>
    %ptr_t = llvm.getelementptr %addr_t[%0, %0] : (!llvm.ptr<array<5 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res_t = llvm.call @printf(%ptr_t) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>) -> i32
    llvm.return
^false:
    %addr_f = llvm.mlir.addressof @false : !llvm.ptr<array<6 x i8>>
    %ptr_f = llvm.getelementptr %addr_f[%0, %0] : (!llvm.ptr<array<6 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res_f = llvm.call @printf(%ptr_f) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>) -> i32
    llvm.return
}

llvm.func internal @print_string(%0: i32, %1: !llvm.ptr<i8>) {
    %addr = llvm.mlir.addressof @string_fmt : !llvm.ptr<array<3 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<3 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    llvm.return
}

llvm.func internal @print_ws(%0: i32) {
    %addr = llvm.mlir.addressof @ws : !llvm.ptr<array<2 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<2 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>) -> i32
    llvm.return
}

llvm.func internal @print_nl(%0: i32) {
    %addr = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<2 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>) -> i32
    llvm.return
}

llvm.func internal @print_no_type(%0: i32, %1: i32) {
    %addr = llvm.mlir.addressof @no_type : !llvm.ptr<array<49 x i8>>
    %ptr = llvm.getelementptr %addr[%0, %0] : (!llvm.ptr<array<49 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %res = llvm.call @printf(%ptr, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr<i8>, i32) -> i32
    llvm.return
}

llvm.func @print(%arg0: !llvm.ptr<struct<(ptr<struct<(ptr, i32)>>, i32)>>) attributes{func_wrapper = #pylang.FNARGS} {
    // get tuple
    %entry = llvm.load %arg0 : !llvm.ptr<struct<(ptr<struct<(ptr, i32)>>, i32)>>
    %entry_data = llvm.extractvalue %entry[0] : !llvm.struct<(ptr<struct<(ptr, i32)>>, i32)>
    %entry_size = llvm.extractvalue %entry[1] : !llvm.struct<(ptr<struct<(ptr, i32)>>, i32)>

    // create for-loop variable
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^for.cond(%0: i32)
^for.cond(%i: i32):                                 // preds = ^for.inc, entry
    %2 = llvm.icmp "slt" %i, %entry_size : i32
    llvm.cond_br %2, ^for.body, ^for.end
^for.body:                                          // preds = ^for.cond
    // extract unknown type
    %unknown_ptr = llvm.getelementptr inbounds %entry_data[%i] : (!llvm.ptr<struct<(ptr, i32)>>, i32) -> !llvm.ptr<struct<(ptr, i32)>>
    %unknown = llvm.load %unknown_ptr : !llvm.ptr<struct<(ptr, i32)>>
    %unknown_data = llvm.extractvalue %unknown[0] : !llvm.struct<(ptr, i32)>
    %unknown_type = llvm.extractvalue %unknown[1] : !llvm.struct<(ptr, i32)>

    // inference type
    // TODO: use type map to get type instead of switch
    llvm.switch %unknown_type : i32, ^sw.default [
        0: ^sw.0,
        1: ^sw.1,
        2: ^sw.2,
        3: ^sw.3,
        4: ^sw.4
    ]
^sw.0:
    llvm.call @print_none(%0) : (i32) -> ()
    llvm.br ^sw.epilog
^sw.1:
    %int = llvm.load %unknown_data : !llvm.ptr -> i32
    llvm.call @print_int(%0, %int) : (i32, i32) -> ()
    llvm.br ^sw.epilog
^sw.2:
    %float = llvm.load %unknown_data : !llvm.ptr -> f64
    llvm.call @print_float(%0, %float) : (i32, f64) -> ()
    llvm.br ^sw.epilog
^sw.3:
    %bool = llvm.load %unknown_data : !llvm.ptr -> i1
    llvm.call @print_bool(%0, %bool) : (i32, i1) -> ()
    llvm.br ^sw.epilog
^sw.4:
    %string = llvm.load %unknown_data : !llvm.ptr -> !llvm.ptr<i8>
    llvm.call @print_string(%0, %string) : (i32, !llvm.ptr<i8>) -> ()
    llvm.br ^sw.epilog
^sw.default:
    llvm.call @print_no_type(%0, %unknown_type) : (i32, i32) -> ()
    llvm.return
^sw.epilog:
    %3 = llvm.add %i, %1 : i32
    %is_end = llvm.icmp "eq" %3, %entry_size : i32
    llvm.cond_br %is_end, ^print_nl, ^print_ws
^print_ws:
    llvm.call @print_ws(%0) : (i32) -> ()
    llvm.br ^for.inc
^print_nl:
    llvm.call @print_nl(%0) : (i32) -> ()
    llvm.br ^for.inc
^for.inc:                                           // preds = ^for.body
    %inc = llvm.add %i, %1 : i32
    llvm.br ^for.cond(%inc: i32)
^for.end:                                           // preds = ^for.cond
    llvm.return
}
}