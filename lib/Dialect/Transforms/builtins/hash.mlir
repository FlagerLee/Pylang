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

module {
llvm.func @frexp(%arg1: f64, %arg2: !llvm.ptr<i32>) -> f64
llvm.func @isinf(%arg: f64) -> i1
llvm.func @isnan(%arg: f64) -> i1

llvm.func external @hash_int(%arg: i64) -> i64 {
    // hash(x) = x % (2 ** _PyHASH_BITS - 1)
    // TODO: make _PyHASH_BITS configurable.
    %0 = llvm.mlir.constant(0: i64) : i64
    %neg = llvm.icmp "slt" %arg, %0 : i64
    %sign, %x = scf.if %neg -> (i1, i64) {
        %false = llvm.mlir.constant(0: i1) : i1
        %new_arg = llvm.sub %0, %arg : i64
        scf.yield %false, %new_arg : i1, i64
    } else {
        %true = llvm.mlir.constant(1: i1) : i1
        scf.yield %true, %arg : i1, i64
    }
    %P = llvm.mlir.constant(2305843009213693951: i64) : i64 // %P = 2 ** 61 - 1
    %mod = llvm.srem %x, %P: i64
    %res = scf.if %sign -> i64 {
        scf.yield %mod : i64
    } else {
        %res = llvm.sub %0, %mod : i64
        scf.yield %res : i64
    }
    llvm.return %res : i64
}

llvm.func external @hash_double(%arg: f64) -> i64 {
    %0 = arith.constant 0 : i64
    // if isnan(arg)
    %nan = llvm.call @isnan(%arg) : (f64) -> i1
    llvm.cond_br %nan, ^isnan, ^notnan
^isnan:
    llvm.return %0 : i64
^notnan:
    // if isinf(arg)
    %inf = llvm.call @isinf(%arg) : (f64) -> i1
    llvm.cond_br %inf, ^isinf, ^notinf
^isinf:
    %inf_num = arith.constant 314159 : i64 // TODO: make _PyHASH_INF configurable
    llvm.return %inf_num : i64
^notinf:
    %true = arith.constant 1 : i1
    %false = arith.constant 0 : i1
    %1 = arith.constant 1 : i64
    %f0 = arith.constant 0.0 : f64
    %e_ptr = llvm.alloca %1 x i32 : (i64) -> !llvm.ptr<i32>
    %org_m = llvm.call @frexp(%arg, %e_ptr) : (f64, !llvm.ptr<i32>) -> f64
    %e = llvm.load %e_ptr : !llvm.ptr<i32>
    // if m < 0, zhen set m = -m and mark
    %org_m_is_neg = llvm.fcmp "olt" %org_m, %f0 : f64
    %sign, %m = scf.if %org_m_is_neg -> (i1, f64) {
        %pos_m = llvm.fsub %f0, %org_m : f64
        scf.yield %false, %pos_m : i1, f64
    } else {
        scf.yield %true, %org_m : i1, f64
    }
    // x = 0
    %P = llvm.mlir.constant(2305843009213693951: i64) : i64 // %P = _PyHASH_MODULUS
    %x, %no_use, %e_after = scf.while (%x = %0, %m_arg = %m, %e_arg = %e) : (i64, f64, i32) -> (i64, f64, i32) {
        // condition
        %condition = llvm.fcmp "oeq" %m_arg, %f0 : f64
        scf.condition(%condition) %x, %m_arg, %e_arg : i64, f64, i32
    } do {
    ^bb0(%x: i64, %m_arg: f64, %e_arg: i32):
        // x = ((x << 28) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - 28)
        %c28 = arith.constant 28 : i64
        %x0 = arith.shli %x, %c28 : i64
        %x1 = arith.andi %x0, %P : i64
        %c33 = arith.constant 33 : i64 // _PyHASH_BITS = 61, _PyHASH_BITS - 28 = 33
        %x2 = arith.shrui %x, %c33 : i64
        %x3 = arith.ori %x1, %x2 : i64 // x3 is x
        // m *= 268435456.0 (2 ** 28)
        %c2p28f = arith.constant 268435456.0 : f64
        %m0 = arith.mulf %m_arg, %c2p28f : f64
        // e -= 28
        %c28i32 = arith.constant 28 : i32
        %e0 = arith.subi %e_arg, %c28i32 : i32
        // y = (Py_uhash_t)m
        // convert m to int
        %y = arith.fptoui %m0 : f64 to i64
        // m -= y
        %yf = arith.uitofp %y : i64 to f64
        %m1 = arith.subf %m0, %yf : f64
        // x += y
        %x4 = arith.addi %x3, %y : i64
        // if (x >= _PyHASH_MODULUS)
        //     x -= _PyHASH_MODULUS
        %xExceed = arith.cmpi uge, %x4, %P : i64
        %x5 = scf.if %xExceed -> i64 {
            %x5 = arith.subi %x4, %P: i64
            scf.yield %x5 : i64
        } else {
            scf.yield %x4 : i64
        }
        scf.yield %x5, %m1, %e0 : i64, f64, i32
    }
    %032 = arith.constant 0 : i32
    %egt0 = arith.cmpi sge, %e_after, %032 : i32
    %final_e = scf.if %egt0 -> i32 {
        // e % _PyHASH_BITS
        %c28 = arith.constant 28 : i32
        %e0 = arith.remsi %e_after, %c28 : i32
        scf.yield %e0 : i32
    } else {
        // _PyHASH_BITS - 1 - ((-1 - e) % _PyHASH_BITS)
        %neg1 = arith.constant -1 : i32
        %e0 = arith.subi %neg1, %e_after : i32 // -1 - e
        %c28 = arith.constant 28 : i32
        %c27 = arith.constant 27 : i32
        %e1 = arith.remsi %e0, %c28 : i32
        %e2 = arith.subi %c27, %e1 : i32
        scf.yield %e2 : i32
    }
    // x = ((x << e) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - e)
    %final_ei64 = arith.extsi %final_e : i32 to i64
    %c28 = arith.constant 28 : i64
    %x0 = arith.shli %x, %final_ei64 : i64
    %x1 = arith.andi %x0, %P : i64
    %x2 = arith.subi %c28, %final_ei64 : i64
    %x3 = arith.shrui %x, %x2 : i64
    %x4 = arith.ori %x1, %x3 : i64
    // x = x * sign
    %x5 = scf.if %sign -> i64 {
        scf.yield %x4 : i64
    } else {
        %x5 = arith.subi %0, %x4 : i64
        scf.yield %x5 : i64
    }
    llvm.return %x5 : i64
}
}