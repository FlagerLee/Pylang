#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF int KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ast
import sys

from PylangCompiler import Compiler, FileLineColLocAdaptor, CmpPredicate


class PythonVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        # init mlir config
        self.compiler = Compiler()
        self.compiler.createFunction("main", [], None, FileLineColLocAdaptor(filepath, 0, 0))
        self.fp = filepath

        self.name2ssa_map = {}

    def compile(self):
        with open(self.fp, "r") as f:
            self.visit(ast.parse(f.read()))

    def visit_Tuple(self, node: ast.Tuple) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)
        ssa_operands = []
        for elt in node.elts:
            ssa_operands.append(self.visit(elt))
        return self.compiler.createTuple(ssa_operands, node_loc)

    def visit_List(self, node: ast.List) -> int:
        pass

    def visit_Assign(self, node: ast.Assign) -> None:
        ssa = self.visit(node.value)
        for target in node.targets:
            assert isinstance(target, ast.Name)
            assert isinstance(target.ctx, ast.Store)
            self.name2ssa_map[target.id] = ssa

    def visit_For(self, node: ast.For) -> int:
        pass

    def visit_While(self, node: ast.While) -> int:
        pass

    def visit_If(self, node: ast.If) -> int:
        pass

    def visit_BoolOp(self, node: ast.BoolOp) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)
        ssas = []
        for value in node.values:
            ssas.append(self.visit(value))
        if isinstance(node.op, ast.And):
            return self.compiler.createAnd(ssas, node_loc)
        else:
            assert isinstance(node.op, ast.Or)
            return self.compiler.createOr(ssas, node_loc)

    def visit_BinOp(self, node: ast.BinOp) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return self.compiler.createAdd(lhs, rhs, node_loc)
        elif isinstance(op, ast.Sub):
            return self.compiler.createSub(lhs, rhs, node_loc)
        elif isinstance(op, ast.Mult):
            return self.compiler.createMul(lhs, rhs, node_loc)
        elif isinstance(op, ast.MatMult):
            raise RuntimeError('Unsupported operation MatMult')
        elif isinstance(op, ast.Div):
            return self.compiler.createDiv(lhs, rhs, node_loc)
        elif isinstance(op, ast.Mod):
            return self.compiler.createMod(lhs, rhs, node_loc)
        elif isinstance(op, ast.Pow):
            return self.compiler.createPow(lhs, rhs, node_loc)
        elif isinstance(op, ast.LShift):
            return self.compiler.createLShift(lhs, rhs, node_loc)
        elif isinstance(op, ast.RShift):
            return self.compiler.createRShift(lhs, rhs, node_loc)
        elif isinstance(op, ast.BitOr):
            return self.compiler.createBitOr(lhs, rhs, node_loc)
        elif isinstance(op, ast.BitXor):
            return self.compiler.createBitXor(lhs, rhs, node_loc)
        elif isinstance(op, ast.BitAnd):
            return self.compiler.createBitAnd(lhs, rhs, node_loc)
        elif isinstance(op, ast.FloorDiv):
            return self.compiler.createFloorDiv(lhs, rhs, node_loc)
        raise RuntimeError('Unsupported operation %s' % str(type(op)))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Invert):
            return self.compiler.createInvert(operand, node_loc)
        elif isinstance(node.op, ast.Not):
            return self.compiler.createNot(operand, node_loc)
        elif isinstance(node.op, ast.USub):
            return self.compiler.createUSub(operand, node_loc)
        else:
            assert isinstance(node.op, ast.UAdd)
            # TODO: what is the significance of uadd?
            return operand

    def visit_Compare(self, node: ast.Compare) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)

        def op2predicate(op: ast.cmpop) -> CmpPredicate:
            if isinstance(op, ast.Eq):
                return CmpPredicate.EQ
            elif isinstance(op, ast.NotEq):
                return CmpPredicate.NEQ
            elif isinstance(op, ast.Lt):
                return CmpPredicate.LT
            elif isinstance(op, ast.LtE):
                return CmpPredicate.LTE
            elif isinstance(op, ast.Gt):
                return CmpPredicate.GT
            elif isinstance(op, ast.GtE):
                return CmpPredicate.GTE
            else:
                raise RuntimeError('Unsupported compare op: %s' % str(op))
        operands = [self.visit(node.left)]
        for comparator in node.comparators:
            operands.append(self.visit(comparator))
        return self.compiler.createCmp(operands, list(map(op2predicate, node.ops)), node_loc)

    def visit_Constant(self, node: ast.Constant) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)
        if isinstance(node.value, bool):
            return self.compiler.createConstantBool(node.value, node_loc)
        elif isinstance(node.value, int):
            return self.compiler.createConstantInt(node.value, node_loc)
        elif isinstance(node.value, float):
            return self.compiler.createConstantFloat(node.value, node_loc)
        elif isinstance(node.value, str):
            return self.compiler.createConstantString(node.value, node_loc)
        else:
            raise RuntimeError('Unsupported constant: %s' % str(type(node.value)))

    def visit_Name(self, node: ast.Name) -> int:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.name2ssa_map:
                raise NameError('name \'%s\' is not defined' % node.id)
            return self.name2ssa_map[node.id]

    def visit_Call(self, node: ast.Call) -> int:
        node_loc = FileLineColLocAdaptor(self.fp, node.lineno, node.col_offset)
        func = node.func
        args = node.args
        arg_value = []
        for arg in args:
            arg_value.append(self.visit(arg))
        return self.compiler.createCall(func.id, arg_value, node_loc)


def filegenerator(filepath: str):
    visitor = PythonVisitor(filepath)
    visitor.compile()
    #visitor.compiler.dump()
    visitor.compiler.lowerToLLVM()
    #visitor.compiler.dump()
    #visitor.compiler.emitLLVMIR()
    visitor.compiler.runJIT()


if __name__ == '__main__':
    filegenerator(sys.argv[1])
