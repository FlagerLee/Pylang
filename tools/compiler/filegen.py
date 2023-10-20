#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ast
import sys
from typing import Any

from PylangCompiler import Compiler, FileLineColLocAdaptor


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

    def visit_Assign(self, node: ast.Assign) -> Any:
        ssa = self.visit(node.value)
        for target in node.targets:
            assert isinstance(target, ast.Name)
            assert isinstance(target.ctx, ast.Store)
            self.name2ssa_map[target.id] = ssa

    def visit_For(self, node: ast.For) -> Any:
        pass

    def visit_While(self, node: ast.While) -> Any:
        pass

    def visit_If(self, node: ast.If) -> Any:
        pass

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        pass

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        pass

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        pass

    def visit_Compare(self, node: ast.Compare) -> Any:
        pass

    def visit_Constant(self, node: ast.Constant) -> Any:
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

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.name2ssa_map:
                raise NameError('name \'%s\' is not defined' % node.id)
            return self.name2ssa_map[node.id]

    def visit_Call(self, node: ast.Call) -> Any:
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
