# Pylang

Pylang currently supports function `print` and base types `int(32-bit)`, `float(64-bit)`, `bool`, `string`

Requires build mlir and fmt first:
```shell
# llvm-mlir
cd third-party/llvm-project
mkdir build && cd build
cmake ../llvm -GNinja -DLLVM_ENABLE_PROJECTS="mlir"
ninja
```
```shell
# fmt
cd third-party/fmt
mkdir build && cd build
cmake ../llvm -GNinja
ninja
```

build:
```shell
mkdir build && cd build
cmake .. -GNinja
ninja
```

usage:
```shell
#add compiled pylang library into python path
export PYTHONPATH=${PROJECT_DIR}/build/lib
python3 tools/compiler/filegen.py path/to/python/file
```

