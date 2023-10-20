set(LLVM_TARGET_DEFINITIONS PylangAttributes.td)
mlir_tablegen(PylangAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PylangAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS PylangDialect.td)
mlir_tablegen(PylangDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PylangDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS PylangOps.td)
mlir_tablegen(PylangOps.h.inc -gen-op-decls)
mlir_tablegen(PylangOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS PylangTypes.td)
mlir_tablegen(PylangTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PylangTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(PylangTableGen)