set(LLVM_OPTIONAL_SOURCES
        PylangBuiltinFunctionsUtils.cpp
        PylangConversion.cpp
        )

add_mlir_dialect_library(PylangConversion
        PylangBuiltinFunctionsUtils.cpp
        PylangConversion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/pylang

        LINK_LIBS PUBLIC
        PylangDialect
        MLIRTransformUtils
        fmt::fmt
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRFuncToLLVM
        )
