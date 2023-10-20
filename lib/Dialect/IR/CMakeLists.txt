add_mlir_dialect_library(PylangDialect
        PylangAttributes.cpp
        PylangDialect.cpp
        PylangOps.cpp
        PylangTypes.cpp

        DEPENDS
        PylangTableGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRFunctionInterfaces
        )