get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(pylang-opt pylang-opt.cpp)

target_link_libraries(pylang-opt
        PRIVATE
        ${dialect_libs}
        ${conversion_libs}
		MLIROptLib
        PylangDialect
        PylangConversion
        )
