# compile builtin functions to libpylang.so
file(GLOB BUILTIN_SRCS "*.cpp")
set(BUILTIN_SRCS ${BUILTIN_SRCS}
        builtins.h
        defs.h
        )
add_library(PylangLib SHARED ${BUILTIN_SRCS})
set_target_properties(PylangLib PROPERTIES
        LINKER_LANGUAGE C
        )
target_link_libraries(PylangLib
        PRIVATE
        fmt::fmt
        )