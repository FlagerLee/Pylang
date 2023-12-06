file(GLOB TEST_SRCS "*.cpp")

foreach(src ${TEST_SRCS})
    string(REGEX MATCH "[A-Za-z0-9_\\-]+\\.cpp" filename ${src})
    string(REPLACE ".cpp" "" test_name ${filename})
    add_executable(${test_name} ${src} builtin.h)
    target_link_libraries(${test_name}
            PRIVATE
            gtest
            gtest_main
            PylangLib
            )
    add_test(BuiltinTests ${test_name})
endforeach()