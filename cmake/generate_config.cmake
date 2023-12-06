function(generate_config)
    # pointer bit width
    cmake_host_system_information(RESULT temp QUERY IS_64BIT)
    if(temp)
        set(PTR_BIT_WIDTH 8)
    else()
        set(PTR_BIT_WIDTH 4)
    endif()

    # have le64toh function
    CHECK_CXX_SOURCE_COMPILES (
            "#include<endian.h> int main() { le64toh(1); }"
            HAVE_LE64TOH
    )
endfunction()
