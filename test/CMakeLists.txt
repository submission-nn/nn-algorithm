#TESTS Flags
set(LINK_TEST_FLAGS -lgtest -pthread m)
set(COMPILE_TEST_FLAGS -g -pthread)

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(${file} ${testfile})
    target_compile_options(${file} PUBLIC ${COMPILE_TEST_FLAGS})
    target_link_libraries(${file} ${LINK_TEST_FLAGS})
endforeach(testfile ${TEST_SOURCES})

add_subdirectory(container)
add_subdirectory(nn)
add_subdirectory(distribution)