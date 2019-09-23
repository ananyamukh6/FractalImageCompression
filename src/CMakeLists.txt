
#set(SRC
#test.cpp
#getname.cpp
#)

set(SRC
compression.cpp
)

find_package( OpenCV REQUIRED )
add_executable(test_exec ${SRC})
target_link_libraries( test_exec ${OpenCV_LIBS} )

file(COPY lena1.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
file(COPY lena2.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
