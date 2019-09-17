
set(SRC
test.cpp
getname.cpp
)

find_package( OpenCV REQUIRED )
add_executable(test_exec ${SRC})
target_link_libraries( test_exec ${OpenCV_LIBS} )
