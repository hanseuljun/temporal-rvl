add_executable(Paper
  rvl.h
  trvl.h
  paper.cpp
)
target_link_libraries(Paper
  ${OpenCV_LIBS}
)
set_target_properties(Paper PROPERTIES
  CXX_STANDARD 17
)

add_executable(Example
  rvl.h
  trvl.h
  example.cpp
)
target_link_libraries(Example
  ${OpenCV_LIBS}
)
set_target_properties(Example PROPERTIES
  CXX_STANDARD 17
)