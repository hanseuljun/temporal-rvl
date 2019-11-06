add_executable(TemporalRvlMain
  rvl.h
  rvl.cpp
  trvl.h
  trvl.cpp
  main.cpp
)
target_link_libraries(TemporalRvlMain
  ${OpenCV_LIBS}
)
set_target_properties(TemporalRvlMain PROPERTIES
  CXX_STANDARD 17
)