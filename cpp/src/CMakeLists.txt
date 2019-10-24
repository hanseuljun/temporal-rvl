add_executable(TemporalRvlMain
  main.cpp
)
target_link_libraries(TemporalRvlMain
  ${OpenCV_LIBS}
)
set_target_properties(TemporalRvlMain PROPERTIES
  CXX_STANDARD 17
)