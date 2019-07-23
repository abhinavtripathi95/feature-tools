add_library(sfop SHARED CImageCpu.cpp
                        CLayer.cpp  
                        COctave.cpp  
                        CSfop.cpp 
                        main.cpp  
)
# We do not need the executable, just a library
# which we can call in python
# add_executable(bin_mysfop CImageCpu.cpp
                        # CLayer.cpp  
                        # COctave.cpp  
                        # CSfop.cpp 
                        # main.cpp)

target_link_libraries(sfop -lpthread)
target_link_libraries(sfop ${X11_LIBRARIES})
# target_link_libraries(bin_mysfop mysfop)
