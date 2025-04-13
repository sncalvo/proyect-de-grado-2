cmake_minimum_required(VERSION 3.10)
project(Glad)

# Add library
add_library(glad 
    src/glad.c
)

# Include directories
target_include_directories(glad 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set properties
set_target_properties(glad PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

find_package(OpenGL REQUIRED)

# If using modern CMake (3.13+), also add installation rules
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.13")
    install(TARGETS glad
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(DIRECTORY include/
        DESTINATION include
    )
endif() 