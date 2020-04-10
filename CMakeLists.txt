project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_subdirectory(src)

file(COPY ${CMAKE_CURRENT_BINARY_DIR}/src/particle_filter
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
