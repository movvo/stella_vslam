# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/Allocator.cu
               ${CMAKE_CURRENT_SOURCE_DIR}/Allocator.hpp
               ${CMAKE_CURRENT_SOURCE_DIR}/Cuda.cu
               ${CMAKE_CURRENT_SOURCE_DIR}/Cuda.hpp
               ${CMAKE_CURRENT_SOURCE_DIR}/Fast.cu
               ${CMAKE_CURRENT_SOURCE_DIR}/Fast.hpp
               ${CMAKE_CURRENT_SOURCE_DIR}/Orb.cu
               ${CMAKE_CURRENT_SOURCE_DIR}/Orb.hpp)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
install(FILES ${HEADERS}
        DESTINATION ${STELLA_VSLAM_INCLUDE_INSTALL_DIR}/feature/cuda)
