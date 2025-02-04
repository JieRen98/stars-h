# @copyright (c) 2017 King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# STARS-H is a software package, provided by King Abdullah
#             University of Science and Technology (KAUST)
#
# @file src/backends/openmp/CMakeLists.txt
# @version 0.3.0
# @author Aleksandr Mikhalev
# @date 2017-11-07


# Collect sources for documentation and compilation
set(SRC)
add_subdirectory("blrm")

# Compilation of OpenMP is always required
add_library(backends_openmp OBJECT ${SRC})
target_compile_options(backends_openmp PUBLIC "${OpenMP_C_FLAGS}")
target_include_directories(backends_openmp PRIVATE ../../../third_party/ ../../../third_party/plasma/quark ../../../third_party/plasma/include)
target_link_libraries(backends_openmp PRIVATE Eigen3::Eigen)
# Put doxygen input to parent scope
set(DOXYGEN_INPUT ${DOXYGEN_INPUT} ${SRC} PARENT_SCOPE)
