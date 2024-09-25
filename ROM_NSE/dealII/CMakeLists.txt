# Set the name of the project and target:
SET(TARGET "main")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
INCLUDE_DIRECTORIES(include)
SET(TARGET_SRC
  ${TARGET_SRC}
  main.cc
  )

# Usually, you will not need to modify anything beyond this point...
IF(EXISTS /home/ifam/fischer/dealii-candi/deal.II-v9.2.0)
    CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
    set (DEAL_II_DIR  /home/ifam/fischer/dealii-candi/deal.II-v9.2.0)
    set (DEAL_II_PATH  /home/ifam/fischer/dealii-candi/deal.II-v9.2.0)

    FIND_PACKAGE(deal.II 9.2.0 EXACT
        HINTS ${DEAL_II_DIR}
    )
ELSEIF(EXISTS /home/hendrik/dealii-candi/deal.II-v9.3.1)
     CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)
     set (DEAL_II_DIR  /home/hendrik/dealii-candi/deal.II-v9.3.1)
     set (DEAL_II_PATH  /home/hendrik/dealii-candi/deal.II-v9.3.1)

     FIND_PACKAGE(deal.II 9.3.0
         HINTS ${DEAL_II_DIR} /home/hendrik/dealii-candi/
     )
ENDIF()

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


IF( NOT DEAL_II_WITH_HDF5 )
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_HDF5 = ON
HDF5 is OFF in your installation but is required for this program."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})


#set (TORCH_DIR  /home/julian/libtorch_nocuda/libtorch)
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

DEAL_II_INVOKE_AUTOPILOT()

target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 14)
