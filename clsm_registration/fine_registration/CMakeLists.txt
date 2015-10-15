PROJECT( MultiResMIRegistration )


#IF( BUILD_OUTSIDE_INSIGHT_APPLICATIONS )

  FIND_PACKAGE(ITK)
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
    MESSAGE(${ITK_USE_FILE})
  ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR
            "Cannot build ITKApps without ITK.  Please set ITK_DIR.")
  ENDIF(ITK_FOUND)

  FIND_PACKAGE(FLTK)
  IF(FLTK_FOUND)
    INCLUDE_DIRECTORIES(${FLTK_INCLUDE_DIR})
  ENDIF(FLTK_FOUND)

#ENDIF( BUILD_OUTSIDE_INSIGHT_APPLICATIONS )



set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_INCLUDE_DIR /usr/include/)
set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/) 
find_package(Boost 1.46.0 COMPONENTS system filesystem REQUIRED) 

IF (Boost_FOUND)
  MESSAGE("FOUND BOOST!!!")
ENDIF()

INCLUDE_REGULAR_EXPRESSION(".*")

INCLUDE_DIRECTORIES(
${MultiResMIRegistration_SOURCE_DIR}/Common
${MultiResMIRegistration_SOURCE_DIR}/../IBSRValidation/Common
${Boost_INCLUDE_DIRS}
)


link_directories(${Boost_LIBRARY_DIR})

MESSAGE("CMAKE_BINARY_DIR: ${Boost_LIBRARIES}")

#ADD_GUI_EXECUTABLE(MultiResMIRegistration Common/MultiResMIRegistration.cxx )
add_executable(MultiResMIRegistration Common/MultiResMIRegistration.cxx  Common/itkQuaternionRigidTransformGradientDescentOptimizerStoppingCriterion.cxx Common/itkMattesNormalizedMutualInformationImageToImageMetric.hxx  )
TARGET_LINK_LIBRARIES(MultiResMIRegistration ${ITK_LIBRARIES} ${Boost_LIBRARIES})
