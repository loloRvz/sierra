cmake_minimum_required(VERSION 2.8.3)
project(sierra)

find_package(catkin_simple REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)
add_compile_definitions(MODE_ROSLOG)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++17  -Wno-reorder -Wno-unknown-pragmas -Werror)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  omav_hovery_interface
  mav_msgs
  roslib
)

add_definitions(-DMODE_ROSLOG)


include_directories(
  include/sierra
  ${catkin_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
)

add_definitions(-std=c++17 -fPIC -Wall -Wno-reorder -Wno-unknown-pragmas) 

catkin_python_setup()
catkin_install_python(PROGRAMS
  setup.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



cs_add_executable(vesc_quick_read_node src/vesc_quick_read_node.cpp)
#add_dependencies(dxl_quick_read_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vesc_quick_read_node ${catkin_LIBRARIES})

cs_add_executable(set_point_publisher_node src/set_point_publisher_node.cpp)
#add_dependencies(set_point_publisher_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(set_point_publisher_node ${catkin_LIBRARIES})

cs_add_executable(gazebo_read_node src/gazebo_read_node.cpp)
#add_dependencies(gazebo_read_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_read_node ${catkin_LIBRARIES})

cs_add_executable(gazebo_write_node src/gazebo_write_node.cpp)
#add_dependencies(gazebo_write_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_write_node ${catkin_LIBRARIES})

cs_install()
cs_export()