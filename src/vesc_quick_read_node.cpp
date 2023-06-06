#include <ros/ros.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>
#include <math.h>

#include "std_msgs/Float32.h"
#include "omav_hovery_interface/ll/dynamixel_motor_adapter.h"
#include "omav_hovery_interface/ll/polling_thread.h"

#include "motor_specs.h"
#include "experiment_parameters.h"

using namespace experiment_parameters;


// Setpoint topic callback
class PositionSetter{
	public:
		void setPointCallback(std_msgs::Float32 msg) {
			set_point = msg.data;
		}
		double getSetPoint() {
			return set_point;
		}

	private:
		double set_point = 0;
};


int main(int argc, char ** argv) {

	// Init rosnode and subscribe to setpoint topic with callback to position setter
	ros::init(argc, argv, "dxl_quick_read_node");
	ros::NodeHandle nh;
	load_params(nh); // Get experiment parameters
	PositionSetter ps;
	ros::Subscriber set_position_sub = nh.subscribe(setpoint_topic_, 1000, &PositionSetter::setPointCallback, &ps);
	ros::Rate rate(SMPL_FREQ);

	

	// Declare motors & interfaces
	std::array<int, 1> dynamixels = {DXL1_ID};
	std::array<double, 1> setPointAngle{M_PI};
	std::array<double, 1> offsets{-1.636};
	omV::ll::MotorInterface<_POS>::MotorStatusArray readBackStatus;
	omV::ll::DynamixelMotorAdapter<_POS> ta_adapter(DEVICE_NAME, BAUDRATE, dynamixels, offsets);

	// Init motor
	ta_adapter.open();
	ta_adapter.enable();

	// Create filename for exprmt data
	time_t curr_time; 
	tm * curr_tm;
	char file_str[100], time_str[100], data_str[100];
	strcpy(file_str, "/home/lolo/omav_ws/src/sierra/data/measurements_quail/"); //Global path
	time(&curr_time);
	curr_tm = localtime(&curr_time);
	strftime(time_str, 100, "%y-%m-%d--%H-%M-%S_", curr_tm);
	strcat(file_str,time_str);  // Add date & time
	//Add input type
	strcat(file_str,input_types_strings[INPUT_TYPE]);
	strcat(file_str,".csv");

	// Open file to write data
	std::cout << "Opening: " << file_str << std::endl;
	std::ofstream myfile;
	myfile.open(file_str);
	myfile << "time[s],"
			  "setpoint[rad],"
			  "position[rad],"
			  "current[mA],"
			  "velocity_computed[rad/s],"
			  "acceleration_computed[rad/s^2]\n"; // Set column descriptions

	// Wait for first setpoint topic to be published
	ros::topic::waitForMessage<std_msgs::Float32>(setpoint_topic_,ros::Duration(5));

	// Time variables
	time_point t_now = std::chrono::system_clock::now();
	time_point t_start = std::chrono::system_clock::now();

	ROS_INFO("Polling motor...");
	while (ros::ok()) {
		// Measure exact loop time
		t_now = std::chrono::system_clock::now();

		// Read motor data & write setpoint
		setPointAngle[0] = ps.getSetPoint();
		ta_adapter.write(setPointAngle);
		readBackStatus = ta_adapter.read();

		// Write data to csv file
		sprintf(data_str, "%10.6f,%07.5f,%06.3f,%08.2f,%03.3f,%03.3f\n",
			duration_cast<microseconds>(t_now - t_start).count()/1e6,
			readBackStatus[0].setpoint - offsets[0], 
			readBackStatus[0].position,
			readBackStatus[0].current,
			NAN,
			NAN);
		myfile << data_str;

		// Loop
		ros::spinOnce();
		rate.sleep();
	}
	ROS_INFO("Stopped polling motor. Exiting...");

	ta_adapter.disable();
	myfile.close();
	return 0;
}
