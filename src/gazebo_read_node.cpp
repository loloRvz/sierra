#include <ros/ros.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include "mav_msgs/Actuators.h"

#include "omav_hovery_interface/ll/dynamixel_motor_adapter.h"
#include "omav_hovery_interface/ll/polling_thread.h"

#include "experiment_parameters.h"

#define MOTOR_ID 0

using namespace experiment_parameters;

// Setpoint topic callback
class VelocitySetter{
	public:
		void setPointCallback(mav_msgs::Actuators msg) {
			set_point = msg.angular_velocities[MOTOR_ID];
		}
		double getSetPoint() {
			return set_point;
		}

	private:
		double set_point = 0;
};

// Position topic callback
class VelocityGetter{
	public:
		void getPosCallback(mav_msgs::Actuators msg) {
			position = msg.angular_velocities[MOTOR_ID];
		}
		double getPos() {
			return position;
		}

	private:
		double position = 0;
};


int main(int argc, char ** argv) {

	// Init rosnode and subscribe to setpoint topic with callback to position setter
	ros::init(argc, argv, "gazebo_read_node");
	ros::NodeHandle nh;
	load_params(nh); // Get experiment parameters
	VelocitySetter vs;
	VelocityGetter vg;
	ros::Subscriber set_position_sub = nh.subscribe("/stork/command/motor_speed", 1000, &VelocitySetter::setPointCallback, &vs);
	ros::Subscriber get_position_sub = nh.subscribe("/stork/gazebo/motor_states", 1000, &VelocityGetter::getPosCallback, &vg);
	ros::Rate rate(SMPL_FREQ);

	// Some variables...
	float set_point_velocity;
	float actual_veloctiy;

	// Create filename for experiment data
	time_t curr_time; 
	tm * curr_tm;
	char file_str[100], time_str[100], data_str[100];
	strcpy(file_str, "/home/lolo/omav_ws/src/sierra/data/measurements_quail_gazebo/"); //Global path
	time(&curr_time);
	curr_tm = localtime(&curr_time);
	strftime(time_str, 100, "%y-%m-%d--%H-%M-%S_", curr_tm);
	strcat(file_str,time_str);  // Add date & time
	//Add input type
	strcat(file_str,input_types_strings[INPUT_TYPE]);
	strcat(file_str,".csv");

	// Open file to write data
	std::ofstream myfile;
	myfile.open(file_str);
	myfile << "time[s],"
			  "setpoint[RPM],"
			  "velocity[RPM]\n"; // Set column descriptions

	// Wait for first setpoint topic to be published
	ros::topic::waitForMessage<mav_msgs::Actuators>("/stork/command/motor_speed",ros::Duration(10));

	// Time variables
	ros::Time t_start = ros::Time::now();
	ros::Time t_now = ros::Time::now();

	ROS_INFO("Polling motor...");
	while (ros::ok()) {
		// Measure exact loop time
		t_now = ros::Time::now();

		// Read motor data & write setpoint
		set_point_velocity = vs.getSetPoint();
		actual_veloctiy = vg.getPos();

		// Write data to csv file
		sprintf(data_str, "%10.5f,%2.1f,%2.1f\n",
			(t_now - t_start).toSec(),
			set_point_velocity, 
			actual_veloctiy);
		myfile << data_str;

		// Loop
		ros::spinOnce();
		rate.sleep();
	}
	ROS_INFO("Stopped polling motor. Exiting...");

	myfile.close();
	return 0;
}
