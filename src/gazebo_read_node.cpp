#include <ros/ros.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include "mav_msgs/Actuators.h"

#include "omav_hovery_interface/ll/dynamixel_motor_adapter.h"
#include "omav_hovery_interface/ll/polling_thread.h"

#include "experiment_parameters.h"

#define MOTOR_ID 6

using namespace experiment_parameters;

// Setpoint topic callback
class PositionSetter{
	public:
		void setPointCallback(mav_msgs::Actuators msg) {
			set_point = msg.angles[MOTOR_ID];
		}
		double getSetPoint() {
			return set_point;
		}

	private:
		double set_point = 0;
};

// Position topic callback
class PositionGetter{
	public:
		void getPosCallback(mav_msgs::Actuators msg) {
			position = msg.angles[MOTOR_ID];
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
	PositionSetter ps;
	PositionGetter pg;
	ros::Subscriber set_position_sub = nh.subscribe("/stork/command/motor_speed", 1000, &PositionSetter::setPointCallback, &ps);
	ros::Subscriber get_position_sub = nh.subscribe("/stork/gazebo/motor_states", 1000, &PositionGetter::getPosCallback, &pg);
	ros::Rate rate(SMPL_FREQ);

	// Some variables...
	float set_point_angle;
	float position_angle;

	// Create filename for experiment data
	time_t curr_time; 
	tm * curr_tm;
	char file_str[100], time_str[100], exprmt_descr_str[100], data_str[100];
	strcpy(file_str, "/home/lolo/omav_ws/src/siesta/data/measurements_quail_gazebo/"); //Global path
	time(&curr_time);
	curr_tm = localtime(&curr_time);
	strftime(time_str, 100, "%y-%m-%d--%H-%M-%S_", curr_tm);
	strcat(file_str,time_str);  // Add date & time
	sprintf(exprmt_descr_str, "%dHz-L%d-",SMPL_FREQ,LOAD_ID);
	strcat(file_str,exprmt_descr_str); //Add load id
	//Add input type
	strcat(file_str,input_types_strings[INPUT_TYPE]);
	strcat(file_str,".csv");

	// Open file to write data
	std::ofstream myfile;
	myfile.open(file_str);
	myfile << "time[s],"
			  "setpoint[rad],"
			  "position[rad],"
			  "current[mA],"
			  "velocity_computed[rad/s],"
			  "acceleration_computed[rad/s^2] \n"; // Set column descriptions

	// Wait for first setpoint topic to be published
	ros::topic::waitForMessage<mav_msgs::Actuators>("/stork/command/motor_speed",ros::Duration(5));

	// Time variables
	ros::Time t_start = ros::Time::now();
	ros::Time t_now = ros::Time::now();

	ROS_INFO("Polling motor...");
	while (ros::ok()) {
		// Measure exact loop time
		t_now = ros::Time::now();

		// Read motor data & write setpoint
		set_point_angle = ps.getSetPoint() + M_PI;
		position_angle = pg.getPos() + M_PI;

		// Write data to csv file
		sprintf(data_str, "%10.6f,%07.5f,%07.5f,%03.3f,%03.3f,%03.3f\n",
			(t_now - t_start).toSec(),
			set_point_angle - M_PI, 
			position_angle - M_PI, 
			NAN,
			NAN,
			NAN);
		myfile << data_str;

		// Loop
		ros::spinOnce();
		rate.sleep();
	}
	ROS_INFO("Stopped polling motor. Exiting...");

	myfile.close();
	return 0;
}
