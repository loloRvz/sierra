#include <ros/ros.h>
#include <ros/package.h>
#include <random>
#include <sstream>
#include <fstream>
#include <vector> 

#include "mav_msgs/Actuators.h"

#include "experiment_parameters.h"

using namespace experiment_parameters;

// Parse CSV function
std::vector<std::vector<float>> parse_csv(std::string csv_path) {
	std::ifstream  data(csv_path);
	std::string line;
	std::vector<std::vector<float>> parsedCsv;

	// Discard first line (column names)
	std::getline(data,line);

	// Read data line per line
	while(std::getline(data,line)){
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<float> parsedRow;
		while(std::getline(lineStream,cell,',')){
			parsedRow.push_back(std::stof(cell));
		}
		parsedCsv.push_back(parsedRow);
	}
	return parsedCsv;  
}

/*** MAIN ***/
int main(int argc, char ** argv) {
	//Init rosnode and setpoint publisher
	ros::init(argc, argv, "gazebo_write_node");
	ros::NodeHandle nh;
	ros::Publisher set_point_pub = nh.advertise<mav_msgs::Actuators>("/stork/command/motor_speed", 1000);

	// Init msg
	mav_msgs::Actuators msg;
	msg.angles.clear();
	msg.angular_velocities.clear();
	msg.normalized.clear();
	for(int i=0; i<6; i++){
		msg.angles.push_back(NAN);
		msg.angular_velocities.push_back(0);
		msg.normalized.push_back(NAN);
	}for(int i=6; i<12; i++){
		msg.angles.push_back(0);
		msg.angular_velocities.push_back(NAN);
		msg.normalized.push_back(NAN);
	}

	// Get experiment parameters
	load_params(nh);

	// Get input signals from csv files
	std::string pkg_path = ros::package::getPath("siesta");
	std::string csv_path = pkg_path + "/data/input_signals/signals.csv";
	std::vector<std::vector<float>> input_signals = parse_csv(csv_path);
	int input_idx = 0;

	// Check for incorrect input type
	if(INPUT_TYPE > N_INPUT_TYPES){
		ROS_ERROR("Input type not valid. Exiting...");
		return 0;
	}

	// Wait for gazebo startup
	ros::topic::waitForMessage<mav_msgs::Actuators>("/stork/gazebo/motor_states",ros::Duration(10));

	ros::Rate rate(CTRL_FREQ);
	ROS_INFO("Publishing input...");
	while(ros::ok()){
		// Update header
		msg.header.stamp = ros::Time::now();

		// Update msg setpoint
		msg.angles[6] = input_signals[input_idx][INPUT_TYPE];

		// Iterate or loop around
		input_idx++;
		if (input_idx >= (int) input_signals.size()) {
			input_idx = 0;
		}

		// Publish set point
		set_point_pub.publish(msg);

		// Ros stuff
		ros::spinOnce();
		rate.sleep();
	}

	ROS_INFO("Stopped publishing input. Exiting...");

	return 0;
}
