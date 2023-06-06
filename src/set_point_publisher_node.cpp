#include <ros/ros.h>
#include <ros/package.h>
#include <random>
#include <sstream>
#include <fstream>
#include <vector> 

#include "std_msgs/Float32.h"

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
	ros::init(argc, argv, "set_point_publisher_node");
	ros::NodeHandle nh;
	ros::Publisher set_point_pub = nh.advertise<std_msgs::Float32>("/set_position", 1000);
	std_msgs::Float32 msg;

	// Get experiment parameters
	load_params(nh);

	// Get input signals from csv files
	std::string pkg_path = ros::package::getPath("sierra");
	std::string csv_path = pkg_path + "/data/input_signals/signals.csv";
	std::vector<std::vector<float>> input_signals = parse_csv(csv_path);
	int input_idx = 0;

	// Check for incorrect input type
	if(INPUT_TYPE > N_INPUT_TYPES){
		ROS_ERROR("Input type not valid. Exiting...");
		return 0;
	}

	ros::Rate rate(CTRL_FREQ);
	ROS_INFO("Publishing input...");
	while(ros::ok()){

		// Get input value
		msg.data = input_signals[input_idx][INPUT_TYPE];
		input_idx++;
		if (input_idx >= (int) input_signals.size()) {
			input_idx = 0;
		}

		// Publish set point
		set_point_pub.publish(msg);

		ros::spinOnce();
		rate.sleep();
	}

	ROS_INFO("Stopped publishing input. Exiting...");

	return 0;
}
