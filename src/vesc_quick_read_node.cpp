#include <ros/ros.h>
#include <ros/package.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>
#include <math.h>

#include "std_msgs/Float32.h"
#include "omav_hovery_interface/ll/polling_thread.h"
#include <omav_hovery_interface/mock/memory_arming_logic.h>
#include <omav_hovery_interface/mock/memory_rcreceiver_adapter.h>
#include <omav_hovery_interface/ll/vescuart_motor_adapter.h>
#include <omav_hovery_interface/omav_base_client.h>
#include <omav_msgs/MotorStatus.h>

#include "motor_specs.h"
#include "experiment_parameters.h"

using namespace experiment_parameters;


// Setpoint topic callback
class VelocitySetter{
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
	VelocitySetter vs;
	ros::Subscriber set_position_sub = nh.subscribe(setpoint_topic_, 1000, &VelocitySetter::setPointCallback, &vs);
	ros::Rate rate(CTRL_FREQ);

	// Declare motors & init
	omV::ll::VESCUartMotorAdapter adapter("/dev/ttyACM0");
	omV::OMAVBaseClient<_VEL> ms_client_;
	std::shared_ptr<omV::ll::VESCUartMotorAdapter> esc_adapter_;
	esc_adapter_ = std::make_shared<omV::ll::VESCUartMotorAdapter>(adapter);
	esc_adapter_->open();

	// auto dummy_rc = std::make_shared<omV::mock::MemoryRCReceiverAdapter>();
	// auto arming_logic = std::make_shared<omV::mock::MemoryArmingLogic>();
	// arming_logic->arm(); 
	// ms_client_.setInterface(esc_adapter_, dummy_rc, arming_logic);
	// ms_client_.start();

	omV::ll::MotorInterface<omV::ll::MotorType::VEL>::MotorStatusArray motor_read;

	// Create filename for exprmt data
	time_t curr_time; 
	tm * curr_tm;
	char file_str[100], time_str[100], data_str[100];
	strcpy(file_str, ros::package::getPath("sierra").c_str());
	strcat(file_str, "/data/measurements/"); //Global path
	time(&curr_time);
	curr_tm = localtime(&curr_time);
	strftime(time_str, 100, "%y-%m-%d--%H-%M-%S_", curr_tm);
	strcat(file_str,time_str); // Add date & time
	strcat(file_str,input_types_strings[INPUT_TYPE]); //Add input type
	strcat(file_str,".csv");

	// Open file to write data
	std::cout << "Opening: " << file_str << std::endl;
	std::ofstream myfile;
	myfile.open(file_str);
	myfile << "time[s],"
			  "setpoint[kRPM],"
			  "velocity[kRPM]\n"; // Set column descriptions

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
		esc_adapter_->write({vs.getSetPoint()});
		//esc_adapter_->write({2000});
		//ms_client_.setFullState({vs.getSetPoint()});
		motor_read = esc_adapter_->read();

		// Write data to csv file
		sprintf(data_str, "%10.6f,%07.2f,%07ld\n",
			duration_cast<microseconds>(t_now - t_start).count()/1e6,
			vs.getSetPoint() / 1000, 
			motor_read[0].stamp/7 / 1000);
		myfile << data_str;

		// Loop
		ros::spinOnce();
		rate.sleep();
	}
	ROS_INFO("Stopped polling motor. Exiting...");

	esc_adapter_->disable();
	myfile.close();
	return 0;
}
