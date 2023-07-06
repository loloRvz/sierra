#ifndef SIERRA_input_signal_params_h
#define SIERRA_input_signal_params_h

#include <string>

namespace experiment_parameters {
    
    //Default values 
    int CTRL_FREQ = 200;

    int SMPL_FREQ = 400;
    
    enum INPUT_TYPES {STEP,RAMP,CHRP,FLIT,NOIS,MIXD,FLITREAL, N_INPUT_TYPES};
    const char* input_types_strings [] = {"step","ramp","chrp","flit","nois","mixd","flitreal"};
    int INPUT_TYPE = STEP; 

    const std::string setpoint_topic_ = "set_position";

    /*** Load input signal parameters ***/
    void load_params(ros::NodeHandle nh){
        nh.getParam("control_frequency",   CTRL_FREQ);
        nh.getParam("sample_frequency",    SMPL_FREQ);
        nh.getParam("input_type",          INPUT_TYPE);
    }


}
#endif