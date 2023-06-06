# Sierra
State-Inferred Electric Rotor Rate Approximation

## Description
This repo comprises work related to my Master Thesis, in which I aim to learn the model dynamics of a Dynamixel XM430-W210 servo-motor using supervised learning. 

It contains:
- Tools for gathering experimental data
- Scripts for preprocessing and training the model

## Filesystem
A brief description of the relevant directories

/config/:
- Experiment setup parameters

/data:
- /experiments/: Experimental results
- /flight_data/: Position and setpoint data from real
- /input_signals/: Generated input signals of various types 
- /models/: Trained networks

/include/, /launch/ & /src/:
- C++ files to run experiments in ROS

/scripts/:
- Python scripts for training purposes

## Instructions
Clone this repo into you catkin workspace's src/ directory

### Data gathering
The 'omav_hovery' drivers are used to interface the dynamixel motor. Clone the repo into your catkin workspace using
~~~
git clone --recurse-submodules https://github.com/ethz-asl/omav_hovery
~~~
and run all its necessary install scripts.


Then, build the ros nodes with 
~~~
catkin build sierra
~~~
After connecting and turn on the motor controller, set the experiment's input signal and load_id in the config file and launch the exeperiment:
~~~
roslaunch sierra data_collection.launch
~~~

This program generates csv files with the captured motor data. The filenames consist of the date and time, the input signal type, and the used load. 

### Model training 
The python scripts for training a network using experimental data are located in the /scripts/ directory.

*sierra_mlp.py*:
- Read the csv dataset
- Compute velocity & acceleration from positions
- Prepare network data (inputs & target) with a given position history length
- Train model
- Evaluate model
