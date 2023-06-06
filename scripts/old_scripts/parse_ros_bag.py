#! /usr/bin/env python3

import rosbag
import os, glob
import pandas as pd
import csv

from omav_hovery_msgs.msg import UAVStatus

### SCRIPT ###
def main():
    # Get script directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Fech all bag files
    list_of_files = glob.glob("/home/lolo/Downloads/bags/*.bag")

    # Loop through all bag files
    for bag_path in list_of_files:
        # Open bag file
        bag = rosbag.Bag(bag_path)
        start_time = bag.get_start_time()
        motor_id = 6

        # Loop through all 6 servomotors
        for motor_id in range(6,12):
            # Name for csv file
            csv_path = dir_path + '/../../data/flight_data/' + os.path.basename(bag_path)[2:10] + "--" + os.path.basename(bag_path)[11:-4] + "_ID" + str(motor_id-6) + "_L0069.csv"
            # Open csv
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header row
                writer.writerow(["time[ms]","setpoint[rad]","position[rad]","velocity[rad/s]","current[mA]","velocity_computed[rad/s]","acceleration_computed[rad/s^2]"])
                for topic, msg, t in bag.read_messages(topics=['/quail/uav_state']):
                    # Write one msg info to csv
                    writer.writerow([( \
                        msg.header.stamp.to_sec()-start_time)*1000, \
                        msg.motors[motor_id].setpoint, \
                        msg.motors[motor_id].position, \
                        'nan', \
                        'nan', \
                        'nan', \
                        'nan'])

    # Close bag
    bag.close()




if __name__ == "__main__":
    main()

