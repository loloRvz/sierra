#! /usr/bin/env python3


from sierra_mlp import *


### SCRIPT ###
def main():
    # Open measured data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # list_of_files = glob.glob(dir_path + '/../data/experiments/gazebo/*.csv')
    list_of_files = glob.glob(dir_path + '/../data/training/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()
    path = list_of_files[1]
    #path = '../data/flight_data/23-02-08--15-54-58_ID2.csv'
    print("Opening: ",path)

    # Prepare & plot dataset
    dataset = CSVDataset(path)
    dataset.plot_data()


if __name__ == "__main__":
    main()