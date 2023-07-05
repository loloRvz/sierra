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
    path = list_of_files[0]
    print("Opening: ",path)

    # Prepare & plot dataset
    dataset = CSVDataset(path)
    dataset.preprocess()
    dataset.plot_data()




if __name__ == "__main__":
    main()