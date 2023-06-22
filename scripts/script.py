#! /usr/bin/env python3

from sierra_mlp import *



### SCRIPT ###
def main():

    # Model parameters
    h_len = 7

    # Open training dataset
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/training/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()
    path = list_of_files[0]
    print("Opening: ",path)

    # Prepare dataset
    dataset = CSVDataset(path)
    data = dataset.df.to_numpy(dtype=np.float64)

    print("Max setpoint: ",np.max(data[:,SETPOINT]))
    print("Max velocity: ",np.max(data[:,VELOCITY]))



if __name__ == "__main__":
    main()
