#! /usr/bin/env python3


from siesta_mlp import *


### SCRIPT ###
def main():
    # Open measured data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/experiments/training/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()

    for path in list_of_files:
        print("Opening: ",path)
        dataset = CSVDataset(path)
        dataset.preprocess(resave=True)


if __name__ == "__main__":
    main()