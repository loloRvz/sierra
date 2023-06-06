#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

### SCRIPT ###
def main():
    # Open measured data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../../data/input_signals/*.csv')
    list_of_files = sorted(list_of_files)
    path = list_of_files[0]
    print("Opening: ",path)

    # Prepare & plot dataset
    df = pd.read_csv(path, dtype=np.float64)
    df.plot()
    plt.show()

if __name__ == "__main__":
    main()