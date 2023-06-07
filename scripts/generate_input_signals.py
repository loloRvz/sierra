#! /usr/bin/env python3

import csv
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


## INPUT SIGNAL PARAMETERS ##
STEP, RAMP, CHRP, FLIT, NOIS, MIXD = range(6)

# General
DATA_LENGTH = 1024   # [s]
CTRL_FREQ = 200     # [Hz]

# Step input
STEP_FREQ = 5       # [Hz]
STEP_MIN = 1500     # [RMP]
STEP_MAX = 9000     # [RMP]

# Ramp input
RAMP_FREQ = 5       # [Hz]
RAMP_MIN = 1500     # [RMP]
RAMP_MAX = 9000     # [RMP]

# Chirp input 
CHRP_STEP_FREQ = 0.5
CHRP_AMPL = 500               # [Rad]
CHRP_FREQ1 = 0.05 *2*math.pi   # [Rad/s]
CHRP_FREQ2 =   15 *2*math.pi   # [Rad/s]
CHRP_PERIOD = 1                # [s]

# White noise input params
NOIS_STEP_FREQ = 0.5
NOIS_VARIANCE = 200  # [s]

# Flight data input
FLIT_FILE = "/../data/flight_data/aggressive.csv"

# Mixed input containing all types of data input
MIXD_INTERVAL = 3  # [s]
MIXD_MIX = [STEP,RAMP,CHRP,FLIT,NOIS]



## FUNCTIONS ##
def chirp_signal(time):
    time = time % CHRP_PERIOD
    return CHRP_AMPL*math.cos(CHRP_FREQ1*time+(CHRP_FREQ2-CHRP_FREQ1)*time*time/(2*CHRP_PERIOD))


### SCRIPT ###
def main():
    # Get global path of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.DataFrame()

    # Compute random step input
    step_inputs = np.random.uniform(low=STEP_MIN,high=STEP_MAX, size=(DATA_LENGTH*STEP_FREQ,))
    step_inputs = np.repeat(step_inputs, CTRL_FREQ/STEP_FREQ)
    df["step"] = pd.DataFrame(step_inputs)

    # Compute random ramp input
    ramp_inputs = np.random.uniform(low=RAMP_MIN,high=RAMP_MAX, size=(DATA_LENGTH*STEP_FREQ,))
    ramp_inputs = np.interp(np.arange(DATA_LENGTH*CTRL_FREQ),np.arange(DATA_LENGTH*RAMP_FREQ)*CTRL_FREQ/RAMP_FREQ,ramp_inputs)
    df["ramp"] = pd.DataFrame(ramp_inputs)

    # Compute chirp signal input
    chrp_inputs = np.random.uniform(low=STEP_MIN,high=STEP_MAX, size=int(DATA_LENGTH*CHRP_STEP_FREQ,))
    chrp_inputs = np.repeat(chrp_inputs, CTRL_FREQ/CHRP_STEP_FREQ)
    chrp_inputs = chrp_inputs + np.array([chirp_signal(t) for t in np.arange(0,DATA_LENGTH,1/CTRL_FREQ)])
    df["chrp"] = pd.DataFrame(chrp_inputs)

    # Parse flight data setpoints
    flit_data_path = dir_path + FLIT_FILE
    flit_df = pd.read_csv(flit_data_path)
    flit_inputs = flit_df["setpoint[rad]"].to_numpy()
    n_repeat = math.floor(DATA_LENGTH*CTRL_FREQ / flit_inputs.size ) + 1
    flit_inputs = np.tile(flit_inputs,n_repeat)
    flit_inputs = flit_inputs[:DATA_LENGTH*CTRL_FREQ] 
    df["flit"] = pd.DataFrame(flit_inputs)

    # Compute white noise input type
    nois_inputs = np.random.uniform(low=STEP_MIN,high=STEP_MAX, size=int(DATA_LENGTH*NOIS_STEP_FREQ,))
    nois_inputs = np.repeat(nois_inputs, CTRL_FREQ/NOIS_STEP_FREQ)
    nois_inputs = nois_inputs + np.random.rand(DATA_LENGTH*CTRL_FREQ) * 2*NOIS_VARIANCE - NOIS_VARIANCE
    df["nois"] = pd.DataFrame(nois_inputs)


    # Compute mixed types input
    inputs_array = df.to_numpy()
    mixd_inputs = np.zeros(inputs_array.shape[0])
    seg_size = MIXD_INTERVAL*CTRL_FREQ

    # Choose which type of input signals to include in mix
    mix = MIXD_MIX
    m = len(mix)

    arr = np.zeros((seg_size,m))
    n = 0
    while n < DATA_LENGTH*CTRL_FREQ:
        # This works for some reason, don't touch
        for i in range(m):
            if n+i*seg_size < DATA_LENGTH*CTRL_FREQ:
                arr[:,i] = np.resize( inputs_array[n+i*seg_size:n+(i+1)*seg_size,mix[i]] , arr[:,i].shape )
        mixd_inputs[n:n+m*seg_size] = np.resize( np.transpose(arr).reshape((1,-1)), mixd_inputs[n:n+m*seg_size].shape )
        n = n + m*seg_size
    df["mixd"] = pd.DataFrame(mixd_inputs)



    print("Computed inputs:")
    print(df)
    df.plot()
    plt.show()

    # Write dataframe to csv file
    df.to_csv(dir_path + "/../data/input_signals/signals.csv", index=False)




if __name__ == "__main__":
    main()

