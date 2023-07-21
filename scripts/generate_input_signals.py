#! /usr/bin/env python3

import csv
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from scipy import linalg


## INPUT SIGNAL PARAMETERS ##
STEP, RAMP, CHRP, FLIT, NOIS, MIXD, FLITREAL = range(7)

# General
DATA_LENGTH = 1000   # [s]
CTRL_FREQ = 200     # [Hz]

RPM_MIN = 1500      # [RPM]
RPM_MAX = 8500      # [RPM]
RPM_START = RPM_MIN + (RPM_MAX-RPM_MIN)/2   # [RPM]

# Step input
STEP_FREQ = 1       # [Hz]
STEP_VAR = 500      # [RMP]

# Ramp input
RAMP_FREQ = 1       # [Hz]

# Chirp input 
CHRP_STEP_FREQ = 0.5
CHRP_AMPL = 20                # [Rad]
CHRP_FREQ1 = 0.05 *2*math.pi   # [Rad/s]
CHRP_FREQ2 =   15 *2*math.pi   # [Rad/s]
CHRP_PERIOD = 1                # [s]

# White noise input params
NOIS_STEP_FREQ = 0.5
NOIS_VARIANCE = 200  # [s]

# Flight data input
FLIT_FILE = "/../data/flight_data/aggressive.csv"

# Mixed input containing all types of data input
MIXD_INTERVAL = 5  # [s]
MIXD_MIX = [RAMP,CHRP,FLIT]



## FUNCTIONS ##
def chirp_signal(time):
    time = time % CHRP_PERIOD
    return CHRP_AMPL*math.cos(CHRP_FREQ1*time+(CHRP_FREQ2-CHRP_FREQ1)*time*time/(2*CHRP_PERIOD))

def limit_signal(sig):
    while(np.amax(sig)>RPM_MAX or np.amin(sig)<RPM_MIN):
        sig = np.where(sig > RPM_MAX, 2*RPM_MAX-sig, sig)
        sig = np.where(sig < RPM_MIN, 2*RPM_MIN-sig, sig)
    return sig
    

### SCRIPT ###
def main():
    # Get global path of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.DataFrame()

    # Compute random input
    rand_inputs = np.random.normal(0, STEP_VAR, DATA_LENGTH*RAMP_FREQ)
    rand_inputs[0] = RPM_START
    A = linalg.toeplitz( np.ones(rand_inputs.size), np.insert(np.zeros(rand_inputs.size-1),0,1) )
    rand_inputs = np.matmul(A,rand_inputs)
    rand_inputs = limit_signal(rand_inputs)


    # Compute random step input
    step_inputs = np.repeat(rand_inputs, CTRL_FREQ/STEP_FREQ)
    df["step"] = pd.DataFrame(step_inputs)

    # Compute random ramp input
    ramp_inputs = np.interp(np.arange(DATA_LENGTH*CTRL_FREQ),np.arange(DATA_LENGTH*RAMP_FREQ)*CTRL_FREQ/RAMP_FREQ,rand_inputs)
    df["ramp"] = pd.DataFrame(ramp_inputs)

    # Compute chirp signal input
    chrp_inputs = ramp_inputs + np.array([chirp_signal(t) for t in np.arange(0,DATA_LENGTH,1/CTRL_FREQ)])
    chrp_inputs = limit_signal(chrp_inputs)
    df["chrp"] = pd.DataFrame(chrp_inputs)

    # Parse flight data setpoints
    flit_data_path = dir_path + FLIT_FILE
    flit_df = pd.read_csv(flit_data_path)
    flit_inputs = flit_df["setpoint[rad]"].to_numpy()
    n_repeat = math.floor(DATA_LENGTH*CTRL_FREQ / flit_inputs.size ) + 1
    flit_inputs = np.tile(flit_inputs,n_repeat)
    flit_inputs = flit_inputs[:DATA_LENGTH*CTRL_FREQ] 
    flit_inputs_adapt = flit_inputs - np.average(flit_inputs) + ramp_inputs
    flit_inputs_adapt = limit_signal(flit_inputs_adapt)
    df["flit"] = pd.DataFrame(flit_inputs_adapt)

    # Compute white noise input type
    nois_inputs = ramp_inputs + np.random.rand(DATA_LENGTH*CTRL_FREQ) * 2*NOIS_VARIANCE - NOIS_VARIANCE
    nois_inputs = limit_signal(nois_inputs)
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


    df["flit_real"] = pd.DataFrame(flit_inputs)


    print("Computed inputs:")
    print(df)
    df.plot()
    plt.hlines(RPM_MIN,0,DATA_LENGTH*CTRL_FREQ)
    plt.hlines(RPM_MAX,0,DATA_LENGTH*CTRL_FREQ)
    plt.show()

    # Write dataframe to csv file
    df.to_csv(dir_path + "/../data/input_signals/signals.csv", index=False)




if __name__ == "__main__":
    main()

