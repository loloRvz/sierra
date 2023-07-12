#! /usr/bin/env python3


from sierra_mlp import *

import math

### SCRIPT ###
def main():
    # Get all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/evaluation/0-step/*.csv')
    list_of_files = sorted(list_of_files)
    #list_of_files.reverse()
    
    # Load data from files
    signal_names = []
    times = []
    setpoints = []
    velocities = []
    for path in list_of_files:
        print("Opening: ",path)

        # Prepare & plot dataset
        dataset = CSVDataset(path)
        dataset.preprocess()
        data = dataset.df.to_numpy()

        # Put values in arrays
        signal_names.append(path[-6:-4])
        times.append(data[:,TIME])
        setpoints.append(data[:,SETPOINT])
        velocities.append(data[:,VELOCITY])


    # Sync measurements times
    min_time = 2
    max_time = min([times_arr[-1] for times_arr in times]) - 2
    velocities = [velocities[i][np.logical_and(min_time < times[i],times[i] < max_time)]  for i in range(len(velocities))]
    setpoints = [setpoints[i][np.logical_and(min_time < times[i],times[i] < max_time)]  for i in range(len(velocities))]
    times = [times[i][np.logical_and(min_time < times[i],times[i] < max_time)]  for i in range(len(velocities))]

    # Interpolate signals to sync times
    velocities_interp = [np.interp(times[0], times[i], velocities[i]) for i in range(len(velocities))]
    setpoints_interp = [np.interp(times[0], times[i], setpoints[i]) for i in range(len(setpoints))]
    errors_interp = [(np.square(velocities_interp[0]-velocities_interp[i])) for i in range(len(velocities))]

    # Compute errors to measurements
    rmse = [math.sqrt(mean_squared_error(velocities_interp[0], p)) for p in velocities_interp]
    rmse_set = [math.sqrt(mean_squared_error(setpoints_interp[0], s)) for s in setpoints_interp]

    # Print RMSE of all signals
    print("RMSE")
    for i in range(len(velocities)):
        print("Signal: ",signal_names[i],", RMSE: ",rmse[i])

    # Plot signals
    plt.figure(1,figsize=(7,5))
    plt.plot(times[0],setpoints[0],"--")
    for i in range(len(velocities)):
        plt.plot(times[0],velocities_interp[i])
        
    leg = ["Setpoint"]
    leg.extend(signal_names)
    #leg.insert(0,"Setpoint")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [RPM]")
    plt.legend(leg)                         
    plt.title("Velocity control comparison")

    plt.figure(2,figsize=(7,5))
    plt.bar(signal_names[1:], rmse[1:],width = 0.5)
    plt.xlabel("History Length")
    plt.ylabel("RMSE")
    plt.title("Velocity control comparison")
    
    plt.show()




if __name__ == "__main__":
    main()

