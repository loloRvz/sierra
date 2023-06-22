#! /usr/bin/env python3


from sierra_mlp import *

import math

### SCRIPT ###
def main():
    # Get all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/evaluation/5-mixd/*.csv')
    list_of_files = sorted(list_of_files)
    #list_of_files.reverse()
    

    # Load data from files
    times = []
    setpoints = []
    velocities = []
    for path in list_of_files:
        print("Opening: ",path)

        # Prepare & plot dataset
        dataset = CSVDataset(path)
        data = dataset.df.to_numpy()

        # Put values in arrays     
        times.append(data[:,TIME])
        setpoints.append(data[:,SETPOINT])
        velocities.append(data[:,VELOCITY])


    # Sync measurements times
    min_time = min([times_arr[-1] for times_arr in times])

    velocities = [velocities[i][min_time > times[i]]  for i in range(len(velocities))]
    setpoints = [setpoints[i][min_time > times[i]]  for i in range(len(velocities))]
    times = [times[i][min_time > times[i]]  for i in range(len(velocities))]

    velocities_interp = [np.interp(times[0], times[i], velocities[i]) for i in range(len(velocities))]
    setpoints_interp = [np.interp(times[0], times[i], setpoints[i]) for i in range(len(setpoints))]

    rmse = [math.sqrt(mean_squared_error(velocities_interp[0], p)) for p in velocities_interp]
    rmse_set = [math.sqrt(mean_squared_error(setpoints_interp[0], s)) for s in setpoints_interp]

    print("RMSE")
    print("Real system ",rmse[0], "(setpoints:",rmse_set[0],")")
    print("Low-Pass Filter",int(rmse[1]), "(setpoints:",int(rmse_set[1]),")")
    print("NN Model",int(rmse[2]), "(setpoints:",int(rmse_set[2]),")")

    signals = ["Setpoint","Real system","Low Pass Filter","NN Model"]

    # Plot signals
    plt.figure(1,figsize=(7,5))
    plt.plot(times[0],setpoints[0],"--")
    for i in range(len(velocities)):
        plt.plot(times[0],velocities_interp[i])
    plt.axhline(y=0, color='k')
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [RPM]")
    plt.legend(signals)                       
    plt.title("Velocity control comparison")

    # plt.figure(2,figsize=(7,5))
    # plt.bar(signals[2:], rmse[1:],width = 0.5)
    # plt.xlabel("Model")
    # plt.ylabel("RMSE")
    # plt.title("Model evaluation")
    
    plt.show()




if __name__ == "__main__":
    main()

