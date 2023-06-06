#! /usr/bin/env python3


from src.sierra.scripts.sierra_mlp import *

from scipy import signal

### SCRIPT ###
def main():
    # Get all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/evaluation/0-step/*.csv')
    list_of_files = sorted(list_of_files)
    #list_of_files.reverse()
    

    # Load data from files
    times = []
    setpoints = []
    positions = []
    torques = []
    for path in list_of_files:
        print("Opening: ",path)

        # Prepare & plot dataset
        dataset = CSVDataset(path)
        dataset.preprocess()
        data = dataset.df.to_numpy()

        # Put values in arrays     
        times.append(data[:,TIME])
        setpoints.append(data[:,SETPOINT])
        positions.append(data[:,POSITION])
        torques.append(data[:,ACCELERATION_COMP] * LOAD_INERTIAS[dataset.load_id])


    # Sync measurements times
    min_time = min([times_arr[-1] for times_arr in times])

    setpoints = [setpoints[i][min_time > times[i]]  for i in range(len(positions))]
    positions = [positions[i][min_time > times[i]]  for i in range(len(positions))]
    times = [times[i][min_time > times[i]]  for i in range(len(positions))]

    positions_interp = [np.interp(times[0], times[i], positions[i]) for i in range(len(positions))]
    setpoints_interp = [np.interp(times[0], times[i], setpoints[i]) for i in range(len(setpoints))]

    rmse = [mean_squared_error(positions_interp[0], p) for p in positions_interp]
    rmse_set = [mean_squared_error(setpoints[0], s) for s in setpoints_interp]
    print("RMSE")
    print("Real system ",rmse[0], "(",rmse_set[0],")")
    print("PD Model DAM",rmse[1], "(",rmse_set[1],")")
    print("NN Model PEH",rmse[2], "(",rmse_set[2],")")
    print("NN Model PH ",rmse[3], "(",rmse_set[3],")")

    signals = ["Setpoint","Real system","PD Model","NN Model Position Error Hist.","NN Model Position History"]

    # Plot signals
    plt.figure(1,figsize=(7,5))
    plt.plot(times[0],setpoints[0],"--")
    for i in range(len(positions)):
        plt.plot(times[0],positions_interp[i])
    plt.axhline(y=0, color='k')
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude [rad]")
    plt.legend(signals)                       
    plt.title("Position control comparison")

    plt.figure(2,figsize=(7,5))
    plt.bar(signals[2:], rmse[1:],width = 0.5)
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model evaluation")
    
    plt.show()




if __name__ == "__main__":
    main()





# Compute lags
# for i in range(1,len(times)):
#     # Compute lag with cross correlation
#     corr = signal.correlate(setpoints[i], setpoints[0], mode='full')
#     lags = signal.correlation_lags(len(setpoints[i]), len(setpoints[0]), mode='full')
#     lag = lags[np.argmax(corr)]/400

#     # Get start lag
#     lag += times[i][0] - times[0][0]

#     #times[i] -= lag
#     print(lag, " seconds")

# times[1] -= 0
# times[2] -= 0