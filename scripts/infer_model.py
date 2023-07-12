#! /usr/bin/env python3


from sierra_mlp import *
from generate_input_signals import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Load trained model
    model_dirs = dir_path + "/../data/models/" + "23-07-04--16-42-16_mixd_overfit-PHL05/"
    list_of_models = glob.glob(model_dirs + '*.pt')
    list_of_models = sorted(list_of_models)

    # Model parameters
    h_len = 5

    # Small simulation
    T1 = 0      # [s]
    T2 = 60     # [s]
    F = 200     # [Hz]
    N = (T2-T1)*F
    SET = 6000     # [kRPM]
    START = 5000  # [kRPM]

    # Get input signal
    input_type = MIXD
    input_df = pd.read_csv("../data/input_signals/signals.csv", dtype=np.float64)
    input_np = input_df.to_numpy(dtype=np.float64)
    setpoint = input_np[:N,input_type]
    time = np.linspace(T1,T2, N)

    # Load data from files
    signal_names = []
    times = []
    setpoints = []
    velocities = []

    # Get measured signal
    eval_data_dirs = ["0-step","1-ramp","2-chrp","3-flit","4-nois","5-mixd","6-flitreal"]
    eval_data_path = sorted(glob.glob("../data/evaluation/" + eval_data_dirs[input_type] + "/*.csv"))[0]
    print("Opening measurement data: ", eval_data_path)
    real_data = pd.read_csv(eval_data_path, dtype=np.float64).to_numpy()
    signal_names.append("Measurement")
    times.append(real_data[:,TIME])
    setpoints.append(real_data[:,SETPOINT])
    velocities.append(real_data[:,VELOCITY])

    # Simulate all models
    for model_dir in list_of_models:
        print("Opening model:", model_dir[-43:])
        model = torch.jit.load(model_dir)

        # Init state
        velocity = np.zeros(time.shape)
        state = np.ones(h_len)*START

        # Simulate response
        for i in range(N):
            velocity[i] = state[0]

            in_arr = np.insert(state,0,setpoint[i])
            in_tens = torch.tensor(in_arr, dtype=torch.float64)
            out_tens = model(in_tens)

            state = np.roll(state,1)
            state[0] = velocity[i] + out_tens.item()

        # Plot validation dataset to time
        signal_names.append("Epoch: " + model_dir[-7:-3])
        times.append(time)
        setpoints.append(setpoint)
        velocities.append(velocity)

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
    plt.bar([epoch[-4:] for epoch in signal_names[1:]], rmse[1:],width = 0.5)
    plt.xlabel("Model Epoch")
    plt.ylabel("RMSE")
    plt.title("Velocity control comparison")
    
    plt.show()





if __name__ == "__main__":
    main()