#! /usr/bin/env python3


from sierra_mlp import *
from generate_input_signals import *


### SCRIPT ###
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__)) # Directory of this file

    ### SIMULATION SETUP ###

    # Params
    T1 = 0          # [s]
    T2 = 55         # [s]
    F = 200         # [Hz]
    N = (T2-T1)*F   # []
    START = 4000    # [kRPM]

    # Input signal
    input_type = FLITREAL
    input_dataset = CSVDataset("../data/input_signals/signals.csv")
    input_np = input_dataset.df.to_numpy(dtype=np.float64)
    setpoint = input_np[:N,input_type]
    time = np.linspace(T1,T2, N)

    # Data arrays
    signal_names = []
    times = []
    setpoints = []
    velocities = []


    ### MEASURED DATA ###

    # Get measured signal (true)
    eval_data_dirs = ["0-step","1-ramp","2-chrp","3-flit","4-nois","5-mixd","6-flitreal"]
    eval_data_path = sorted(glob.glob("../data/evaluation/" + eval_data_dirs[input_type] + "/*.csv"))[0]
    print("Opening measurement data: ", eval_data_path)
    eval_dataset = CSVDataset(eval_data_path)
    eval_dataset.preprocess()
    eval_np = eval_dataset.df.to_numpy(dtype=np.float64)
    # setpoint = eval_np[:N,SETPOINT]

    # Save data to arrays
    signal_names.append("Measurement")
    times.append(eval_np[:,TIME])
    setpoints.append(eval_np[:,SETPOINT])
    velocities.append(eval_np[:,VELOCITY])


    ### P-CONTROLLER ###

    # Params
    K_P = 0.15
    cmd_max = 15.0
    cmd_min = -cmd_max

    # Init state
    velocity = np.zeros(time.shape)
    prev_vel = START

    # Simulate response
    for i in range(N):
        velocity[i] = prev_vel

        cmd = K_P*(setpoint[i] - velocity[i])
        if cmd > cmd_max: cmd = cmd_max
        if cmd < cmd_min: cmd = cmd_min

        prev_vel = velocity[i] + cmd

    # Save data to arrays
    signal_names.append("P Model")
    times.append(time)
    setpoints.append(setpoint)
    velocities.append(velocity)


    ### NEURAL NETWORK MODEL ###

    # Load pre-trained model
    model_dirs = dir_path + "/../data/models/" + "23-07-27--16-14-41_flit-PHL05_2hid32/delta_8000"
    list_of_models = glob.glob(model_dirs + '*.pt')
    list_of_models = sorted(list_of_models)

    # Model parameter
    h_len = 5

    # Simulate all models
    for model_dir in list_of_models[:]:
        print("Opening model:", model_dir[-43:])
        model = torch.jit.load(model_dir)

        # Init state
        velocity = np.zeros(time.shape)

        prev_set = np.ones(1)*START
        prev_vel = np.ones(h_len)*START

        # Simulate response
        for i in range(N):
            velocity[i] = prev_vel[0]

            in_tens = torch.tensor(np.concatenate((prev_set,prev_vel), axis=None), dtype=torch.float64)
            out_tens = model(in_tens)

            prev_set = np.roll(prev_set,1)
            prev_vel = np.roll(prev_vel,1)
            
            prev_set[0] = setpoint[i]
            prev_vel[0] = velocity[i] + out_tens.item()

        # Plot validation dataset to time
        signal_names.append("Epoch: " + model_dir[-7:-3])
        times.append(time)
        setpoints.append(setpoint)
        velocities.append(velocity)


    ### DATA PROCESSING ###

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


    ### PLOTTING ###

    # Print RMSE of all signals
    print("RMSE")
    for i in range(len(velocities)):
        print("Signal: ",signal_names[i],", RMSE: ",rmse[i])

    # Plot signals
    plt.figure(1,figsize=(14,5))
    plt.plot(times[0],setpoints[0],linestyle="dashed",linewidth=2)            # Plot setpoint
    plt.plot(times[0],velocities_interp[0],linewidth=2)    # Plot measured values
    # Plot NN models
    for i in range(1,len(velocities)):
        plt.plot(times[0],velocities_interp[i],linewidth=1.5)
    # for i in range(1,len(velocities)):
    #     plt.plot(times[0],errors_interp[i],"--",color="C"+str(i+1))
    leg = ["Setpoint","Measurement","P Model","NN Model"]
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [RPM]")
    plt.legend(leg)                       
    plt.subplots_adjust(left=0.08,right=0.96)
    plt.xlim([10,20])
    plt.ylim([6000,6600])
    #plt.title("Velocity Control Model Simulation")

    plt.figure(2,figsize=(7,5))
    plt.plot([int(epoch[-4:]) for epoch in signal_names[2:]],rmse[2:])
    plt.hlines(rmse[1],0,int(signal_names[-1][-4:]),color="green")
    plt.legend(["NN Models","P-Controller Simulation"])
    plt.xlabel("Model Epoch")
    plt.ylabel("RMSE")
    plt.ylim([0, 100])
    plt.title("Model Performance")

    try:
        plt.figure(3,figsize=(2.5,5))
        bars = (["P Model","NN Model"])
        x_pos = [0,1]
        plt.bar(x_pos, rmse[1:], width = 0.5, align='center')
        plt.subplots_adjust(left=0.275)
        plt.xticks(x_pos, bars)
        plt.xlabel("Model Type")
        plt.ylabel("RMSE")
        plt.title("Model Performance")
    except:
        print("Error plotting bar plot")

    
    plt.show()





if __name__ == "__main__":
    main()