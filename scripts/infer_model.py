#! /usr/bin/env python3


from sierra_mlp import *
from generate_input_signals import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Load trained model
    model_dirs = dir_path + "/../data/models/" + "23-07-04--09-08-11_mixd-PHL05new/"
    list_of_models = glob.glob(model_dirs + '*.pt')
    list_of_models = sorted(list_of_models)

    # Model parameters
    h_len = 5

    # Small simulation
    T1 = 0      # [s]
    T2 = 30     # [s]
    F = 200     # [Hz]
    N = (T2-T1)*F
    SET = 6000     # [kRPM]
    START = 3000  # [kRPM]

    # Get input signal
    input_type = MIXD
    input_df = pd.read_csv("../data/input_signals/signals.csv", dtype=np.float64)
    input_np = input_df.to_numpy(dtype=np.float64)
    setpoint = input_np[:N,input_type]
    time = np.linspace(T1,T2, N)

    # Init plot
    plt.figure(1)
    plt.plot(time,setpoint,"--")

    # Open all models
    for model_dir in list_of_models:
        print("Opening model:", model_dir)
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
        plt.plot(time,velocity)


    legend = ["Epoch %d"%((i+1)*100) for i in range(len(list_of_models))]
    legend.insert(0,"Setpoint")
    plt.hlines(RPM_MIN,T1,T2)
    plt.hlines(RPM_MAX,T1,T2)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [RPM]")
    plt.legend(legend)
    plt.title("Model Simulation | Step responses")
    plt.show()





if __name__ == "__main__":
    main()