#! /usr/bin/env python3


from sierra_mlp import *
from generate_input_signals import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Load trained model
    model_dirs = dir_path + "/../data/models/" + "23-06-23--11-40-58_mixd-PHL07_denorm/"
    list_of_models = glob.glob(model_dirs + '*.pt')
    list_of_models = sorted(list_of_models)

    # Model parameters
    h_len = 7

    # Small simulation
    T1 = 0      # [s]
    T2 = 30     # [s]
    F = 200     # [Hz]
    N = (T2-T1)*F
    SET = 6000     # [kRPM]
    START = 1500  # [kRPM]

    # Generate randomly varying step input
    rand_inputs = np.random.normal(0, RAMP_VAR, DATA_LENGTH*RAMP_FREQ)
    rand_inputs[0] = RPM_START
    A = linalg.toeplitz( np.ones(rand_inputs.size), np.insert(np.zeros(rand_inputs.size-1),0,1) )
    rand_inputs = np.matmul(A,rand_inputs)
    rand_inputs = limit_signal(rand_inputs)
    step_inputs = np.repeat(rand_inputs, CTRL_FREQ/STEP_FREQ)

    time = np.linspace(T1,T2, N)
    setpoint = step_inputs[:N]

    plt.figure(1)
    plt.plot(time,setpoint,"--")

    for model_dir in list_of_models:
        print("Opening model:", model_dir)
        model = torch.jit.load(model_dir)

        velocity = np.zeros(time.shape)
        state = np.ones(h_len)*START

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