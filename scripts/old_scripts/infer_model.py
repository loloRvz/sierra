#! /usr/bin/env python3


from src.sierra.scripts.sierra_mlp import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Load trained model
    model_dir = dir_path + "/../data/models/" + "delta_5800.pt"
    print("Opening model:", model_dir)
    model = torch.jit.load(model_dir)

    # Model parameters
    h_len = 8

    #peh_arr = np.full((1,h_len), -0.097)
    peh_arr = np.full((1,h_len), -0.01)
    # peh_arr[0][h_len-1] = -1
    peh_tens = torch.tensor(peh_arr, dtype=torch.float64)
    out_tens = model(peh_tens)

    print("In: ", peh_tens)
    print("Out: ", out_tens)



if __name__ == "__main__":
    main()