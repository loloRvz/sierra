#! /usr/bin/env python3


from sierra_mlp import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Open measured data
    list_of_files = glob.glob(dir_path + '/../data/evaluation/*.csv')
    list_of_files = sorted(list_of_files)
    #list_of_files.reverse()

    # Load trained model
    model_dir = dir_path + "/../data/models/" + "23-06-12--12-53-45_step-PHL07" + "/delta_3000.pt"
    print("Opening model:", model_dir)
    model = torch.jit.load(model_dir)

    # Model parameters
    h_len = 7

    # Loop through eval datasets
    for path in list_of_files:
        # Prepare data
        print("Opening: ",path)    
        dataset = CSVDataset(path)
        dataset.prepare_data(h_len)
        full_dl = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

        # Evaluate
        mse,std = evaluate_model(full_dl, model)
        print('RMSE: %.5f, STD: %.5f, NRMSE: %.5f' % (np.sqrt(mse), std, np.sqrt(mse)/std))
        plot_model_predictions(dataset, model, np.sqrt(mse))





if __name__ == "__main__":
    main()