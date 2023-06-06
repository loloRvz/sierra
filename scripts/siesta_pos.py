#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

from derivative import SavitzkyGolay
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.linalg import toeplitz

import torch
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn import Module
from torch.nn import Softsign
from torch.nn import Linear
from torch import Tensor
from torch.utils.data import random_split, Subset, DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_squared_error

# Data columns
TIME, SETPOINT, POSITION, CURRENT, VELOCITY_COMP, ACCELERATION_COMP = range(6)

LOAD_INERTIAS = np.array([1.7e-3, \
                          1.7e-3 + 224.5440144e-6, \
                          1.7e-3 + 548.4378187e-6, \
                          1.7e-3 + 287.7428055e-6, \
                          1.7e-3 + 548.4378187e-6 + 224.5440144e-6, \
                          1, 1, 1, 1, 
                          287e-6])    #kg*m^2

MOTOR_FRICTIONS = np.array([5e-3, \
                            5e-3, \
                            5e-3, \
                            5e-3, \
                            5e-3, \
                            1, 1, 1, 1, 
                            20e-3])    #kg*m^2/s

K_TI = 1 # T = K_TI * I


def ode_func(t,w,data,k,J,b):
    i_t = interp1d(data[:,TIME], data[:,CURRENT]/1000)
    # compute acceleration
    dwdt = k/J * i_t(t) - b/J * w
    return dwdt


### CLASSES ###

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        self.path = path
        self.load_id = int(os.path.basename(self.path)[26])
        self.df = pd.read_csv(path, dtype=np.float64)
        self.X = np.empty([1,1]).astype('float64')
        self.y = np.empty([1]).astype('float64')

        print("Loading model: ", os.path.basename(self.path))
        print("Load id: ", self.load_id)
        print("Load inertia: ", LOAD_INERTIAS[self.load_id])
        
    # preprocess data (compute velocities and accelerations)
    def preprocess(self, resave=False):
        data = (self.df).to_numpy()

        # Compute position derivatives if necessary
        fd = SavitzkyGolay(left=3, right=3, order=1, iwindow=True)
        #fd = SavitzkyGolay(left=0.005, right=0.005, order=1, iwindow=False)

        # Compute velocity from position 
        if np.sum(np.isnan(data[:,VELOCITY_COMP])) > 1 or resave:
            data[:,VELOCITY_COMP] = fd.d(data[:,POSITION],data[:,TIME])
            resave = True

        # Compute acceleration from velocity or velocity_comp
        if np.sum(np.isnan(data[:,ACCELERATION_COMP])) > 1 or resave:
            data[:,ACCELERATION_COMP] = fd.d(data[:,VELOCITY_COMP],data[:,TIME])
            resave = True

        #Integrate current (torque) to get velocity
        # print("Integrating velocity from current...")
        # # Find velocity values close to zeros
        # eps = 0.1
        # v_is_nill = np.logical_and(data[:,VELOCITY_COMP] <= eps, data[:,VELOCITY_COMP] >= -eps)

        # # Integrate between those values
        # idx1 = 0
        # for i,v in np.ndenumerate(v_is_nill):
        #     if i[0] == 0:
        #         data[0,VELOCITY_INT] = 0
        #         continue

        #     if i[0] == len(v_is_nill)-1:
        #         print(i[0])
        #         v = True

        #     if v:
        #         idx2 = i[0]
        #         sol = solve_ivp(ode_func,(data[idx1,TIME],data[idx2,TIME]),[0], \
        #             method="RK23", \
        #             t_eval = data[idx1:idx2,TIME], \
        #             args=(data,K_TI,LOAD_INERTIAS[self.load_id],MOTOR_FRICTIONS[self.load_id]))
        #         data[idx1:idx2,VELOCITY_INT] = sol.y
        #         idx1 = idx2

        # data[idx2,VELOCITY_INT] = 0
        # resave = True
        
        # Save dataframe to csv if velocity or acceleration computed
        if resave:
            print("Computed derivatives. Resaving dataframe to csv...")
            self.df = pd.DataFrame(data, columns = self.df.columns.values, dtype=np.float32)
            self.df.to_csv(self.path, index=False)

    # plot dataset
    def plot_data(self):
        data = self.df.to_numpy()

        # Make data a bit more readable - ignore units for now
        data[:,CURRENT] /= 1000
        data[:,VELOCITY_COMP] /= 10
        data[:,ACCELERATION_COMP] /= 1000

        fig,ax=plt.subplots()
        #ax.plot(data[:,TIME],data[:,SETPOINT:ACCELERATION_COMP+1])
        ax.plot(data[:,TIME],data[:,SETPOINT])
        ax.plot(data[:,TIME],data[:,POSITION])
        ax.plot(data[:,TIME],data[:,VELOCITY_COMP])
        ax.plot(data[:,TIME],data[:,ACCELERATION_COMP])
        ax.plot(data[:,TIME],data[:,CURRENT])
        ax.axhline(y=0, color='k')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend([ "Setpoint [rad]", \
                    "Position [rad]", \
                    "Derived Velocity [10rad/s]", \
                    "Derived Accleration [1000rad/s^2]", \
                    "Current [A]"]) # \                      
        plt.title("Motor data reading @400Hz")

        """ 
        fig2,ax=plt.subplots()
        plt.scatter(data[:,ACCELERATION_COMP],data[:,CURRENT])
        ax.set_xlabel("Acceleration [1000rad/^2]")
        ax.set_ylabel("Current [A]")                      
        plt.title("Current vs Acceleration Relation")
        """

        plt.show()

    # prepare inputs and labels for learning process
    def prepare_data(self,hist_len,T_via, freq=400):
        data = self.df.to_numpy(dtype=np.float64)
        self.X = np.resize(self.X,(data.shape[0],hist_len))

        n = 1

        #Get position error history
        for i in range(hist_len-1):
            self.X[:,i] = np.roll(data[:,POSITION], n*i)
            self.X[:n*i,i] = np.nan
        self.X[:,hist_len-1] = data[:,SETPOINT] - data[:,POSITION]
        self.X = self.X[n*(hist_len-1):,:] #Cut out t<0

        # Compute torque depending on load inertia (declared in filename)
        if T_via == 'a':
            self.y = data[:,ACCELERATION_COMP]*LOAD_INERTIAS[self.load_id]
        else:
            self.y = data[:,CURRENT] / 1000 * K_TI
        self.y = self.y[n*(hist_len-1):] #Cut out t<0

        # Get corresponding times
        self.t = data[n*(hist_len-1):,TIME]#Cut out t<0

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        print("Training size: ", train_size)
        print("Testing size: ", test_size)

        # calculate the split
        train, test = random_split(self, [train_size, test_size])
        train_dl = DataLoader(train, batch_size=32, shuffle=True, pin_memory=True)
        test_dl = DataLoader(test, batch_size=1024, shuffle=True, pin_memory=True)
        return train_dl, test_dl

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs, dev, layerDim):
        super(MLP, self).__init__()
        self.dev = dev
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, layerDim).to(self.dev)
        xavier_uniform_(self.hidden1.weight).to(self.dev)
        self.act1 = Softsign().to(self.dev)
        # second hidden layer
        self.hidden2 = Linear(layerDim, layerDim).to(self.dev)
        xavier_uniform_(self.hidden2.weight).to(self.dev)
        self.act2 = Softsign().to(self.dev)
        # third hidden layer
        self.hidden3 = Linear(layerDim, layerDim).to(self.dev)
        xavier_uniform_(self.hidden3.weight).to(self.dev)
        self.act3 = Softsign().to(self.dev)
        # output
        self.hidden4 = Linear(layerDim, n_outputs, bias=False).to(self.dev)
        xavier_uniform_(self.hidden4.weight).to(self.dev)
        
        self.to(torch.float64)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = X.to(self.dev)
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # fifth hidden layer and output
        X = self.hidden4(X)
        return X


### FUNCTIONS ###

# train model
def train_model(train_dl, test_dl, model, dev, model_dir, lr):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr, momentum=0.9)
    writer = SummaryWriter()
    # enumerate epochs
    epoch = 0
    try:
        while True:
            # Compute loss and gradient on train dataset
            meanLoss = 0
            steps = 0
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(dev), targets.to(dev).unsqueeze(1)
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
                meanLoss = meanLoss + loss
                steps = steps + 1

            # Compute loss on test dataset
            meanLossTest = 0
            stepsTest = 0
            for i, (inputs, targets) in enumerate(test_dl):
                inputs, targets = inputs.to(dev), targets.to(dev).unsqueeze(1).float()
                # compute the model output
                yhat = model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                meanLossTest = meanLossTest + loss
                stepsTest = stepsTest + 1

            meanLoss = meanLoss/steps
            writer.add_scalar('Loss/train', meanLoss, epoch)

            meanLossTest = meanLossTest/stepsTest
            writer.add_scalar('Loss/test', meanLossTest, epoch)
            
            if epoch % 10 == 0 and epoch != 0:
                print("Epoch: ", epoch)
            if epoch % 250 == 0 and epoch != 0:
                print("Epoch: ", epoch)
                model_scripted = torch.jit.script(model)
                model_scripted.double()
                model_scripted.save(model_dir +  "/delta_" + str(epoch) + ".pt")
            # if epoch >= 300:
            #     break
            epoch = epoch + 1
    except KeyboardInterrupt:
        print('interrupted!')

# evaluate model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)

    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    std = np.std(actuals)
    return mse, std

# plot predictions
def plot_model_predictions(dataset, model, RMSE):
    # Get full dataloader from set
    full_dl = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)

    # Compute predictions
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(full_dl):
        yhat = model(inputs)
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)

    # Plot validation dataset to time
    fig,ax=plt.subplots()
    ax.plot(dataset.t,dataset.X[:,0])
    ax.plot(dataset.t,np.squeeze(actuals))
    ax.plot(dataset.t,np.squeeze(predictions))
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.axhline(y=0, color='k')
    ax.legend(["Position error [rad]","Measured torque [Nm]","Predicted torque [Nm]"])
    plt.title("Model Validation | RMSE: %f" % RMSE)
    plt.show()


### SCRIPT ###
def main():
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Using GPU!")
    else:
        dev = "cpu"
        print("Using CPU D:")

    # Model parameters
    h_len = 8
    T_via = 'a'

    # Open training dataset
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/training/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()
    path = list_of_files[1]
    print("Opening: ",path)

    # Prepare dataset
    dataset = CSVDataset(path)
    dataset.preprocess(resave=True)
    dataset.prepare_data(hist_len=h_len, T_via = T_via)
    train_dl, test_dl = dataset.get_splits(n_test=0.1) # Get data loaders

    # Make dir for model
    model_dir = "../data/models/"+os.path.basename(path)[:-4]+"-POS"+str(h_len).zfill(2)+"_T"+T_via
    print("Opening directory: ",model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Train model
    model = MLP(h_len, 1, dev, 32)
    model.to(torch.float64)
    train_model(train_dl, test_dl, model, dev, model_dir, lr=0.01)

    # Evaluate model
    mse,std = evaluate_model(test_dl, model)
    print('MSE: %.3f, RMSE: %.3f, STD: %.3f' % (mse, np.sqrt(mse), std))



if __name__ == "__main__":
    main()
