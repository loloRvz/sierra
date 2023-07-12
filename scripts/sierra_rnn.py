#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn import Module
from torch.nn import Softsign
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.utils.data import random_split, Subset, DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_squared_error

# Data columns
TIME, SETPOINT, VELOCITY = range(3)

MIN_RPM = 1500
MAX_RPM = 9000
MAX_RPM_PER_TICK = 20

### CLASSES ###

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        self.path = path
        self.df = pd.read_csv(path, dtype=np.float64)
        self.X = np.empty([1,1]).astype('float64')
        self.y = np.empty([1]).astype('float64')

        print("Loading model: ", os.path.basename(self.path))
        
    # plot dataset
    def plot_data(self):
        data = self.df.to_numpy()

        fig,ax=plt.subplots()
        ax.plot(data[:,TIME],data[:,SETPOINT])
        ax.plot(data[:,TIME],data[:,VELOCITY])
        ax.axhline(y=0, color='k')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend([ "Setpoint [RPM]", \
                    "Velocity [RPM]"]) # \                      
        plt.title("Motor data reading")

        plt.show()

    # preprocess dataset
    def preprocess(self):
        data = self.df.to_numpy(dtype=np.float64)
        data = data[500:,:] #Cut out ramp up time
        data = data[data[:,VELOCITY] > 1000]
        self.df = pd.DataFrame(data, columns = self.df.columns.values, dtype=np.float32)

    # prepare inputs and labels for learning process
    def prepare_data(self,hist_len):
        data = self.df.to_numpy(dtype=np.float64)

        self.X = np.resize(self.X,(data.shape[0],hist_len+1))

        # Get input data
        self.X[:,0] = data[:,SETPOINT]
        for i in range(hist_len):
            self.X[:,i+1] = np.roll(data[:,VELOCITY], i)
        
        # Get output data
        self.y = np.roll(data[:,VELOCITY], -1) - data[:,VELOCITY]

        # Cut out t<0
        self.X = self.X[hist_len:-1,:] 
        self.y = self.y[hist_len:-1]

        # Get corresponding times
        self.t = data[hist_len:-1,TIME] #Cut out t<0

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        print("Training size: ", train_size)
        print("Testing size: ", test_size)

        # calculate the split
        train, test = random_split(self, [train_size, test_size])
        train_dl = DataLoader(train, batch_size=32, shuffle=True)
        test_dl = DataLoader(test, batch_size=32, shuffle=True)
        return train_dl, test_dl

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition
class RNN(Module):
    # define model elements
    def __init__(self, input_dim, output_dim, hidden_dim, layer_num, dev):
        super(RNN, self).__init__()
        self.dev = dev

        self.min_rpm = MIN_RPM
        self.max_rpm = MAX_RPM
        self.max_rpm_per_tick = MAX_RPM_PER_TICK

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_num, batch_first=True, nonlinearity='relu')
        self.fc = Linear(hidden_dim, output_dim)

        self.to(torch.float64)

    # forward propagate input
    def forward(self, X):
        h = torch.zeros(self.layer_num, X.size(0), self.hidden_dim).requires_grad_()

        X = X.to(self.dev)
        X = torch.sub(X,self.min_rpm)
        X = torch.div(X,self.max_rpm-self.min_rpm)

        outp, hx = self.rnn(X, h.detach())
        outp = self.fc(outp[:, -1, :]) 
        return outp

         
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
            for inputs, targets in train_dl:
                inputs, targets = inputs.to(dev), targets.to(dev).unsqueeze(1)
                model.train()
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
                meanLoss += loss
                steps += 1

            # Compute loss on test dataset
            meanLossTest = 0
            stepsTest = 0
            for inputs, targets in test_dl:
                inputs, targets = inputs.to(dev), targets.to(dev).unsqueeze(1)
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
            if epoch % 100 == 0 and epoch != 0:
                print("Epoch: ", epoch)
                model_scripted = torch.jit.script(model)
                model_scripted.double()
                model_scripted.save(model_dir +  "/delta_" + str(epoch).zfill(4) + ".pt")
            if epoch >= 2000:
                break
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
    ax.legend(["Sepoint [kRPM]","Measured Velocity [kRPM]","Predicted Velocity [kRPM]"])
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
    h_len = 7

    # Open training dataset
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/training/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()
    path = list_of_files[0]
    print("Opening: ",path)

    # Prepare dataset
    dataset = CSVDataset(path)
    dataset.preprocess()
    dataset.prepare_data(hist_len=h_len)
    train_dl, test_dl = dataset.get_splits(n_test=0.1) # Get data loaders

    # Make dir for model
    model_dir = "../data/models/"+os.path.basename(path)[:-4]+"-PHL"+str(h_len).zfill(2)+"RNN"
    print("Opening directory: ",model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Train model
    model = RNN(input_dim=h_len+1, output_dim=1, hidden_dim=32, layer_num=2, dev=dev)
    model.to(torch.float64)
    train_model(train_dl, test_dl, model, dev, model_dir, lr=0.0001)

    # Evaluate model
    mse,std = evaluate_model(test_dl, model)
    print('MSE: %.3f, RMSE: %.3f, STD: %.3f' % (mse, np.sqrt(mse), std))



if __name__ == "__main__":
    main()
