#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

from derivative import SavitzkyGolay
from torch.utils.data import Dataset

import tfest
from scipy.signal import lsim,TransferFunction


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


### CLASSES ###

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        self.path = path
        self.load_id = int(os.path.basename(self.path)[26])
        self.df = pd.read_csv(path, dtype=np.float64)
        self.X = np.empty([1]).astype('float64')
        self.y = np.empty([1]).astype('float64')

        print("Loading model: ", os.path.basename(self.path))
        print("Load id: ", self.load_id)
        print("Load inertia: ", LOAD_INERTIAS[self.load_id])
        
    # preprocess data (compute velocities and accelerations)
    def preprocess(self, resave=False):
        data = (self.df).to_numpy()

        # Compute position derivatives if necessary
        fd = SavitzkyGolay(left=2, right=2, order=1, iwindow=True)

        # Compute velocity from position 
        if np.sum(np.isnan(data[:,VELOCITY_COMP])) > 1 or resave:
            data[:,VELOCITY_COMP] = fd.d(data[:,POSITION],data[:,TIME])
            resave = True

        # Compute acceleration from velocity or velocity_comp
        if np.sum(np.isnan(data[:,ACCELERATION_COMP])) > 1 or resave:
            data[:,ACCELERATION_COMP] = fd.d(data[:,VELOCITY_COMP],data[:,TIME])
            resave = True
        
        # Save dataframe to csv if velocity or acceleration computed
        # if resave:
        #     print("Computed derivatives. Resaving dataframe to csv...")
        #     self.df = pd.DataFrame(data, columns = self.df.columns.values, dtype=np.float32)
        #     self.df.to_csv(self.path, index=False)

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
    def prepare_data(self,T_via):
        data = self.df.to_numpy(dtype=np.float64)
        self.X = np.resize(self.X,(data.shape[0],1))

        # Get position errors (P)
        position_error = data[:,SETPOINT] - data[:,POSITION]
        self.X[:,0] = position_error

        # Compute torque depending on load inertia (declared in filename)
        if T_via == 'a':
            self.y = data[:,ACCELERATION_COMP]*LOAD_INERTIAS[self.load_id]
        else:
            self.y = data[:,CURRENT] / 1000 * K_TI


### FUNCTIONS ###


### SCRIPT ###
def main():

    # Model parameters
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
    dataset.prepare_data(T_via = T_via)
    data = dataset.df.to_numpy(dtype=np.float64)

    # Least squares
    A = np.transpose(np.stack((data[:,SETPOINT]-data[:,POSITION], data[:,VELOCITY_COMP])))
    trq = data[:,ACCELERATION_COMP]*LOAD_INERTIAS[dataset.load_id]
    Ks = np.linalg.lstsq( A, trq, rcond=None)[0]
    print("Least squares",Ks)
    predicted_torques = np.matmul(A, Ks)

    # SysID
    # N1 = 1000
    # N2 = 4000
    # te = tfest.tfest(data[N1:N2,SETPOINT]-data[N1:N2,POSITION], data[N1:N2,ACCELERATION_COMP]*LOAD_INERTIAS[dataset.load_id] ) 
    # te.estimate(0,2, method="fft", time=60)
    # print(te.get_transfer_function())

    # J = 287e-6
    # num = [0.015, 0.025]
    # den = [J, 0, 0]
    # tout, yout, xout = lsim(TransferFunction(num, den), data[:N,SETPOINT]-data[:N,POSITION], data[:N,TIME])


    plot = True
    if plot:
        # Plot validation dataset to time
        fig,ax=plt.subplots()
        ax.plot(data[:,TIME],data[:,SETPOINT])
        ax.plot(data[:,TIME],data[:,POSITION])
        ax.plot(data[:,TIME],data[:,ACCELERATION_COMP]*LOAD_INERTIAS[dataset.load_id])
        ax.plot(data[:,TIME],predicted_torques)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Amplitude")
        ax.axhline(y=0, color='k')
        ax.legend(["Setpoint [rad]","Position [rad]","Measured torque [Nm]","Predicted torque [Nm]","pos_err","pos_err_d"])
        plt.show()



if __name__ == "__main__":
    main()
