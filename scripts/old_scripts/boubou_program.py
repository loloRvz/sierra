#! /usr/bin/env python3


from siesta_mlp import *

from scipy import signal
import math

### SCRIPT ###
def main():
    # y = np.array([math.log(200),math.log(8)])

    # A = np.matrix([[1 , -1000000], [1, -25000000]])

    # b = np.matmul(np.linalg.inv(A),y)
    # b = np.squeeze(b)

    #b[0] = math.exp(b[0])

    y = np.array([40,2])

    A = np.matrix([[1e6, 1], [25e6, 1]])

    b = np.matmul(np.linalg.inv(A),y)

    print("A/B", b[0][0])


if __name__ == "__main__":
    main()


