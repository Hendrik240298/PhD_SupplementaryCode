import os
import numpy as np
import matplotlib.pyplot as plt

path2fem = '../result/FEM/'
path2rom = '../result/ROM/'

filename =  '../result/FEM/mu=0.001000/pressure.txt'

RE = np.array([110,120,130])

for i in range(0,RE.size):
    mu = 'mu=' + "{0:5f}".format(0.1/RE[i])
    pressure = np.loadtxt(path2fem+mu+ '/pressure.txt',dtype=float,delimiter=',')
    pressure_rom =  np.loadtxt(path2rom+mu+ '/pressure.txt',dtype=float,delimiter=',')
    plt.figure(i)
    plt.plot(pressure_rom[:,0],pressure[:,1])
    plt.plot(pressure_rom[:,0],pressure[:,1])
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('pressure')
    plt.xlim([pressure_rom[0,0],pressure_rom[np.shape(pressure_rom)[0]-1,0]])

plt.show()
#plt.show()
#with open(filename, 'r') as f:
#    for line in f.readlines():
#        data = np.vstack((data, [1,1]))
        #print(line.split(','))
#print(data)
