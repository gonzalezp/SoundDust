"""
To-Do:
    (1) Create a function that returns energy eigenfunctions
    (2) Create a function that returns energy eigenvalues
    (3) Create a function that returns time-compnent of energy eigenfunction
    (4) Plot static energy eigenfunctions/probability densities
    (5) Animate time-dependent energy eigenfunctions/probabillity density
"""

import numpy as np
from matplotlib import pyplot as plt

# Define some global variables
### atomic units
hbar = 1
mass = 1
radius = 1
ci = 0+1j
### moment of inertia
I = mass*radius*radius

### define an array of angles between 0 and 2pi
theta = np.linspace(0, 2*np.pi, 100)

def EnergyFunc(m):
    return np.sqrt(1/(2*np.pi))*np.exp(ci*m*theta)


### lets look at an array ( coefficient)

psi4 = EnergyFunc(4)
### if you want to calculate probability density, do this below
###first input the conjugate of psi4
psi4star = np.conj(psi4)
### now do your input for probability density
probdens = (psi4star)*(psi4)
### creating a plot of our new array made of complex numbers
### We need to put theta, and the either the real or imaginary part of our function
###for this we will plot both

### ploting function
###plt.plot(theta, np.real(psi4), 'red', theta, np.imag(psi4), 'blue')
###plt.show()

###ploting probability
plt.plot(theta, np.real(probdens), 'red')

