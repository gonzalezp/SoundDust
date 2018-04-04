"""
To-Do:
    (I) Create a function that returns energy eigenfunctions
    (II) Create a function that returns energy eigenvalues
    (III) Create a function that returns time-compnent of energy eigenfunction
    (IV) Plot static energy eigenfunctions/probability densities
    (V) Animate time-dependent energy eigenfunctions/probabillity density
"""

import numpy as np
from matplotlib import pyplot as plt
### if we want to animate videowise we use
from matplotlib import animation
#then we set up the figure and fill in for
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2*np.pi), ylim=(-0.5, 0.5))
line, = ax.plot([], [], lw=2)

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
###energyeigenvalue of particle in a ring which is the 2nd derivative of enegeryfunc times hbar/
def EnergyValue(m): 
    return hbar*hbar*m*m/(2*I)
###now lets write how our enegry value evolves in time
def TimeComponent(m, t):
    E = EnergyValue(m)
    tc = np.exp(-ci*E*t/hbar)
    return tc


    
#
#making an empty array
Psi = np.zeros(len(theta))
#print(Psi)
#foranimation
Psi = EnergyFunc(5)

### making a loop thorugh values of m from 0 to 20
#for m in range(0,21) :
    #psim = EnergyFunc(m)
    #Psi= Psi + np.sqrt(1/20)*psim
    #print(Psi)


### lets look at an array ( coefficient)

###psi4 = EnergyFunc(4)
### if you want to calculate probability density, do this below
###first input the conjugate of psi4
#psi4star = np.conj(psi4)
### now do your input for probability density
#probdens = (psi4star)*(psi4)
### creating a plot of our new array made of complex numbers
### We need to put theta, and the either the real or imaginary part of our function
###for this we will plot both

###now lets add another coefficient like c6
##psi6 = EnergyFunc(6)
##psi6star = np.conj(psi6)
### setting another probability density specific to c6 below
#probdens2 = (psi6star)*(psi6)

### if we went to plot a superposition of c4 and c6 we do this
#superpositionpsi = (psi6star + psi4star)*(psi4 + psi6)

###plotting super position, in this case psi4 and psi6
#plt.plot(theta, np.real(superpositionpsi), 'red', theta, np.imag(superpositionpsi), 'blue')
#plt.show()

### ploting function
#plt.plot(theta, np.real(psi4), 'red', theta, np.imag(psi4), 'blue')
#plt.show()

###ploting probability
###plt.plot(theta, np.real(probdens), 'red')
###plt.show()

###ploting the loop array
##plt.plot(theta, np.real(Psi), 'red', theta, np.imag(Psi), 'blue')
##plt.show()
    ### to animate use
def init():
    line.set_data([],[])
    return line,

def animate(i):
    ### ft is the time component and i is used instead of time
    
   # psi = EnergyFunc(4)
   # ft = TimeComponent (4, i)
   # y = psi*ft
   # psistar = np.conj(y)
   # line.set_data(theta, y)
   # return line,

#probability density of our energyfunction
  psi = EnergyFunc(4)
  ft = TimeComponent (4, i)
  y = psi*ft
  psistar = np.conj(y)
  pbdense = np.real(y*psistar)
  line.set_data(theta, pbdense)
  return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval=100, blit=True)
plt.show ()