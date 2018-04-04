# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:04:54 2018

@author: ampomahn
"""

"""
TO-DO:
    (1) Create a function that returns energy eigenefuntions
    (2) Create a funtion that returns energy eigenvalues
    (3) Create a function that returns time-compmnent of energy eigenfunctions
    (4) Plot static energy eigenfunctions/probability densities
    (5) Animate time-dependent energy eigenfuntions/probability densities
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig= plt.figure()
ax = plt.axes(xlim=(0, 2*np.pi), ylim=(-0.5, 0.5))
line, = ax.plot([], [], lw=2)

# Define some globl variables
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

def EnergyValue (m):
    return hbar*hbar*m*m/(2*I)

def TimeComponent(m, t):
    E = EnergyValue(m)
    tc = np.exp(-ci*E*t/hbar)
    return tc

Psi = np.zeros(len(theta))
Psi = EnergyFunc(5)

def init():
    line.set_data([], [])
    return line,


def animate(i):
    
    psi = EnergyFunc(4)
    #ft is funtion of time
    ft = TimeComponent(4, i)
    y = np.real(psi*ft)
    
    
    line.set_data(theta, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval = 100, blit = True)
#m=4
#psi4 = EnergyFunc(4)
#conjugate of psi4
#psi4star = np.conj(psi4)
#density = (psi4star)*(psi4)
#probdens = (psi4star)*(psi4)

#m=6
#psi6 = EnergyFunc(6)

#psi6star = np.conj(psi6)

#probdens = (psi6star)*(psi6)

#superpsi = (psi4star+psi6star)*(psi4+psi6)

#plt.plot(theta, np.real(probdens), 'red', theta, np.real(superpsi), 'blue')
plt.show()
#now plot for theta