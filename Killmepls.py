import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from numpy.random import choice


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2*np.pi), ylim=(-5,5))
line, = ax.plot([], [], lw=2)

def PR_Func(theta,n,r):
        ci = 0.+1j
        pr = 1./(np.sqrt(2*np.pi*r))
        pz = np.exp(ci*n*theta)
        psi = pr*pz
        return psi

def Gauss_Packet(sig, x, x0, k0):
        ci = 0. + 1j

        pre = 1./(sig*np.sqrt(2.*np.pi))
        gx = np.exp(-0.5*((x-x0)/sig)**2)
        pw = np.exp(ci*k0*x)
        return pre*gx*pw

def FourierAnalysis(x, PsiX, n, r):
    cn = np.zeros(len(n),dtype=complex)
    dx = x[1]-x[0]
    for i in range (0,len(n)):

        som = 0+0j
        psi_i = PR_Func(x, n[i], r)

        for j in range (0, len(x)):
           som = som + psi_i[j]*PsiX[j]*dx

        cn[i] = som

    return cn
def PR_En(n,r):
    En=(n**2/2*r**2)
    return En

def PR_Time(n, r, t):
    E = PR_En(n, r)
    ci = 0.+1j
    phi_n_t = np.exp(-1*ci*E*t)
    return phi_n_t


def Normalize(pu):
    som=0
    for i in range(0,len(pu)):

        som=som +pu[i]

    for i in range(0,len(pu)):
        temp=pu[i]/som
        pu[i]=temp
    return pu

'''
P = (np.conj(y))*y
list_of_candidates = x
Pn = Normalize(P)
pr = np.real(Pn)
### Draw a random number using probability distribution
draw = choice(list_of_candidates, 100, p=pr)
print(draw)
x0 = draw[0]   
NewWvfxn = Pos_Eignfn(sig, xs, x0)
plt.plot(x, np.real(y), 'green', x, np.real(P), 'orange')
plt.show()
'''
rr = 3*np.pi/2.

### If you initialize in state n with energy En, then
### measure position, yielding value x0... expand the position eigenfunction
### at x0 in terms of the energy eigenfunctions... yielding array of expansion
### coefficiencts cn... then measure energy again, yielding energy value Em
### then you have the following three pieces of data:
### 1. n -> quantum number of initial state (for time<30)
### 2. cn -> array of expansion coefficients for position eigenfunction (for 30<t<60)
### 3. m  -> quantum number of final state (for time>60)
k0=0
xt = np.linspace(0,rr,500)
nx = np.linspace(1,10,10)
y = PR_Func(xt,6,rr)
P = np.real(np.conj(y)*y)
cn = FourierAnalysis(xt, y, nx, rr)
print(cn)


list_of_candidates=xt
Pn=Normalize(P)
pr=np.real(P)
draw=choice(list_of_candidates,15,p=pr)
print("Measurement of position yielded x0 = ",draw[0])

PosEigenfxn = Gauss_Packet(0.5, xt, draw[1], k0)
cn2 = FourierAnalysis(xt, PosEigenfxn, nx, rr)

#energy=PIB_En(n, L)
#list_of_candidates2=energy
#Pn=Normalize(P)
#pr=np.real(P)
#draw2=choice(list_of_candidates2, 15, p=pr)
#print("Measurement of energy yielded Ex = ", draw2[0])
psi_exp = np.zeros_like(y)

for i in range (0,len(cn)):
    psi_exp = psi_exp + cn[i]*PR_Func(xt, nx[i]*PR_Time(nx[i],rr,30), rr)

def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially to generate the animation
def animate(i):

    ### Once PIB_Func and PIB_En are defined, the following
    ### code can be used to plot the time-evolution of an energy eigenfunction

    ### Define x-grid - this will be for a particle in a box of length L=30 atomic units (Bohr radii)
    ### We will represent the function with 1000 grid points (dx = 30/1000)

    ### when i < 30, animate your initial state (some energy eigenfunction)
        x = np.linspace(0,rr,500)
    ### Imaginary unit i
        ci = 0.+1j
        psi_t = np.zeros(len(x),dtype=complex)

        if i<30:
                psi = PR_Func(xt,6,rr)
                ft = PR_Time(6, rr, i)
                psi_t = psi*ft

        #psi_t = PIB_Func(x, 10, L)*PIB_Time(10, L, 10*i)
        #print(i)
        #print(PIB_Time(10,L,10*i))


    ### at t=30, a position measurement was made... yielding x0 = 22
    ### so for t>30 && t<60, animate the position eigenfunction that is centered
    ### at x0 = 22
        else:
                if i<60:
#        print(i)

                        for g in range(0,len(cn2)):
                                ps = PR_Func(xt,nx[g],rr)
                                ft = PR_Time(nx[g],rr,i-30)
                                psi_t = psi_t + cn2[g]*ft*ps
#        psi_t= PIB_Func(x,10,L)*PIB_Time(10, L, i*100)

                else:

                        if i<10000:
                                psi = PR_Func(xt, 20,rr)
                                ft = PR_Time(20, rr, i)
                                psi_t = psi*ft


#        for g in range(len(cn2)):
#        psi_t = psi_t + cn2[g]*cn2[g]*PIB_Time(n[g], L, i*100)
        psi_t_star = np.conj(psi_t)
        y = np.real(psi_t)
        z = np.imag(psi_t)
        p = np.real(y*y+z*z)
        line.set_data(xt, y)
        return line,

    ### at t=60, an energy is measured, yielding E0 = ....
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval=200, blit=True)
### uncomment to save animation as mp4
#anim.save('pib_wp.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
plt.show()
