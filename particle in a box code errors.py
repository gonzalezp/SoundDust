import numpy as np
from matplotlib import pyplot as plt

def GaussPacket(x, x0, sigma, K0):
    ci = 0.+1j
    pre = 1./(sigma*np.sqrt(2*np.pi))
    gauss = np.exp(-0.5*((x-x0)/sigma)**2)
    pw = np.exp(ci*k0*x)
    return pre*gauss*pw
def PIB_Func(x, n, L):
    psi_n = np.sqrt(2./L)*np.sin(n*np.pi*x/L)
    return psi_n

def FourierAnalysis(x, PsiX, narray, L):
    cn = np.zeros(len(narray),dtype=complex)
    dx = x[1]-x[0]
    for i in range (0,len(cn)):
        som = 0+0j
        psi_i = PIB_Func(x, narray[i], L)
        for j in range (0, len(x)):
            
            som = som + psi_i[j]*PsiX[j]*dx
            cn[i] = som
            return cn

L = 500
x = np.linspace(0, 50, 50)
x0 =200.
k0 = 0
sigma = 15.

y= GaussPacket(x, x0, sigma, k0)

narray = np.linspace(1, 90, 90)
cn = FourierAnalysis(x, y, narray, L)
print(cn)
fg = np.zereos(len(x),dtype=complex)
for i in range(0,len(cn)):
    fg = fg+ cn[i]*PIB_Func(x, narray[i], L)

plt.plot(x, np.real(y), 'red', x, np.real(fg), 'b--')
plt.show()


