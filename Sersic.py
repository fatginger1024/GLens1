from math import *
import numpy as np
from NFWlens import NFWlens
from scipy.optimize import bisect
from scipy.special import gamma,gammainc
from matplotlib.pyplot import figure,show

class Sersic():
    """
    =======================================================
    Galaxy parameters:
    =======================================================
    - Mstar: the mass of the galaxy
    - Re: effective radius of the galaxy, i.e. the projected 
    radius encircling half of the total luminosity
    - m: Sersic law index, default set to be 4. Normally 
    1<= m <= 4
    =======================================================
    """
    
    def __init__(self,Mstar=1e12,Re=3,m=4):
        self.Mstar =  Mstar#Msun
        self.Re = Re/1e3 #Mpc
        self.m = m
        
        
    def b(self):
        m = self.m
        return 2*m-1/3.+4/(405*m)+46/(25515*m**2)+131/(1148175*m**3)-2194697/(30690717750*m**4)

    def L(self):
        Re = self.Re
        m =  self.m
        return Re**2*2*pi*m/self.b()**(2*m)*gamma(2*m)

    def Sigma(self,R):
        m = self.m
        Re = self.Re
        return self.Mstar*np.exp(-self.b()*(R/Re)**(1./m))/self.L()
   
    
    def M2d(self,R):
        Re = self.Re
        m = self.m
        b = self.b()
        return 2*pi*self.Mstar/self.L()*m*Re**2/b**(2*m)*gammainc(2*m,b*R**(1/m)/Re**(1/m))*gamma(2*m)
    
if __name__=="__main__":
    galaxy = Sersic()
    lens =  NFWlens()
    Sigma_cr = lens.d.get_Sigmacr()
    Da1 = lens.d.angular_diameter_distance(lens.d.z0,lens.d.z1)
    xmin,xmax = -7,1
    x = np.logspace(xmin,xmax,10000)
    y = np.concatenate([(lens.M2d(rx)+galaxy.M2d(rx))/(pi*rx**2) for rx in x])
    fig = figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    ax.plot(x,y,'-')
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\bar{\Sigma}(<r)$")
    ax.axhline(y=Sigma_cr,lw=2,color='k')
    show()
    func = lambda x:(lens.M2d(x)+galaxy.M2d(x))/(pi*x**2)-Sigma_cr
    rt = bisect(func,1e-8,1)
    thetaE = rt/Da1*206265
    print("The Einstein radius is found to be: ","%.3e" % rt,"Mpc.")
    print("Equivalent to: ","%.3e" % thetaE,"arcsec.")