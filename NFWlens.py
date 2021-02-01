from math import *
import numpy as np
from Distances import Distances

class NFWlens():
    """
    =======================================================
    Lens Parameters:
    =======================================================
    - z0: the redshift of the observer, default is 0
    - z1: the redshift of the lens
    - z2: the redshift of the source
    - M200: the mass of the DM halo
    - c: concentration parameter of the lens halo 
    =======================================================
    Cosmological parameters:
    =======================================================
    - Om: matter density parameter
    - Or: radiation density parameter
    - Ol: dark energy density parameter
    - H0: Hubble constant
    =======================================================
    """
    def __init__(self,z0=0,z1=.3,z2=1.5,M200=1e15,c=5,Om=.25,Or=8.4e-5,Ol=.75,H0=70):
        self.H0  =  H0/3.086e19 #convert to s^-1
        self.zlens = z1
        self.h = H0/100
        self.Ez = np.sqrt(Or*(1+self.zlens)**4+Om*(1+self.zlens)**3+Ol)
        self.Hz = self.H0*self.Ez
        G  = 6.67e-11 # m^3  kg^-1 s^-2
        self.c = c #concentration parameter
        self.rho0 = 3*self.H0**2/(8*pi*G)/1.989e30/(3.24e-23)**3  # M_sun * Mpc^(-3)
        self.rhoz = 3*self.Hz**2/(8*pi*G)/1.989e30/(3.24e-23)**3  # M_sun * Mpc^(-3)
        self.rhos = 200/3*self.rhoz*self.c**3/(np.log(1+self.c)-c/(1+self.c))
        self.M200 = M200
        self.r200 = (self.M200/(4/3*pi*200*self.rhoz))**(1/3) #Mpc
        self.rs = self.r200/c #Mpc
        self.M3d1 = (np.log(1 + self.r200/self.rs) - self.r200/(self.r200+self.rs))/M200 #Msun ^-1
        self.rhoss = 1/(4*pi*self.rs**3)/self.M3d1
        self.d = Distances(z0=z0,z1=z1,z2=z2,Om=.25,Ol=.75,Or=8.4e-5,H0=70)
        self.kappas = self.rhos*self.rs/self.d.get_Sigmacr()
        self.thetas = self.rs/self.d.angular_diameter_distance(self.d.z0,self.d.z1)

        
    def gfunc(self,x):
        x = np.atleast_1d(x)
        g = x*.0
        arr = x[x<1]
        g[x<1] = np.log(arr/2) + 1/np.sqrt(1 - arr**2)*np.arccosh(1/arr)
        arr = x[x==1]
        g[x==1] = 1 + np.log(0.5)
        arr = x[x>1]
        g[x>1] = np.log(arr/2) + 1/np.sqrt(arr**2-1)*np.arccos(1/arr)
        return g

    def Ffunc(self,x):
        x = np.atleast_1d(x)
        F = x*.0
        c1 = x<1
        c2 = x==1
        c3 = x>1
        F[c1] = 1/(x[c1]**2-1)*(1 - 1/np.sqrt(1-x[c1]**2)*np.arccosh(1/x[c1]))
        F[c2] = 1/3.
        F[c3] = 1/(x[c3]**2-1)*(1 - 1/np.sqrt(x[c3]**2-1)*np.arccos(1/x[c3]))
        return F

    def hfunc(self,x):
        x = np.atleast_1d(x)
        h = x*.0
        arr = x[x<1]
        h[x<1] = np.log(arr/2)**2 - np.arccosh(1/arr)**2
        arr = x[x>=1]
        h[x>=1] = np.log(arr/2)**2 + np.arccos(1/arr)**2
        return h
    

    def rho(self,r):
        return 1./(r/self.rs)/(1 + r/self.rs)**2*self.rhos

    def M3d(self,r):
        return (np.log(1 + r/self.rs) - r/(r+self.rs))/self.M3d1

    def Sigma(self,r):
        x = r/self.rs
        return 2*self.rhos*self.rs*self.Ffunc(x)

    def M2d(self,r):
        return self.gfunc(r/self.rs)/self.M3d1
    
    def meanSigma(self,x):
        return 4*self.rhos*self.rs*self.gfunc(x)/x**2

    def Phi(self,x):
        return 2*self.kappas*self.thetas**2*hfunc(x)
    
    
    
if __name__=="__main__":
    lens = NFWlens()
    r200 = lens.r200
    m =  lens.M3d(r200)
    print("rs is: ","%.3e" % lens.rs,"Mpc.")
    print("r200 is: ","%.3e" % r200,"Mpc.")
    print("The mass within r200 is:","%.3e" % m,"Msun.")

   
    
