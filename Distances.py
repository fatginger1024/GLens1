from math import *
import numpy as np
from scipy.integrate import quad


class Distances():
    
    def __init__(self,z0=0,z1=.3,z2=1.5,Om=.25,Ol=.75,Or=8.4e-5,H0=70):
        
        self.Om = Om
        self.Or = Or
        self.Ol = Ol
        self.z0 = z0
        self.z1 = z1
        self.z2 = z2
        self.H0 = H0
        
        
        
    def comoving_distance(self,z1,z2):
        invE = lambda z: 1./np.sqrt(self.Or*(1+z)**4+self.Om*(1+z)**3+self.Ol)
        res = quad(invE,z1,z2)[0]*3e5/self.H0
        return res

    def angular_diameter_distance(self,z1,z2):
        invE = lambda z: 1/np.sqrt(self.Or*(1+z)**4+self.Om*(1+z)**3+self.Ol)
        res = quad(invE,z1,z2)[0]*3e5/self.H0
        return res/(1+z2)
    
    def DA(self,OR=0.):
        DH = 3e5/self.H0
        Dm1 = self.comoving_distance(self.z0,self.z1)
        Dm2 = self.comoving_distance(self.z0,self.z2)
        return (Dm2*np.sqrt(1+OR*Dm1**2/DH**2)-Dm1*np.sqrt(1+OR*Dm2**2/DH**2))/(1+self.z2)
    
    def get_Sigmacr(self):
        G = 6.67e-11*3.24e-23*1.989e30*1e-6  #Mpc Msun^-1 (km/s)^2
        c = 3e5 #km/s
        Da1 = self.angular_diameter_distance(self.z0,self.z1)
        Da2 = self.angular_diameter_distance(self.z0,self.z2)
        Da12 = self.angular_diameter_distance(self.z1,self.z2)
        return c**2/(4*pi*G) * Da2/(Da1*Da12)
        
        
    
if __name__=="__main__":

    d = Distances()
    Dm1 =  d.comoving_distance(0,.3)
    Dm2 = d.comoving_distance(0,1.5)
    Da1 = d.angular_diameter_distance(0,.3)
    Da2 = d.angular_diameter_distance(0,1.5)
    Da12 = d.angular_diameter_distance(.3,1.5)
    Sigma_cr = d.get_Sigmacr()  #Msun Mpc^-2
    print("The angular diameter distances are found to be: ","{:.2f},{:.2f},{:.2f}".format(Da1,Da2,Da12),"Mpc.")
    print("The critical mass density is found to be: ","%.3e" % Sigma_cr,"Msun/Mpc^2.")


