from math import *
import numpy as np
from Distances import Distances
from NFWlens import NFWlens
from Sersic import Sersic
import matplotlib.pyplot as plt
from scipy.optimize import bisect

class lens_stat1D():
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
    Galaxy parameters:
    =======================================================
    - Mstar: the mass of the galaxy
    - Re: effective radius of the galaxy, i.e. the projected 
    radius encircling half of the total luminosity
    - m: Sersic law index, default set to be 4. Normally 
    1<= m <= 4
    =======================================================
    Cosmological parameters:
    =======================================================
    - Om: matter density parameter
    - Or: radiation density parameter
    - Ol: dark energy density parameter
    - H0: Hubble constant
    =======================================================
    - galaxy: whether a Serisic model is applied on top of 
    the NFW profile
    =======================================================
    """
    
    def __init__(self,z0=0,z1=.3,z2=1.5,M200=1e13,Mstar=10**11.5,c=5,Re=3,m=4,Om=.25,Or=8.4e-5,Ol=.75,H0=70,galaxy=True):
        self.z0 = 0
        self.z1 = z1
        self.z2 = z2
        self.d = Distances(z0=z0,z1=z1,z2=z2,Om=Om,Ol=Ol,Or=Or,H0=H0)
        self.lens = NFWlens(z0=z0,z1=z1,z2=z2,M200=M200,c=c,Om=Om,Or=Or,Ol=Ol,H0=H0)
        self.galaxy = Sersic(Mstar=Mstar,Re=Re,m=m)
        self.apply_galaxy=galaxy
        self.Da1 = self.d.angular_diameter_distance(self.z0,z1) #ang distance to the lens
        self.Da2 = self.d.angular_diameter_distance(self.z0,z2) #ang distance to the source
        self.thetas = self.lens.rs/self.Da1 #in radians
        self.Sigmacr = self.d.get_Sigmacr() 
        self.kappas =  self.lens.rhos*self.lens.rs/self.Sigmacr

    def alpha(self,x):
        r = x*self.lens.rs
        alpha_galaxy = self.galaxy.M2d(r)/(pi*r**2)/self.Sigmacr*x*self.apply_galaxy
        #return  4*self.kappas/x*(np.log(x/2)+2/np.sqrt(1-x**2)*np.arctanh(np.sqrt((1-x)/(1+x))))+alpha_galaxy
        return 4*self.kappas/x*self.lens.gfunc(x)+alpha_galaxy

    def kappa(self,x):
        r = x*self.lens.rs
        kappa_galaxy = self.galaxy.Sigma(r)/self.Sigmacr*self.apply_galaxy
        return 2*self.kappas*self.lens.Ffunc(x)+kappa_galaxy

    def gamma(self,x):
        r = x*self.lens.rs
        gamma_galaxy = (self.galaxy.M2d(r)/(pi*r**2)-self.galaxy.Sigma(r))/self.Sigmacr*self.apply_galaxy
        return 2*self.kappas*(2*self.lens.gfunc(x)/x**2-self.lens.Ffunc(x))+gamma_galaxy

    def magt(self,x):
        return 1/(1-self.kappa(x)-self.gamma(x))

    def magr(self,x):
        return 1/(1-self.kappa(x)+self.gamma(x))

    def Sige(self,x):
        x = np.atleast_1d(x)
        Sige = x*0.
        mask1 = x<1
        mask2 = x==0
        mask3 = x>1
        Sige[mask1] = 4/x[mask1]**2*self.lens.rs*self.lens.rhos*(2/np.sqrt(1-x[mask1]**2)*np.arctanh(np.sqrt((1-x[mask1])/(1+x[mask1])))+np.log(x[mask1]/2))
        Sige[mask3] = 4/x[mask3]**2*self.lens.rs*self.lens.rhos*(2/np.sqrt(x[mask3]**2-1)*np.arctan(np.sqrt((x[mask3]-1)/(1+x[mask3])))+np.log(x[mask3]/2))
        Sige[mask2] =  4*self.lens.rhos*self.lens.rs*(1+np.log(1/2))
        return  Sige

    def Sig(self,x):
        x = np.atleast_1d(x)
        Sig = x*0.
        mask1 = x<1
        Sig[mask1] = 2*self.lens.rs*self.lens.rhos/(x[mask1]**2-1)*(1-2/np.sqrt(1-x[mask1]**2)*np.arctanh(np.sqrt((1-x[mask1])/(1+x[mask1]))))
        mask2 = x == 0
        Sig[mask2]  = 2*self.lens.rs*self.lens.rhos/3
        mask3 = x>1
        Sig[mask3] = 2*self.lens.rs*self.lens.rhos/(x[mask3]**2-1)*(1-2/np.sqrt(x[mask3]**2-1)*np.arctan(np.sqrt((x[mask3]-1)/(1+x[mask3]))))
        return Sig

    def kap(self,x):
        return self.Sig(x)/self.Sigmacr

    def gam(self,x):
        return (self.Sige(x)-self.Sig(x))/self.Sigmacr

    def det(self,x):
        return (1-self.kap(x))**2-self.gam(x)**2



    def detA(self,x):
        x = np.atleast_1d(x)
        detA = x*0.
        mask1 = x>0
        detA[mask1] = (1-self.kappa(x[mask1]))**2-self.gamma(x[mask1])**2
        mask2 = x<=0
        detA[mask2] = (1-self.kappa(-x[mask2]))**2-self.gamma(-x[mask2])**2
        return detA


    def beta(self,x):
        x = np.atleast_1d(x)
        beta = x*0.
        mask1=x>0
        beta[mask1] = x[mask1]-self.alpha(x[mask1])
        mask2 = x<0
        beta[mask2] = x[mask2]+self.alpha(-x[mask2])
        return beta
    
if __name__=="__main__":
    stat = lens_stat1D(galaxy=True)

    xmin,xmax = -4,-1
    xval = np.logspace(xmin,xmax,1000)
    xval = np.delete(xval,0)
    thetas = xval*stat.thetas*206265
    alphas  = stat.alpha(xval)
    div = stat.detA(xval)
    plt.figure(figsize=(7,5))
    plt.plot(thetas,div)
    plt.axhline(y=0,ls='--',color='k')
    plt.xlabel(r'$\theta$',fontsize=15)
    plt.ylabel(r'$det \ A$',fontsize=15)
    plt.show()
    ind = np.argmin(div)
    rt1 = bisect(stat.detA,10**xmin,xval[ind],maxiter=500)
    rt2 = bisect(stat.detA,xval[ind],10**xmax,maxiter=500)
    rt_theta1 = rt1*stat.thetas*206265
    rt_theta2 = rt2*stat.thetas*206265
    caus = stat.beta(rt1)*stat.thetas*206265
    
    print("Einstein radius is: ","%.3e" % rt_theta2,"arcsec.")
    print("Radial critical curve is at: ","%.3e" % rt_theta1,"arcsec.")
    print("Caustic is at: ","%.3e" % np.abs(caus),"arcsec.")
    print("Caustic is at: ","%.3e" % np.abs(stat.beta(rt2)*stat.thetas*206265),"arcsec.")
