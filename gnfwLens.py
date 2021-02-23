from math import *
import numpy as np
from scipy.special import hyp2f1
from scipy.integrate import quad,romberg,fixed_quad,quad_vec
from Distances import Distances
from Sersic import Sersic
from onedimLens import lens_stat1D
import matplotlib.pyplot as plt
from scipy.optimize import bisect


class lens_stat_gnfw():
    def __init__(self,z0=0,z1=.3,z2=1.5,M200=1e13,Mstar=10**11.5,c=5,Re=3,m=4,Om=.25,Or=8.4e-5,Ol=.75,H0=70,ratio=1,cratio=1,alpha=1,source_mag=25.,galaxy=True,get_rho=False):
        self.statt = lens_stat1D(z0=z0,z1=z1,z2=z2,M200=M200,Mstar=Mstar*ratio,
                                c=c*cratio,Re=Re,m=m,Om=Om,Or=Or,Ol=Ol,H0=H0,galaxy=galaxy)
        self.alpha = alpha # power law index of the gnfw profile
        self.z0 = 0
        self.z1 = z1
        self.z2 = z2
        self.mag_unlensed = source_mag
        self.H0  =  H0/3.086e19 #convert to s^-1
        self.h = H0/100
        self.d = Distances(z0=z0,z1=z1,z2=z2,Om=Om,Ol=Ol,Or=Or,H0=H0)
        self.Ez = np.sqrt(Or*(1+z1)**4+Om*(1+z1)**3+Ol)
        self.Hz = self.H0*self.Ez
        G  = 6.67e-11 # m^3  kg^-1 s^-2
        self.c = c*cratio #concentration parameter
        self.rhoz = 3*self.Hz**2/(8*pi*G)/1.989e30/(3.24e-23)**3  # M_sun * Mpc^(-3)
        self.rhos = 200/3*self.rhoz*(3-alpha)*c**(alpha)/hyp2f1(3-alpha,3-alpha,4-alpha,-c)
        self.M200 = M200*ratio
        self.r200 = (self.M200/(4/3*pi*200*self.rhoz))**(1/3) #Mpc
        self.rs = self.r200/c #Mpc
        self.galaxy = Sersic(Mstar=Mstar,Re=Re,m=m)
        self.apply_galaxy=galaxy
        self.Da1 = self.d.angular_diameter_distance(self.z0,z1) #ang distance to the lens
        self.Da2 = self.d.angular_diameter_distance(self.z0,z2) #ang distance to the source
        self.thetas = self.rs/self.Da1 #in radians
        self.Sigmacr = self.d.get_Sigmacr() 
        self.kappas =  self.rhos*self.rs/self.Sigmacr
        self.b = 4*self.rhos*self.rs/self.Sigmacr
        if get_rho == False:
            self.xval_min,self.caus1,self.beta_mag = self.get_xval_min()
        
    def rho(self,r):
        alpha = self.alpha
        return self.rhos/((r/self.rs)**(alpha)*(1+(r/self.rs))**(3-alpha))
    
    def Sigma(self,r):
        x = r/self.rs
        return 2*self.rhos*self.rs*x**(1-self.alpha)*quad_vec(lambda t:np.sin(t)*(np.sin(t)+x)**(self.alpha-3),0,np.pi/2)[0]
    
    def M2d(self,r):
        return 2*np.pi*quad_vec(lambda x:x*self.Sigma(x),0,r)[0]
    
    def F(self,x):
        x = np.atleast_1d(x)
        F = x*0.
        mask1 = x<1
        F[mask1] = 1./np.sqrt(1-x[mask1]**2)*np.arctanh(np.sqrt(1-x[mask1]**2))
        mask2 = x>1
        F[mask2] = 1./np.sqrt(x[mask2]**2-1)*np.arctan(np.sqrt(x[mask2]**2-1))
        mask3 = x==1
        F[mask3] = 1
        return F
    
    def dF(self,x):
        return (1-x**2*self.F(x))/(x*(x**2-1))
        
    def f(self,x):
        alpha = self.alpha
        def f0(x):
            x = np.atleast_1d(x)
            #f = 1./2*(1+x**2*(2-3*self.F(x)))/(x**2-1)**2
            f = x*0.
            mask1 = x<1
            f[mask1] = x[mask1]/(2*(1-x[mask1]**2)**2)*(1+2*x[mask1]**2-6*x[mask1]**2/np.sqrt(1-x[mask1]**2)*np.arctanh(np.sqrt((1-x[mask1])/(1+x[mask1]))))
            mask2 = x>1
            f[mask2] = x[mask2]/(2*(x[mask2]**2-1)**2)*(1+2*x[mask2]**2-6*x[mask2]**2/np.sqrt(x[mask2]**2-1)*np.arctan(np.sqrt((x[mask2]-1)/(1+x[mask2]))))
            return f
        
        def f1(x):
            x = np.atleast_1d(x)
            #f = (1-self.F(x))/(x**2-1)
            f = x*0.
            mask1 = x<1
            f[mask1] = 1./(1-x[mask1]**2)*(-1+2/np.sqrt(1-x[mask1]**2)*np.arctanh(np.sqrt((1-x[mask1])/(1+x[mask1]))))
            mask2 = x>1
            f[mask2] = 1./(x[mask2]**2-1)*(1-2/np.sqrt(x[mask2]**2-1)*np.arctan(np.sqrt((x[mask2]-1)/(x[mask2]+1))))
            return f
        
        def f2(x):
            x = np.atleast_1d(x)
            #f = 1./2*(pi/x-2*self.F(x))
            f = x*0.
            mask1 = x<1
            f[mask1] = 1./x[mask1]*(pi/2-x[mask1]/np.sqrt(1-x[mask1]**2)*np.arctanh(1-x[mask1]**2))
            mask2 = x>1
            f[mask2] = 1./x[mask2]*(pi/2-x[mask2]/np.sqrt(x[mask2]**2-1)*np.arctan(x[mask2]**2-1))
            return f
        """    
        if alpha == 0:
            return f0(x)
        elif  alpha  == 1:
            return f1(x)
        elif alpha == 2:
            return f2(x)
        
        else:
        """
        x = np.atleast_1d(x)
        return x**(1-alpha)*((1+x)**(alpha-3)+(3-alpha)*quad_vec(lambda y:(y+x)**(alpha-4)*(1-np.sqrt(1-y**2)),0,1)[0])
       
        #return np.array([x[i]**(1-alpha)*((1+x[i])**(alpha-3)+(3-alpha)*quad(lambda y:(y+x[i])**(alpha-4)*(1-np.sqrt(1-y**2)),0,1)[0]) for i in range(len(x))])

    def g(self,x):
        alpha = self.alpha
        def g0(x):
            x = np.atleast_1d(x)
            #g  =  1./(2*x)*(2*np.log(x/2)+(x**2+(2-3*x**2)*self.F(x))/(1-x**2))
            g = x*0.
            mask1 = x<1
            g[mask1] = (2-3*x[mask1]**2)/(x[mask1]*(1-x[mask1]**2)**(3/2))*np.arctanh(np.sqrt((1-x[mask1])/(1+x[mask1]))) + x[mask1]/(2*(1-x[mask1]**2))+np.log(x[mask1]/2)/x[mask1]
            mask2 = x>1
            g[mask2] = -(2-3*x[mask2]**2)/(x[mask2]*(x[mask2]**2-1)**(3/2))*np.arctan(np.sqrt((x[mask2]-1)/(1+x[mask2]))) - x[mask2]/(2*(x[mask2]**2-1))+np.log(x[mask2]/2)/x[mask2]
            return g
        
        def g1(x):
            x = np.atleast_1d(x)
            #g = (np.log(x/2)+self.F(x))/x
            g = x*0.
            mask1 = x<1
            g[mask1] = 2/(x[mask1]*np.sqrt(1-x[mask1]**2))*np.arctanh(np.sqrt((1-x[mask1])/(1+x[mask1])))+np.log(x[mask1]/2)/x[mask1]
            mask2 = x>1
            g[mask2] = 2/(x[mask2]*np.sqrt(x[mask2]**2-1))*np.arctan(np.sqrt((x[mask2]-1)/(1+x[mask2])))+np.log(x[mask2]/2)/x[mask2]
            return g
        
        def g2(x):
            x = np.atleast_1d(x)
            #g = pi/2+np.log(x/2)/x+(1-x**2)/x*self.F(x)
            g = x*0.
            mask1 = x<1
            g[mask1] = np.sqrt(1-x[mask1]**2)/x[mask1]*np.arctanh(np.sqrt(1-x[mask1]**2))+np.log(x[mask1]/2)/x[mask1]+pi/2
            mask2 = x>1
            g[mask2] = -np.sqrt(x[mask2]**2-1)/x[mask2]*np.arctan(np.sqrt(x[mask2]**2-1))+np.log(x[mask2]/2)/x[mask2]+pi/2
            return g
        
        def hypF(a,b,c,z):
            return hyp2f1(a,b,c,z)
        """
        if alpha == 0:
            return g0(x)
        elif alpha  == 1:
            return g1(x)
        elif alpha == 2:
            return g2(x)
        
        else:
        """
        x = np.atleast_1d(x)
        return x**(2-alpha)*(hypF(3-alpha,3-alpha,4-alpha,-x)/(3-alpha)+quad_vec(lambda y:(y+x)**(alpha-3)*(1-np.sqrt(1-y**2))/y,0,1)[0])
        #return np.array([x[i]**(2-alpha)*(hypF(3-alpha,3-alpha,4-alpha,-x[i])/(3-alpha)+quad(lambda y:(y+x[i])**(alpha-3)*(1-np.sqrt(1-y**2))/y,0,1)[0]) for i in range(len(x))])

        
    def alpha_(self,x):
        r = x*self.rs
        alpha_galaxy = self.galaxy.M2d(r)/(pi*r**2)/self.Sigmacr*x*self.apply_galaxy
        return self.b*self.g(x)+alpha_galaxy
    
    def kappa(self,x):
        r = x*self.rs
        kappa_galaxy = self.galaxy.Sigma(r)/self.Sigmacr*self.apply_galaxy
        return self.b*self.f(x)/2+kappa_galaxy
    
    def gamma(self,x):
        r = x*self.rs
        gamma_galaxy = (self.galaxy.M2d(r)/(pi*r**2)-self.galaxy.Sigma(r))/self.Sigmacr*self.apply_galaxy
        return self.b*self.g(x)/x-self.b*self.f(x)/2+gamma_galaxy
    
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
        beta[mask1] = x[mask1]-self.alpha_(x[mask1])
        mask2 = x<0
        beta[mask2] = x[mask2]+self.alpha_(-x[mask2])
        return beta
    
    def get_xval_min(self,tol=26.3):
        #specifying the ranges of theta, rt2 is the location of the tangential critical curve
        xmin,xmax = -1,1
        xval = np.linspace(xmin,xmax,10000)
        #xval = np.concatenate((-xval[::-1],xval),axis=None)
        thetas = xval*self.thetas*206265
        betas = self.beta(xval)*self.thetas*206265
        #check if singularity is in range(0,2*rt2)
        ind = np.argmin(betas[xval>0])
        xval_min = xval[xval>0][ind]
        caus1 = betas[xval>0][ind]
        #xval = np.linspace(xval_min*1.1,10,10000)
        #thetas = xval*self.thetas*206265
        #betas = self.beta(xval)*self.thetas*206265
        msk = (xval>xval_min*1.1)*(betas<0)
        mag = np.abs(1/self.detA(xval[msk]))
        #mag = np.concatenate(np.abs([1/self.stat.detA(xval[msk][i]) for i in range(len(xval[msk]))]),axis=None)
        app_m = self.get_lensed_mag(mag)
        mg_msk = app_m<tol
        #print(xval[msk][mg_msk].min(),xval[msk][mg_msk].max())
        """
        plt.plot(betas[msk][mg_msk],app_m[mg_msk])
        plt.axhline(y=0,ls='--',color='k')
        plt.xlabel(r'$\beta$',fontsize=15)
        plt.ylabel(r'$apparent \ magnitude$',fontsize=15)
        plt.show()
        """
        beta_mag = betas[msk][mg_msk][0]
        
        return xval_min,caus1,beta_mag
    
    def get_lensed_mag(self,m):
        return self.mag_unlensed - 2.5*np.log10(m)
    
  
if __name__=="__main__":
    stat = lens_stat_gnfw()
    xval_min = stat.xval_min
    theta1_caus = stat.caus1
    beta_caus = stat.beta_mag
    tot_cross = pi*theta1_caus**2
    cross_sec = pi*beta_caus**2
    print("xval_min,beta_caus,theta_caus:",xval_min,beta_caus,theta1_caus)
    print("Total cross section is: ","%.3e"%tot_cross,"arcsec^2.")
    print("Strong lensing cross section is: ","%.3e"%cross_sec,"arcsec^2.")
    xmin,xmax = -1,1
    xval = np.linspace(xmin,xmax,10000)
    xval = np.delete(xval,0)
    thetas = xval*stat.thetas*206265
    betas  = stat.beta(xval)*stat.thetas*206265
    """
    plt.figure(figsize=(7,5))
    plt.plot(thetas,betas)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\beta$")
    """