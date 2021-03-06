
from math import *
import numpy as np
from scipy.optimize import bisect,newton
from onedimLens import lens_stat1D
import matplotlib.pyplot as plt

class cross_section():
    
    def __init__(self,z0=0,z1=.3,z2=1.5,M200=1e13,Mstar=10**11.5,c=5,Re=3,m=4,ratio=1,
                 Om=.25,Or=8.4e-5,Ol=.75,H0=70,galaxy=True,source_mag=25.,show=False):
        
        self.stat = lens_stat1D(z0=z0,z1=z1,z2=z2,M200=M200,Mstar=Mstar*ratio,
                                c=c,Re=Re,m=m,Om=Om,Or=Or,Ol=Ol,H0=H0,galaxy=galaxy)
        #get critical curves
        self.z1 = z1
        self.z2 = z2
        self.show = show
        self.get_betas()
        # get source unlensed apparent magnitudes
        self.mag_unlensed = source_mag
        
    
        
    def get_betas(self):
     
        #specifying the ranges of theta, rt2 is the location of the tangential critical curve
        xmin,xmax = 1e-8,1
        xval = np.linspace(xmin,xmax,100000)
        xval = np.concatenate((-xval[::-1],xval),axis=None)
        thetas = xval*self.stat.thetas*206265
        betas = self.stat.beta(xval)*self.stat.thetas*206265
        #check if singularity is in range(0,2*rt2)
        ind = np.argmin(betas[xval>0])
        xval_min = xval[xval>0][ind]
        self.xval_min = xval_min
        self.caus1 = betas[xval>0][ind]
        #print("Check if singularity is in range (0, 2*rt2): ",(xval_min>0)&(xval_min<self.rt2))
        if self.show == True:
            plt.figure(figsize=(7,5))
            plt.plot(thetas,betas)
            plt.axhline(y=0,ls='--',color='k')
            plt.xlabel(r'$\theta$',fontsize=15)
            plt.ylabel(r'$\beta$',fontsize=15)
            plt.show()


            
        
    def get_beta_crit(self,tol=26.3):
        
        xval = np.linspace(self.xval_min*1.1,1,10000)
        thetas = xval*self.stat.thetas*206265
        betas = self.stat.beta(xval)*self.stat.thetas*206265
        #plt.plot(thetas,betas)
        #plt.show()
        msk = betas<0
        mag = np.concatenate(np.abs([1/self.stat.detA(xval[msk][i]) for i in range(len(xval[msk]))]),axis=None)
        app_m = self.get_lensed_mag(mag)
        mg_msk = app_m<tol
        beta_mag = betas[msk][mg_msk][0]
        #print(xval[msk][mg_msk].min(),xval[msk][mg_msk].max())
        if self.show == True:
            plt.figure(figsize=(7,5))
            plt.plot(betas[msk][mg_msk],app_m[mg_msk])
            plt.axhline(y=0,ls='--',color='k')
            plt.xlabel(r'$\beta$',fontsize=15)
            plt.ylabel(r'$apparent \ magnitude$',fontsize=15)
            plt.show()
            
        return beta_mag
        
        
 
    def get_lensed_mag(self,m):
        return self.mag_unlensed - 2.5*np.log10(m)
    
    
        
       
if __name__=="__main__":
    
    csection = cross_section(show=True)
    print(csection.show)
    theta1_caus = csection.caus1
    beta_caus = csection.get_beta_crit()
    #mag3 = csection.get_loc()[2]
    tot_cross = pi*theta1_caus**2
    cross_sec = pi*beta_caus**2
    print("Total cross section is: ","%.3e"%tot_cross,"arcsec^2.")
    print("Strong lensing cross section is: ","%.3e"%cross_sec,"arcsec^2.")
    print("xval_min,beta_caus,theta_caus:",csection.xval_min,beta_caus,theta1_caus)
    
    
    

    
