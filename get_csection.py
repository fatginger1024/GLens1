from math import *
import numpy as np
from scipy.optimize import bisect,newton
from onedimLens import lens_stat1D
import matplotlib.pyplot as plt

class cross_section():
    
    def __init__(self,z0=0,z1=.3,z2=1.5,M200=1e13,Mstar=10**11.5,c=5,Re=3,m=4,
                 Om=.25,Or=8.4e-5,Ol=.75,H0=70,galaxy=True,show=False):
        
        self.stat = lens_stat1D(z0=z0,z1=z1,z2=z2,M200=M200,Mstar=Mstar,c=c,Re=Re,m=m,Om=Om,Or=Or,Ol=Ol,H0=H0,galaxy=galaxy)
        #get critical curves
        self.z1 = z1
        self.z2 = z2
        self.rt = self.get_critical_curves()
        self.rt1 = self.rt[0]
        self.rt2 = self.rt[1]
        #convert critical curves to arcsecs
        self.rt_theta1 = self.rt1*self.stat.thetas*206265
        self.rt_theta2 = self.rt2*self.stat.thetas*206265
        #get caustics
        self.caus = self.get_caustics(self.rt)
        self.caus1 = self.caus[0]
        self.caus2 = self.caus[1]
        self.get_betas(show=show)
        
        
    def get_critical_curves(self,xmin=-4,xmax=0):
        
        xval = np.logspace(xmin,xmax,10000)
        #avoid division by zero
        xval = np.delete(xval,0)
        #convert to arcsec
        thetas = xval*self.stat.thetas*206265
        #get deflection angle
        alphas = self.stat.alpha(xval)
        #get determinant of the Jacobian (A)
        div = self.stat.detA(xval)
        #get minimum of the determinant 
        ind = np.argmin(div)
        try:
            #the location of the tangential critical curve
            rt1 = bisect(self.stat.detA,10**xmin,xval[ind],maxiter=500)
            #the location of the radial crritical curve
            rt2 = bisect(self.stat.detA,xval[ind],10**xmax,maxiter=500)
        except ValueError:
            print(self.z1,self.z2)
            print(self.stat.detA(10**xmin),self.stat.detA(10**xmax),self.stat.detA(xval[ind]))
        return np.concatenate((rt1,rt2),axis=None)

    def get_caustics(self,rt):
        caus = np.abs(self.stat.beta(rt))*self.stat.thetas*206265
        
        return caus
    
    def get_betas(self,show=True):
        
        #specifying the ranges of theta, rt2 is the location of the tangential critical curve
        xmin,xmax = -12,np.log10(self.rt2*5)
        xval = np.logspace(xmin,xmax,100000)
        xval = np.concatenate((-xval[::-1],xval),axis=None)
        thetas = xval*self.stat.thetas*206265
        betas = self.stat.beta(xval)*self.stat.thetas*206265
        #check if singularity is in range(0,2*rt2)
        ind = np.argmin(betas[xval>0])
        xval_min = xval[xval>0][ind]
        self.xval_min = xval_min
        #print("Check if singularity is in range (0, 2*rt2): ",(xval_min>0)&(xval_min<self.rt2))
        if show == True:
            plt.figure(figsize=(7,5))
            plt.plot(thetas,betas)
            plt.axhline(y=0,ls='--',color='k')
            plt.xlabel(r'$\theta$',fontsize=15)
            plt.ylabel(r'$\beta$',fontsize=15)
            plt.show()
        
        
    def get_loc_calm(self, ind):
        b = self._betas[ind]
        fbeta = lambda x:self.stat.beta(x)*self.stat.thetas*206265-b
        r3 = bisect(fbeta,self.xval_min,self.rt2*5)*self.stat.thetas*206265
        mag3 = np.abs(1/self.stat.detA(r3/(self.stat.thetas*206265)))
        self.mag3[ind] = mag3
        return mag3

    def get_loc_getm(self, ind):
        if ind in self.mag3.keys():
            return self.mag3[ind]
        else:
            return self.get_loc_calm(ind)

    def run_get_loc(self, beg, end, mag_th=1.):
        if end-beg == 1:
            return end, self.mag3[end]
        mid = int((end+beg)//2)
        mag3_beg = self.get_loc_getm(beg)
        mag3_end = self.get_loc_getm(end)
        mag3_mid = self.get_loc_getm(mid)
        #print(beg, end, mid, mag3_beg, mag3_end, mag3_mid)
        if (mag3_end > mag_th) and (mag3_mid < mag_th):
            beg = mid
        elif (mag3_end > mag_th) and (mag3_mid > mag_th):
            end = mid
        return self.run_get_loc(beg, end, mag_th=mag_th)

    def get_loc(self):
        self._betas = np.linspace(-self.caus1*.99,0,10000)
        self.mag3 = {}
        ind, mag3 = self.run_get_loc(beg=0,end=len(self._betas)-1)
        return ind,self._betas[ind], mag3
 
        


if __name__=="__main__":
    
    csection = cross_section(show=True)
    theta1 = csection.rt_theta1
    theta2 = csection.rt_theta2
    theta1_caus = csection.caus1
    theta2_caus = csection.caus2
    beta_caus = csection.get_loc()[1]
    mag3 = csection.get_loc()[2]
    tot_cross = pi*theta1_caus**2
    cross_sec = pi*beta_caus**2
    print("The tangential critical curve is at: ",np.round(theta2,3),"arcsec.")
    print("The radial critical curve is at: ",np.round(theta1,3),"arcsec.")
    print("The tangential caustic is at: ",np.round(theta2_caus,3),"arcsec.")
    print("The radial caustic is at: ",np.round(theta1_caus,3),"arcsec.")
    print("Total cross section is: ","%.3e"%tot_cross,"arcsec^2.")
    print("Strong lensing cross section is: ","%.3e"%cross_sec,"arcsec^2.")
    print(csection.xval_min)
    print(beta_caus)
    print(mag3)
    
    
    

    
