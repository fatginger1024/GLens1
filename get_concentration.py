from math import *
import numpy as np

def OL(z,OL0=.75,OM0=.25):
    return OL0/(OL0+OM0*(1+z)**3)

def OM(z):
    return 1-OL(z)

def Phi(z):
    return OM(z)**(4/7.)-OL(z)+(1+OM(z)/2)*(1+OL(z)/70)

def D(z,OM0=.25):
    return OM(z)/OM0 * Phi(0)/Phi(z) /(1+z)
    
def sigma(M,z):
        xi = 1/(M/1e10)
        return D(z)*22.26*xi**.292/(1+1.53*xi**.275+3.36*xi**.198)
    
def nu(M,z,delta_sc=1.686):
    return delta_sc/sigma(M,z)

def concentration(M,z):
    def c0(z):
        return 3.395*(1+z)**(-.215)
    def beta(z):
        return .307*(1+z)**(.540)
    def gamma1(z):
        return .628*(1+z)**(-.047)
    def gamma2(z):
        return .317*(1+z)**(-.893)
    def nu0(z):
        a = 1/(1+z)
        return (4.135-.564/a-.210/a**2+.0557/a**3-.00348/a**4)/D(z)
    return c0(z)*(nu(M,z)/nu0(z))**(-gamma1(z))*(1+(nu(M,z)/nu0(z))**(1/beta(z)))**(-beta(z)*(gamma1(z)+gamma2(z)))