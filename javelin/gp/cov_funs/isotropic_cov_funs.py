from __future__ import annotations
import numpy as np
from scipy.special import kv, gamma

def _sl(C,cmin,cmax): return slice(cmin, C.shape[1] if cmax==-1 else int(cmax))
def _apply(C,fn,cmin=0,cmax=-1,symm=False):
    s=_sl(C,cmin,cmax); C[:,s]=fn(np.asarray(C[:,s],float));
    if symm: symmetrize(C,cmin,cmax)
def imul(C,a,cmin=0,cmax=-1,symm=False): C[:,_sl(C,cmin,cmax)]*=a
def symmetrize(C,cmin=0,cmax=-1):
    n=C.shape[0]; end=n if cmax==-1 else min(int(cmax),n)
    for j in range(int(cmin),end): C[j,:j]=np.asarray(C[:j,j]).ravel()
def gaussian(C,cmin=0,cmax=-1,symm=False): _apply(C,lambda z:np.exp(-z*z),cmin,cmax,symm)
def exponential(C,cmin=0,cmax=-1,symm=False): _apply(C,lambda z:np.exp(-np.abs(z)),cmin,cmax,symm)
def pow_exp(C,pow,cmin=0,cmax=-1,symm=False): _apply(C,lambda z:np.exp(-np.abs(z)**pow),cmin,cmax,symm)
def pareto_exp(C,alpha,cmin=0,cmax=-1,symm=False): _apply(C,lambda z:np.where(np.abs(z)<1,np.exp(-np.abs(z)),1/(np.e*np.abs(z)**alpha)),cmin,cmax,symm)
def pow_tail(C,beta,cmin=0,cmax=-1,symm=False):
    pivot=2.; coef=np.exp(pivot**beta-pivot); _apply(C,lambda z:np.where(np.abs(z)<pivot,np.exp(-np.abs(z)),coef*np.exp(-np.abs(z)**beta)),cmin,cmax,symm)
def kepler_exp(C,tcut,cmin=0,cmax=-1,symm=False):
    p=1.5; coef=(1-np.exp(-abs(tcut)))/tcut**p; _apply(C,lambda z:np.where(z>tcut,np.exp(-np.abs(z)),1-coef*z**p),cmin,cmax,symm)
def wkepler_exp(C,tcut,cmin=0,cmax=-1,symm=False):
    p=1.2; coef=(1-np.exp(-abs(tcut)))/tcut**p; _apply(C,lambda z:np.where(z>tcut,np.exp(-np.abs(z)),1-coef*z**p),cmin,cmax,symm)
def sphere(C,cmin=0,cmax=-1,symm=False): _apply(C,lambda z:np.where(z<1,1-1.5*z+0.5*z**3,0.),cmin,cmax,symm)
def quadratic(C,phi,cmin=0,cmax=-1,symm=False): _apply(C,lambda z:1-z*z/(1+phi*z*z),cmin,cmax,symm)
def matern(C,diff_degree,cmin=0,cmax=-1,symm=False):
    nu=float(diff_degree); pref=2**(1-nu)/gamma(nu)
    def f(z):
        r=np.abs(z)*np.sqrt(2*nu); out=np.ones_like(r); m=r>0; out[m]=pref*r[m]**nu*kv(nu,r[m]); return out
    _apply(C,f,cmin,cmax,symm)
def brownian(C,x,y,cmin=0,cmax=-1,symm=False):
    x=np.asarray(x,float); y=np.asarray(y,float); s=_sl(C,cmin,cmax)
    for j in range(s.start,s.stop): C[:,j]=0.5*(np.linalg.norm(x,axis=1)+np.linalg.norm(y[j])-np.linalg.norm(x-y[j],axis=1))
    if symm:symmetrize(C,cmin,cmax)
def frac_brownian(C,x,y,h,cmin=0,cmax=-1,symm=False):
    x=np.asarray(x,float); y=np.asarray(y,float); s=_sl(C,cmin,cmax)
    for j in range(s.start,s.stop): C[:,j]=0.5*((np.sum(x*x,1)**h)+(np.sum(y[j]*y[j])**h)-(np.sum((x-y[j])**2,1)**h))
    if symm:symmetrize(C,cmin,cmax)
def nsmatrn(C,ddx,ddy,hx,hy,nmax,cmin=0,cmax=-1,symm=False):
    # conservative Python implementation of the non-stationary Matérn
    z=np.asarray(C,float); ddx=np.asarray(ddx); ddy=np.asarray(ddy); hx=np.asarray(hx); hy=np.asarray(hy)
    for j in range(cmin, z.shape[1] if cmax==-1 else cmax):
        for i in range(z.shape[0] if not symm else j+1):
            nu=0.5*(ddx[i]+ddy[j]); r=abs(z[i,j])*np.sqrt(max(2*nu,1e-15)); z[i,j]=hx[i]*hy[j]*(1.0 if r==0 else 2**(1-nu)/gamma(nu)*r**nu*kv(nu,r))
    if symm:symmetrize(C,cmin,cmax)
