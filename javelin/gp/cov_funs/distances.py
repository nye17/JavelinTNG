from __future__ import annotations
import numpy as np

def _bounds(C,cmin,cmax):
    return cmin, (C.shape[1] if cmax==-1 else int(cmax))

def euclidean(D,x,y,cmin=0,cmax=-1,symm=False):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    cmin,cmax=_bounds(D,cmin,cmax)
    for j in range(cmin,cmax):
        imax=j if symm else x.shape[0]
        if symm: D[j,j]=0.0
        if imax: D[:imax,j]=np.linalg.norm(x[:imax]-y[j],axis=1)

def geographic(D,x,y,cmin=0,cmax=-1,symm=False):
    x=np.asarray(x,float); y=np.asarray(y,float); cmin,cmax=_bounds(D,cmin,cmax)
    for j in range(cmin,cmax):
        imax=j if symm else x.shape[0]
        if symm: D[j,j]=0.0
        if imax:
            lat1=x[:imax,0]; lon1=x[:imax,1]; lat2=y[j,0]; lon2=y[j,1]
            cs=np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)
            D[:imax,j]=np.arccos(np.clip(cs,-1,1))

def aniso_geo_rad(D,x,y,inc=0.0,ecc=0.0,cmin=0,cmax=-1,symm=False):
    # local tangent-plane approximation retained for compatibility
    xx=np.asarray(x,float); yy=np.asarray(y,float); cmin,cmax=_bounds(D,cmin,cmax)
    ci,si=np.cos(inc),np.sin(inc); q=max(np.sqrt(max(1-ecc*ecc,1e-15)),1e-8)
    for j in range(cmin,cmax):
        imax=j if symm else xx.shape[0]
        if symm:D[j,j]=0
        if imax:
            dx=(xx[:imax,1]-yy[j,1])*np.cos(0.5*(xx[:imax,0]+yy[j,0])); dy=xx[:imax,0]-yy[j,0]
            u=ci*dx+si*dy; v=-si*dx+ci*dy
            D[:imax,j]=np.sqrt(u*u+(v/q)**2)

def paniso_geo_rad(D,x,y,ctrs,scals,amps=None,cmin=0,cmax=-1,symm=False):
    return geographic(D,x,y,cmin,cmax,symm)
