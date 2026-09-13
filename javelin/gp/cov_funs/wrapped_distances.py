from __future__ import annotations
import numpy as np
from .distances import euclidean, aniso_geo_rad, paniso_geo_rad, geographic as geo_rad

def geo_deg(D,x,y,**kwargs): return geo_rad(D,np.deg2rad(x),np.deg2rad(y),**kwargs)
def aniso_geo_deg(D,x,y,**kwargs): return aniso_geo_rad(D,np.deg2rad(x),np.deg2rad(y),**kwargs)
def partition_aniso_geo_rad(D,x,y,**kwargs): return paniso_geo_rad(D,x,y,**kwargs)
def partition_aniso_geo_deg(D,x,y,**kwargs): return paniso_geo_rad(D,np.deg2rad(x),np.deg2rad(y),**kwargs)
for _f in [euclidean, geo_rad, geo_deg, aniso_geo_rad, aniso_geo_deg, paniso_geo_rad, partition_aniso_geo_rad, partition_aniso_geo_deg]:
    if _f.__doc__ is None:
        _f.__doc__ = "Distance metric."
