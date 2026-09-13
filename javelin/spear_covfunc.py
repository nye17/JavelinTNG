"""Pure Python implementation of the historical SPEAR/JAVELIN covariance kernel."""
from __future__ import annotations
import numpy as np


def expcov(djd,tau): return np.exp(-abs(djd)/tau)
def getcmat_delta(id1,id2,jd1,jd2,tau,tspike1,scale1,tspike2,scale2): return abs(scale1*scale2)*np.exp(-abs(jd1-jd2-tspike1+tspike2)/tau)

def getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2):
    if id1==1 and id2>=2:
        tlow=jd2-jd1-slag2-0.5*swid2; thig=jd2-jd1-slag2+0.5*swid2; em=abs(scale2); w=swid2
    elif id2==1 and id1>=2:
        tlow=jd1-jd2-slag1-0.5*swid1; thig=jd1-jd2-slag1+0.5*swid1; em=abs(scale1); w=swid1
    elif id1>=2 and id2>=2 and swid1<=0.01:
        tlow=jd2-(jd1-slag1)-slag2-0.5*swid2; thig=jd2-(jd1-slag1)-slag2+0.5*swid2; em=abs(scale2*scale1); w=swid2
    else:
        tlow=jd1-(jd2-slag2)-slag1-0.5*swid1; thig=jd1-(jd2-slag2)-slag1+0.5*swid1; em=abs(scale2*scale1); w=swid1
    if w<=0: return getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
    if thig<=0: v=np.exp(thig/tau)-np.exp(tlow/tau)
    elif tlow>=0: v=np.exp(-tlow/tau)-np.exp(-thig/tau)
    else: v=2-np.exp(tlow/tau)-np.exp(-thig/tau)
    return tau*(em/w)*v

def getcmat_lauto(id,jd1,jd2,tau,slag,swid,scale):
    w=swid; tmid=jd1-jd2; tlow=tmid-w; thig=tmid+w
    if w<=0: return scale*scale*np.exp(-abs(tmid)/tau)
    if thig<=0 or tlow>=0:
        v=np.exp(-abs(tmid)/tau)*(np.exp(0.5*w/tau)-np.exp(-0.5*w/tau))**2
    else:
        v=-2*np.exp(-abs(tmid)/tau)+np.exp(-w/tau)*(np.exp(-tmid/tau)+np.exp(tmid/tau))
        v += 2*(w-tmid)/tau if tmid>=0 else 2*(w+tmid)/tau
    return (scale*tau/w)**2*v

def getcmat_lcross(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2):
    if swid1<=0.01 or swid2<=0.01: return getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    if swid1>=swid2: t1=slag1-.5*swid1; t2=slag1+.5*swid1; t3=slag2-.5*swid2; t4=slag2+.5*swid2; bottleneck=swid2; ti,tj=jd1,jd2
    else: t1=slag2-.5*swid2; t2=slag2+.5*swid2; t3=slag1-.5*swid1; t4=slag1+.5*swid1; bottleneck=swid1; ti,tj=jd2,jd1
    tlow=(ti-tj)-(t2-t3); tm1=(ti-tj)-(t2-t4); tm2=(ti-tj)-(t1-t3); thig=(ti-tj)-(t1-t4)
    if thig<=0 or tlow>=0: v=np.exp(-abs(tlow)/tau)+np.exp(-abs(thig)/tau)-np.exp(-abs(tm1)/tau)-np.exp(-abs(tm2)/tau)
    else:
        v=np.exp(tlow/tau)+np.exp(-thig/tau)-np.exp(-abs(tm1)/tau)-np.exp(-abs(tm2)/tau)
        if tm2<=0: v+=2*thig/tau
        elif tm1<=0: v+=2*bottleneck/tau
        elif tlow<0: v-=2*tlow/tau
    return tau*tau*scale1*scale2/(swid1*swid2)*v

def covmatij(id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2):
    if min(id1,id2)<=0: return -sigma*sigma
    if id1==id2:
        v=getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2) if id1==1 or swid1<=.01 else getcmat_lauto(id1,jd1,jd2,tau,slag1,swid1,scale1)
    elif min(id1,id2)==1:
        v=getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2) if max(swid1,swid2)<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    elif swid1<=.01 and swid2<=.01: v=getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
    elif swid1<=.01 or swid2<=.01: v=getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    else: v=getcmat_lcross(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    return sigma*sigma*v

def covmatpmapij(id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2,scale_hidden):
    if id1==id2==1: v=getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2)
    elif id1==id2==2:
        v=getcmat_delta(id1,id2,jd1,jd2,tau,0,scale_hidden,0,scale_hidden)
        v+= getcmat_delta(id1,id2,jd1,jd2,tau,0,scale_hidden,slag2,scale2) if swid2<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,0,0,scale_hidden,slag2,swid2,scale2)
        v+= getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,0,scale_hidden) if swid1<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,0,0,scale_hidden)
        v+= getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2) if swid1<=.01 else getcmat_lauto(id1,jd1,jd2,tau,slag1,swid1,scale1)
    else:
        v=getcmat_delta(id1,id2,jd1,jd2,tau,0,1,0,scale_hidden)
        v+=getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2) if max(swid1,swid2)<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    return sigma*sigma*v

def covmatdpmapij(id1,id2,jd1,jd2,sigma,tau,slag1,swid1,scale1,slag2,swid2,scale2,slag3,swid3,scale3):
    if id1==id2==1: v=getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag1,scale1)
    elif id1==id2==2:
        def auto(s,w,a): return getcmat_delta(id1,id2,jd1,jd2,tau,s,a,s,a) if w<=.01 else getcmat_lauto(id1,jd1,jd2,tau,s,w,a)
        v=auto(slag2,swid2,scale2)+auto(slag3,swid3,scale3)
        if swid2<=.01 and swid3<=.01: v+=getcmat_delta(id1,id2,jd1,jd2,tau,slag2,scale2,slag3,scale3)+getcmat_delta(id1,id2,jd1,jd2,tau,slag3,scale3,slag2,scale2)
        else: v+=getcmat_lc(id1,id2,jd1,jd2,tau,slag2,swid2,scale2,slag3,swid3,scale3)+getcmat_lc(id1,id2,jd1,jd2,tau,slag3,swid3,scale3,slag2,swid2,scale2) if (swid2<=.01 or swid3<=.01) else getcmat_lcross(id1,id2,jd1,jd2,tau,slag2,swid2,scale2,slag3,swid3,scale3)+getcmat_lcross(id1,id2,jd1,jd2,tau,slag3,swid3,scale3,slag2,swid2,scale2)
    else:
        if id1==1:
            v=(getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag2,scale2) if max(swid1,swid2)<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2))
            v+=(getcmat_delta(id1,id2,jd1,jd2,tau,slag1,scale1,slag3,scale3) if max(swid1,swid3)<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag3,swid3,scale3))
        else:
            v=(getcmat_delta(id1,id2,jd1,jd2,tau,slag2,scale2,slag1,scale1) if max(swid1,swid2)<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag2,swid2,scale2,slag1,swid1,scale1))
            v+=(getcmat_delta(id1,id2,jd1,jd2,tau,slag3,scale3,slag1,scale1) if max(swid1,swid3)<=.01 else getcmat_lc(id1,id2,jd1,jd2,tau,slag3,swid3,scale3,slag1,swid1,scale1))
    return sigma*sigma*v



def _getcmat_delta_vec(jd1, jd2, tau, tspike1, scale1, tspike2, scale2):
    """Vectorized delta-transfer covariance for arrays of epochs."""
    return abs(scale1 * scale2) * np.exp(-np.abs(jd1 - jd2 - tspike1 + tspike2) / tau)


def _getcmat_lc_vec(id1, id2, jd1, jd2, tau,
                     slag1, swid1, scale1, slag2, swid2, scale2):
    """Vectorized continuum/line (or narrow-line/broad-line) covariance."""
    if id1 == 1 and id2 >= 2:
        tlow = jd2 - jd1 - slag2 - 0.5 * swid2
        thig = jd2 - jd1 - slag2 + 0.5 * swid2
        em = abs(scale2)
        w = swid2
    elif id2 == 1 and id1 >= 2:
        tlow = jd1 - jd2 - slag1 - 0.5 * swid1
        thig = jd1 - jd2 - slag1 + 0.5 * swid1
        em = abs(scale1)
        w = swid1
    elif id1 >= 2 and id2 >= 2 and swid1 <= 0.01:
        tlow = jd2 - (jd1 - slag1) - slag2 - 0.5 * swid2
        thig = jd2 - (jd1 - slag1) - slag2 + 0.5 * swid2
        em = abs(scale2 * scale1)
        w = swid2
    else:
        tlow = jd1 - (jd2 - slag2) - slag1 - 0.5 * swid1
        thig = jd1 - (jd2 - slag2) - slag1 + 0.5 * swid1
        em = abs(scale2 * scale1)
        w = swid1

    if w <= 0:
        return _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)

    out = np.empty(np.broadcast_shapes(np.shape(tlow), np.shape(thig)), dtype=float)
    neg = thig <= 0
    pos = tlow >= 0
    mid = ~(neg | pos)
    out[neg] = np.exp(thig[neg] / tau) - np.exp(tlow[neg] / tau)
    out[pos] = np.exp(-tlow[pos] / tau) - np.exp(-thig[pos] / tau)
    out[mid] = 2.0 - np.exp(tlow[mid] / tau) - np.exp(-thig[mid] / tau)
    return tau * (em / w) * out


def _getcmat_lauto_vec(jd1, jd2, tau, slag, swid, scale):
    """Vectorized line auto-covariance."""
    w = swid
    tmid = jd1 - jd2
    if w <= 0:
        return scale * scale * np.exp(-np.abs(tmid) / tau)

    tlow = tmid - w
    thig = tmid + w
    outer = (thig <= 0) | (tlow >= 0)
    out = np.empty(np.broadcast_shapes(np.shape(jd1), np.shape(jd2)), dtype=float)
    out[outer] = (
        np.exp(-np.abs(tmid[outer]) / tau)
        * (np.exp(0.5 * w / tau) - np.exp(-0.5 * w / tau)) ** 2
    )
    inner = ~outer
    ti = tmid[inner]
    v = -2.0 * np.exp(-np.abs(ti) / tau)
    v += np.exp(-w / tau) * (np.exp(-ti / tau) + np.exp(ti / tau))
    v += np.where(ti >= 0, 2.0 * (w - ti) / tau, 2.0 * (w + ti) / tau)
    out[inner] = v
    return (scale * tau / w) ** 2 * out


def _getcmat_lcross_vec(id1, id2, jd1, jd2, tau,
                         slag1, swid1, scale1, slag2, swid2, scale2):
    """Vectorized covariance between two finite-width transfer functions."""
    if swid1 <= 0.01 or swid2 <= 0.01:
        return _getcmat_lc_vec(
            id1, id2, jd1, jd2, tau,
            slag1, swid1, scale1, slag2, swid2, scale2,
        )

    if swid1 >= swid2:
        t1 = slag1 - 0.5 * swid1
        t2 = slag1 + 0.5 * swid1
        t3 = slag2 - 0.5 * swid2
        t4 = slag2 + 0.5 * swid2
        bottleneck = swid2
        ti, tj = jd1, jd2
    else:
        t1 = slag2 - 0.5 * swid2
        t2 = slag2 + 0.5 * swid2
        t3 = slag1 - 0.5 * swid1
        t4 = slag1 + 0.5 * swid1
        bottleneck = swid1
        ti, tj = jd2, jd1

    dt = ti - tj
    tlow = dt - (t2 - t3)
    tm1 = dt - (t2 - t4)
    tm2 = dt - (t1 - t3)
    thig = dt - (t1 - t4)

    outer = (thig <= 0) | (tlow >= 0)
    out = np.empty(np.broadcast_shapes(np.shape(jd1), np.shape(jd2)), dtype=float)
    out[outer] = (
        np.exp(-np.abs(tlow[outer]) / tau)
        + np.exp(-np.abs(thig[outer]) / tau)
        - np.exp(-np.abs(tm1[outer]) / tau)
        - np.exp(-np.abs(tm2[outer]) / tau)
    )

    inner = ~outer
    lo = tlow[inner]
    m1 = tm1[inner]
    m2 = tm2[inner]
    hi = thig[inner]
    v = (
        np.exp(lo / tau)
        + np.exp(-hi / tau)
        - np.exp(-np.abs(m1) / tau)
        - np.exp(-np.abs(m2) / tau)
    )
    v += np.where(m2 <= 0, 2.0 * hi / tau,
                  np.where(m1 <= 0, 2.0 * bottleneck / tau,
                           np.where(lo < 0, -2.0 * lo / tau, 0.0)))
    out[inner] = v
    return tau * tau * scale1 * scale2 / (swid1 * swid2) * out


def _covmat_normal_block(id1, id2, jd1, jd2, sigma, tau, p1, p2):
    """Vectorized normal SPEAR covariance for one pair of light-curve IDs."""
    slag1, swid1, scale1 = p1
    slag2, swid2, scale2 = p2
    if min(id1, id2) <= 0:
        return np.full(np.broadcast_shapes(np.shape(jd1), np.shape(jd2)), -sigma * sigma)
    if id1 == id2:
        if id1 == 1 or swid1 <= 0.01:
            v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
        else:
            v = _getcmat_lauto_vec(jd1, jd2, tau, slag1, swid1, scale1)
    elif min(id1, id2) == 1:
        if max(swid1, swid2) <= 0.01:
            v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
        else:
            v = _getcmat_lc_vec(id1, id2, jd1, jd2, tau,
                                 slag1, swid1, scale1, slag2, swid2, scale2)
    elif swid1 <= 0.01 and swid2 <= 0.01:
        v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
    elif swid1 <= 0.01 or swid2 <= 0.01:
        v = _getcmat_lc_vec(id1, id2, jd1, jd2, tau,
                             slag1, swid1, scale1, slag2, swid2, scale2)
    else:
        v = _getcmat_lcross_vec(id1, id2, jd1, jd2, tau,
                                 slag1, swid1, scale1, slag2, swid2, scale2)
    return sigma * sigma * v


def _fill_normal_vectorized(mat, jd1, jd2, id1, id2, sigma, tau,
                            lags, wids, scales, cmin=0, cmax=-1):
    """Fill normal SPEAR covariance by vectorized ID blocks.

    This preserves the historical scalar formulas but removes the nested Python
    pair loop that made large mock-generation matrices prohibitively slow.
    """
    jd1 = np.ravel(jd1)
    jd2 = np.ravel(jd2)
    id1 = np.asarray(id1, dtype=int)
    id2 = np.asarray(id2, dtype=int)
    cmax = mat.shape[1] if cmax == -1 else int(cmax)
    col_range = np.arange(mat.shape[1])
    active_cols = (col_range >= int(cmin)) & (col_range < cmax)

    for a in np.unique(id1):
        rows = np.flatnonzero(id1 == a)
        if rows.size == 0:
            continue
        p1 = (lags[a - 1], wids[a - 1], scales[a - 1])
        t1 = jd1[rows][:, None]
        for b in np.unique(id2[active_cols]):
            cols = np.flatnonzero((id2 == b) & active_cols)
            if cols.size == 0:
                continue
            p2 = (lags[b - 1], wids[b - 1], scales[b - 1])
            t2 = jd2[cols][None, :]
            mat[np.ix_(rows, cols)] = _covmat_normal_block(
                int(a), int(b), t1, t2, sigma, tau, p1, p2
            )



def _covmat_pmap_block(id1, id2, jd1, jd2, sigma, tau, p1, p2, scale_hidden):
    slag1, swid1, scale1 = p1
    slag2, swid2, scale2 = p2
    if id1 == id2 == 1:
        v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
    elif id1 == id2 == 2:
        v = _getcmat_delta_vec(jd1, jd2, tau, 0.0, scale_hidden, 0.0, scale_hidden)
        if swid2 <= .01:
            v += _getcmat_delta_vec(jd1, jd2, tau, 0.0, scale_hidden, slag2, scale2)
        else:
            v += _getcmat_lc_vec(id1,id2,jd1,jd2,tau,0.0,0.0,scale_hidden,slag2,swid2,scale2)
        if swid1 <= .01:
            v += _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, 0.0, scale_hidden)
        else:
            v += _getcmat_lc_vec(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,0.0,0.0,scale_hidden)
        if swid1 <= .01:
            v += _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
        else:
            v += _getcmat_lauto_vec(jd1, jd2, tau, slag1, swid1, scale1)
    else:
        v = _getcmat_delta_vec(jd1, jd2, tau, 0.0, 1.0, 0.0, scale_hidden)
        if max(swid1, swid2) <= .01:
            v += _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
        else:
            v += _getcmat_lc_vec(id1,id2,jd1,jd2,tau,slag1,swid1,scale1,slag2,swid2,scale2)
    return sigma*sigma*v


def _covmat_dpmap_block(id1, id2, jd1, jd2, sigma, tau, lags, wids, scales):
    s1,w1,a1 = lags[0],wids[0],scales[0]
    s2,w2,a2 = lags[1],wids[1],scales[1]
    s3,w3,a3 = lags[2],wids[2],scales[2]
    if id1 == id2 == 1:
        v = _getcmat_delta_vec(jd1,jd2,tau,s1,a1,s1,a1)
    elif id1 == id2 == 2:
        def auto(s,w,a):
            return (_getcmat_delta_vec(jd1,jd2,tau,s,a,s,a) if w<=.01
                    else _getcmat_lauto_vec(jd1,jd2,tau,s,w,a))
        v = auto(s2,w2,a2) + auto(s3,w3,a3)
        if w2<=.01 and w3<=.01:
            v += _getcmat_delta_vec(jd1,jd2,tau,s2,a2,s3,a3)
            v += _getcmat_delta_vec(jd1,jd2,tau,s3,a3,s2,a2)
        elif w2<=.01 or w3<=.01:
            v += _getcmat_lc_vec(id1,id2,jd1,jd2,tau,s2,w2,a2,s3,w3,a3)
            v += _getcmat_lc_vec(id1,id2,jd1,jd2,tau,s3,w3,a3,s2,w2,a2)
        else:
            v += _getcmat_lcross_vec(id1,id2,jd1,jd2,tau,s2,w2,a2,s3,w3,a3)
            v += _getcmat_lcross_vec(id1,id2,jd1,jd2,tau,s3,w3,a3,s2,w2,a2)
    else:
        if id1 == 1:
            term2 = (_getcmat_delta_vec(jd1,jd2,tau,s1,a1,s2,a2) if max(w1,w2)<=.01
                     else _getcmat_lc_vec(id1,id2,jd1,jd2,tau,s1,w1,a1,s2,w2,a2))
            term3 = (_getcmat_delta_vec(jd1,jd2,tau,s1,a1,s3,a3) if max(w1,w3)<=.01
                     else _getcmat_lc_vec(id1,id2,jd1,jd2,tau,s1,w1,a1,s3,w3,a3))
        else:
            term2 = (_getcmat_delta_vec(jd1,jd2,tau,s2,a2,s1,a1) if max(w1,w2)<=.01
                     else _getcmat_lc_vec(id1,id2,jd1,jd2,tau,s2,w2,a2,s1,w1,a1))
            term3 = (_getcmat_delta_vec(jd1,jd2,tau,s3,a3,s1,a1) if max(w1,w3)<=.01
                     else _getcmat_lc_vec(id1,id2,jd1,jd2,tau,s3,w3,a3,s1,w1,a1))
        v = term2 + term3
    return sigma*sigma*v


def _fill_mode_vectorized(mat, jd1, jd2, id1, id2, sigma, tau,
                          lags, wids, scales, cmin=0, cmax=-1, mode='pmap'):
    jd1=np.ravel(jd1); jd2=np.ravel(jd2); id1=np.asarray(id1,int); id2=np.asarray(id2,int)
    cmax=mat.shape[1] if cmax==-1 else int(cmax)
    active=(np.arange(mat.shape[1])>=int(cmin)) & (np.arange(mat.shape[1])<cmax)
    for a in np.unique(id1):
        rows=np.flatnonzero(id1==a); t1=jd1[rows][:,None]
        p1=(lags[a-1],wids[a-1],scales[a-1])
        for b in np.unique(id2[active]):
            cols=np.flatnonzero((id2==b)&active); t2=jd2[cols][None,:]
            p2=(lags[b-1],wids[b-1],scales[b-1])
            if mode=='pmap':
                block=_covmat_pmap_block(int(a),int(b),t1,t2,sigma,tau,p1,p2,scales[-1])
            else:
                block=_covmat_dpmap_block(int(a),int(b),t1,t2,sigma,tau,lags,wids,scales)
            mat[np.ix_(rows,cols)]=block

def _fill(mat,jd1,jd2,id1,id2,sigma,tau,lags,wids,scales,cmin=0,cmax=-1,mode='normal'):
    nx,ny=mat.shape; cmax=ny if cmax==-1 else int(cmax)
    for j in range(int(cmin),cmax):
        for i in range(nx if not np.shares_memory(jd1,jd2) else min(nx,j+1)):
            a,b=int(id1[i]),int(id2[j]); p1=(lags[a-1],wids[a-1],scales[a-1]); p2=(lags[b-1],wids[b-1],scales[b-1])
            if mode=='normal': v=covmatij(a,b,jd1[i],jd2[j],sigma,tau,*p1,*p2)
            elif mode=='pmap': v=covmatpmapij(a,b,jd1[i],jd2[j],sigma,tau,*p1,*p2,scales[-1])
            else: v=covmatdpmapij(a,b,jd1[i],jd2[j],sigma,tau,lags[0],wids[0],scales[0],lags[1],wids[1],scales[1],lags[2],wids[2],scales[2])
            mat[i,j]=v

def covmat_bit(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,cmin=0,cmax=-1,symm=False): _fill_normal_vectorized(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,cmin,cmax)
def covmatpmap_bit(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,cmin=0,cmax=-1,symm=False): _fill_mode_vectorized(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,cmin,cmax,'pmap')
def covmatdpmap_bit(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,cmin=0,cmax=-1,symm=False): _fill_mode_vectorized(mat,jd1,jd2,id1,id2,sigma,tau,slagarr,swidarr,scalearr,cmin,cmax,'dpmap')

class _Compat:
    pass
spear_covfunc=_Compat()
for _n,_v in list(globals().items()):
    if callable(_v) and not _n.startswith('_'): setattr(spear_covfunc,_n,_v)
