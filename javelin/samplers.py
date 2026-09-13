"""Sampler compatibility layer for JAVELIN.

Backends
--------
internal : bundled historical emcee-compatible sampler (default fallback)
emcee    : external emcee >= 3 using a v2-style compatibility adapter
auto     : external emcee if available, otherwise internal
"""
from __future__ import annotations
import os
import numpy as np

_BACKEND = os.environ.get('JAVELIN_EMCEE_BACKEND','auto').lower()

def set_sampler_backend(name):
    global _BACKEND
    name=name.lower()
    if name not in {'auto','internal','emcee','external'}: raise ValueError('backend must be auto, internal, or emcee')
    _BACKEND='emcee' if name=='external' else name

def get_sampler_backend(): return _BACKEND

class ExternalEmceeSampler:
    def __init__(self,nwalkers,ndim,lnpostfn,args=(),kwargs=None,threads=1,**opts):
        import emcee
        self._emcee=emcee; self._pool=None; self._cache_chain=None; self._cache_logp=None
        if threads and int(threads)>1:
            from multiprocessing.pool import ThreadPool
            self._pool=ThreadPool(int(threads))
        self._sampler=emcee.EnsembleSampler(nwalkers,ndim,lnpostfn,args=args,kwargs={} if kwargs is None else kwargs,pool=self._pool)
    def _invalidate(self): self._cache_chain=self._cache_logp=None
    def run_mcmc(self,p0,N,rstate0=None,**kwargs):
        self._invalidate()
        progress = kwargs.pop('progress', False)
        initial_state = p0
        if rstate0 is not None:
            try:
                initial_state = self._emcee.State(p0, random_state=rstate0)
            except Exception:
                # If an older/newer emcee object rejects the explicit State,
                # fall back to positions and set the sampler RNG state.
                try:
                    self._sampler.random_state = rstate0
                except Exception:
                    pass
        state=self._sampler.run_mcmc(initial_state,N,progress=progress,**kwargs)
        return np.asarray(state.coords), np.asarray(state.log_prob), getattr(state,'random_state',None)
    def reset(self): self._sampler.reset(); self._invalidate()
    def close(self):
        if self._pool is not None:
            try:
                self._pool.close(); self._pool.join()
            finally:
                self._pool=None

    @property
    def chain(self):
        if self._cache_chain is None: self._cache_chain=np.transpose(self._sampler.get_chain(),(1,0,2)).copy()
        return self._cache_chain
    @property
    def flatchain(self): return self.chain.reshape((-1,self.chain.shape[-1]))
    @property
    def lnprobability(self):
        if self._cache_logp is None: self._cache_logp=self._sampler.get_log_prob().T.copy()
        return self._cache_logp
    @property
    def flatlnprobability(self): return self.lnprobability.reshape(-1)
    @property
    def acceptance_fraction(self): return self._sampler.acceptance_fraction
    def __del__(self):
        try: self.close()
        except Exception: pass

def EnsembleSampler(*args,**kwargs):
    backend=_BACKEND
    if backend in {'auto','emcee'}:
        try:
            import emcee
            return ExternalEmceeSampler(*args,**kwargs)
        except ImportError:
            if backend=='emcee': raise ImportError("External emcee backend requested. Install with: pip install 'javelin-tng[emcee]'")
    from .emcee_internal import EnsembleSampler as Internal
    return Internal(*args,**kwargs)
