import numpy as np
import pandas as pd
import py_vollib.black_scholes
import py_vollib_vectorized


def compute_payoff_vec(spots, strikes, opt_types):
    is_call = (opt_types == "call")
    return np.where(is_call, np.maximum(spots - strikes, 0),
                             np.maximum(strikes - spots, 0))


def compute_option_price(spot, strike, cp, ttm, r, q, sigma):
    """
    Vectorized option pricing using py_vollib_vectorized.black_scholes.
    
    Parameters:
        spot: float or array
        strike: float or array
        cp: 'call' or 'put' or array-like
        ttm: float or array (in years)
        r: risk-free rate (scalar or array)
        q: dividend yield (scalar or array)
        sigma: implied volatility (float or array)
    
    Returns:
        np.ndarray of option prices
    """
    cp = np.asarray(cp)
    cp = np.where(cp == 'call', 'c', 'p')  # convert to py_vollib format

    #print(f"spot: {spot}")
    #print(f"strike: {strike}")
    #print(f"cp: {cp}")
    #print(f"ttm: {ttm}")
    #print(f"r: {r}")
    #print(f"q: {q}")
    #print(f"sigma: {sigma}")
    return py_vollib.black_scholes.black_scholes(
        flag=cp,
        S=np.asarray(spot),
        K=np.asarray(strike),
        t=np.asarray(ttm),
        r=np.asarray(r),
        sigma=np.asarray(sigma),
        return_as='numpy'
    )

def compute_option_price_old(spot, strike, cp, ttm, r, q, sigma):
    """
    Vectorized option pricing using py_vollib_vectorized.black_scholes.
    
    Parameters:
        spot: float or array
        strike: float or array
        cp: 'call' or 'put' or array-like
        ttm: float or array (in years)
        r: risk-free rate (scalar or array)
        q: dividend yield (scalar or array)
        sigma: implied volatility (float or array)
    
    Returns:
        np.ndarray of option prices
    """
    cp = np.asarray(cp)
    print(f"cp: {cp}")
    cp = np.where(cp == 'call', 'c', 'p')  # convert to py_vollib format
    print(f"cp: {cp}")

    return py_vollib.black_scholes.black_scholes(
        flag=cp,
        S=np.asarray(spot),
        K=np.asarray(strike),
        t=np.asarray(ttm),
        r=np.asarray(r),
        sigma=np.asarray(sigma)
    )