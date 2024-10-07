import numpy as np
from scipy.integrate import quad
import scipy.special
from scipy.optimize import fsolve as fsolve
from scipy.integrate import quad
from scipy.special import jv as jv

#==========================================
# CONFINED AQUIFERS:
# THEIS (1945) solution:
# First we define the well function


def WF_T1945(u):
    return scipy.special.exp1(u)


def T1945(Q, T, S, r, t):
    if isinstance(t, float):
        u = (r ** 2 * S) / (4. * T * t)
        return Q / (4. * np.pi * T) * WF_T1945(u)
    else:
        s = np.zeros((len(t)), dtype=np.float)
        for c, ti in enumerate(t):
            u = (r ** 2 * S) / (4. * T * ti)
            s[c] = Q / (4. * np.pi * T) * WF_T1945(u)
        return s

#==========================================
# LEAKY CONFINED AQUIFERS:

# HANTUSCH AND JACOB (1955) solution [Schwartz and Zhang, 2003 pp. 240-245]
# First we define the "Well function"


def WF_HJ1955(u, r, B):
    """
    Hantusch and Jacob (1955) "Well function"
    :param u: r ** 2 * S / 4. / T / t
    :param r: distance to observation point
    :param B: np.sqrt(T*bm/km)
    :return: solution to the well function
    """
    def integrand(z, r, B):
        x = r ** 2 / (4 * B ** 2)
        return (1 / z) * np.exp(-z-(x/z))
    return quad(integrand, u, np.inf, args=(r, B))[0]


def HJ1955(Q, T, S, r, t, bm, km):
    """
    Hantusch and Jacob (1955) solution for leaky confined aquifer.
    Solution without storage in confining bed
    :param Q:
    :param T:
    :param S:
    :param r:
    :param t:
    :param bm:
    :param km:
    :return:
    """
    B = np.sqrt(T*bm/km)

    if isinstance(t, float):
        u = r ** 2 * S / 4. / T / t
        return Q / 4. / np.pi / T * WF_HJ1955(u,r,B)
    else:
        s = np.zeros((len(t)), dtype=np.float)
        for c, ti in enumerate(t):
            u = r ** 2 * S / 4. / T / ti
            s[c] = Q / 4. / np.pi / T * WF_HJ1955(u,r,B)
        return s

## HANTUSCH (1960) solution [Schwartz and Zhang, 2003 pp. 248-253]
# First we define the "Well function"


def WF_H1960(u, beta):
    def integrand(y, u, beta):
        x = (beta * np.sqrt(u)) / np.sqrt(y * (y - u))
        return (np.exp(-y) / y) * scipy.special.erfc(x)
    return quad(integrand, u, np.inf, args=(u, beta))[0]


def H1960(Q, Ka, K1, Sa, S1, ba, b1, r, t):
    """
    :param Q: Well pumping rate
    :param Ka: Hydraulic conductivity of aquifer
    :param K1: Hydraulic conductivity of aquitard
    :param Sa: Storativity of aquifer
    :param S1: Storativity of aquitard
    :param ba: Thickness of aquifer
    :param b1: Thickness of aquitard
    :param r: Distance to observation well
    :param t: output times for drawdown
    :return: Estimated drawdown at distance r
    """
    T = Ka * ba
    beta = (r / 4) * (np.sqrt((K1 * S1) / (b1 * T * Sa)))
    if isinstance(t, float):
        u = r ** 2 * Sa / 4. / T / t
        return Q / (4. * np.pi * T) * WF_H1960(u, beta)
    else:
        s = np.zeros((len(t)), dtype=np.float)
        for c, ti in enumerate(t):
            u = r ** 2 * Sa / 4. / T / ti
            s[c] = Q / (4. * np.pi * T) * WF_H1960(u, beta)
        return s
