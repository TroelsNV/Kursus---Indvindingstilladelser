import numpy as np
from scipy.special import factorial
import scipy.special
#from mpmath import *
from scipy.integrate import quad


# STREAMFLOW DEPLEATION SOLUTIONS:

"""
NB! Solution adopted from FUNCTION spread sheet downloaded from 
https://sites.google.com/site/brucehuntsgroundwaterwebsite/ 


Calculation of drawdowns created by a well in a delayed-yield aquifer next to a stream.
All input and output variables are dimensionless and are defined as follows:
       W_4=Abs(h)*T/Q   x=x/L  y=y/L  t=t*T/(S*L^2)  lambda=lambda*L/T
                    K=(K'/B')*L^2/T  epsilon=S/sigma
where Abs(h)=drawdown, T=transmissivity of the bottom aquifer, Q=well flow rate, L=shortest
distance between the well and stream edge, S=storage coefficient of the bottom aquifer
(either storativity or specific yield), x=coordinate measured from the stream edge toward the
well (normal to the stream edge), y=coordinate measured along the stream edge,lambda=stream
bed leakage coefficient and K" and B" = permeability and thickness of the top aquifer. All
prime superscripts have been omitted in the program for notational convenience.
NOTE: Setting K'=0 for any value of epsilon gives the solution obtained by B.Hunt(1999)
GROUND WATER,37(1),98-102 for either a completely confined or completely unconfined aquifer.
"""


def g_4(p, K, epsilon, Lambda):
    m0 = np.sqrt(p * (p + K * (1 + epsilon)) / (p + epsilon * K))
    return Lambda * np.exp(-m0) / (p * (Lambda + 2 * m0))


def exp1(x):
    # To compute the exponential integral Exp1(x) for 0<x<infinity.
    A0 = -0.57721566
    a1 = 0.99999193
    a2 = -0.24991055
    A3 = 0.05519968
    A4 = -0.00976004
    A5 = 0.00107857
    B0 = 0.2677737343
    B1 = 8.6347608925
    B2 = 18.059016973
    B3 = 8.5733287401
    B4 = 1
    c0 = 3.9584969228
    c1 = 21.0996530827
    c2 = 25.6329561486
    c3 = 9.5733223454
    C4 = 1
    if x <= 1:
        return -np.log(x) + A0 + x * (a1 + x * (a2 + x * (A3 + x * (A4 + x * A5))))
    else:
        p1 = B0 + x * (B1 + x * (B2 + x * (B3 + x * B4)))
        P2 = c0 + x * (c1 + x * (c2 + x * (c3 + x * C4)))
        return (p1 / P2) * np.exp(-x) / x

def stehcoef(ii, n):
    M = int(np.round(n / 2., 0))
    upperlimit = np.min([ii, M])
    lowerlimit = int(np.floor((ii + 1) / 2.))

    s_coef = 0.

    for kk in range(lowerlimit,upperlimit+1):
        num = factorial(2 * kk) * kk ** M
        denom = factorial(M - kk) * factorial(kk) * factorial(kk - 1) * factorial(ii - kk) * factorial(2 * kk - ii)
        s_coef = s_coef + num / denom

    s_coef = s_coef * (-1) ** (ii + M)

    return s_coef

def bessI0(x):
    A0 = 1
    a1 = 3.5156229
    a2 = 3.0899424
    A3 = 1.2067492
    A4 = 0.2659732
    A5 = 0.0360768
    A6 = 0.0045813
    B0 = 0.39894228
    B1 = 0.01328592
    B2 = 0.00225319
    B3 = -0.00157565
    B4 = 0.00916281
    B5 = -0.02057706
    B6 = 0.02635537
    B7 = -0.01647633
    B8 = 0.00392377
    if x <= 3.75:
        t = (x / 3.75) ** 2
        BI0 = A0 + t * (a1 + t * (a2 + t * (A3 + t * (A4 + t * (A5 + t * A6)))))
    else:
        t = 3.75 / x
        BI0 = B0 + t * (B1 + t * (B2 + t * (B3 + t * (B4 + t * (B5 + t * (B6 + t * (B7 + t * B8)))))))
        BI0 = BI0 * np.exp(x) / np.sqrt(x)

    return BI0


def bessK0(x):
    A0 = -0.57721566
    a1 = 0.4227842
    a2 = 0.23069756
    A3 = 0.0348859
    A4 = 0.00262698
    A5 = 0.0001075
    A6 = 0.0000074
    B0 = 1.25331414
    B1 = -0.07832358
    B2 = 0.02189568
    B3 = -0.01062446
    B4 = 0.00587872
    B5 = -0.0025154
    B6 = 0.00053208
    if x <= 2.:
        t = (x / 2.) ** 2.
        BK0 = A0 + t * (a1 + t * (a2 + t * (A3 + t * (A4 + t * (A5 + t * A6)))))
        return BK0 - np.log(x / 2.) * bessI0(x)
    else:
        t = 2 / x
        BK0 = B0 + t * (B1 + t * (B2 + t * (B3 + t * (B4 + t * (B5 + t * B6)))))
        return BK0 * np.exp(-x) / np.sqrt(x)


#This computes the Laplace transform used in the calculation of drawdown for flow to
#a well in a delayed-yield aquifer.
def lap_trans(r, p, epsilon, K):
    M = np.sqrt(p + K - epsilon * K ** 2 / (p + epsilon * K))
    return bessK0(r * M) / (2 * np.pi * p)


def Boulton1963(r, t, epsilon, K):
    if t <= 0.:
        return 0.
    else:
        if t * epsilon * K >= 10:
            return exp1((1 + epsilon) * r ** 2 / (4 * t * epsilon)) / (4 * np.pi)
        else:
            ret = 0.
            n = 8
            for jj in range(1,n+1):
                ret = ret + stehcoef(jj, n) * lap_trans(r, jj * np.log(2) / t, epsilon, K)

            return ret * np.log(2) / t

def integrand_3(u, x, y, t, Lambda, K, epsilon):
    r = np.sqrt((1 + np.abs(x) + 2 * np.log(1 / u) / Lambda) ** 2 + y ** 2)

    return Boulton1963(r, t, epsilon, K)


def Hunt_drawdown(x1, y1, L1, t1, Lambda1, S1, sigma1, T1, Q, Km = 0., Bm = 1.):
    """
    Calculation of drawdowns created by a well in a delayed-yield aquifer next to a stream.
    All input and output variables are dimensionless and are defined as follows:
           W_4=Abs(h)*T/Q   x=x/L  y=y/L  t=t*T/(S*L^2)  lambda=lambda*L/T
                        K=(K' / B')*L^2/T  epsilon=S/sigma
    where Abs(h)=drawdown, T=transmissivity of the bottom aquifer, Q=well flow rate, L=shortest
    distance between the well and stream edge, S=storage coefficient of the bottom aquifer
    (either storativity or specific yield), x=coordinate measured from the stream edge toward the
    well (normal to the stream edge), y=coordinate measured along the stream edge,lambda=stream
    bed leakage coefficient and K" and B" = permeability and thickness of the top aquifer. All
    prime superscripts have been omitted in the program for notational convenience.
    NOTE: Setting K' = 0
    for any value of epsilon gives the solution obtained by B.Hunt(1999)
    GROUND WATER,37(1),98-102 for either a completely confined or completely unconfined aquifer.

    :param x1:
    :param y1:
    :param L1:
    :param t1:
    :param Lambda1:
    :param Km:
    :param Bm:
    :param S1:
    :param sigma1:
    :param Q:
    :return:
    """

    #Calculate scaled variables:
    x = x1/L1
    y = y1/L1
    t = t1 * T1 / (S1 * L1 ** 2)
    Lambda = Lambda1 * L1 / T1
    K = (Km / Bm) * L1 ** 2 / T1
    epsilon = S1 / sigma1

    if t <= 0.:
        ret = 0.

    else:
        n = 20
        delta = 1 / n
        ret = 0  #
        y_1 = 0  #

        for ii in range(1,n,2):
            y_2 = integrand_3(ii * delta, x, y, t, Lambda, K, epsilon)
            y_3 = integrand_3((ii + 1) * delta, x, y, t, Lambda, K, epsilon)
            ret = ret + delta * (y_1 + 4 * y_2 + y_3) / 3
            y_1 = y_3
    r = np.sqrt((x - 1) ** 2 + y ** 2)

    dd_scale = (Boulton1963(r, t, epsilon, K) - ret)

    return dd_scale*Q/T1


def Hunt_depleation(t1, T1, S1, L1, Lambda1, sigma1, Q, Km = 0., Bm = 1.):
    """
    This calculates the total flow depletion lost from a stream when a well abstracts a flow
    Q from a delayed-yield (semi-confined) aquifer. All input and output variables are
    dimensionless with the following definitions:
    Q_4=flow depletion/Q  t'=t*T/(S*L^2)  Lambda'=Lambda*L/T  Epsilon=S/Sigma  K=(K'/B')*L^2/T
    NOTE: Setting K'=0 for any value of epsilon gives the solution obtained by B.Hunt(1999)
    GROUND WATER,37(1),98-102 for either a completely confined or completely unconfined aquifer.
    :param t1:
    :param T1:
    :param S1:
    :param L1:
    :param Lambda1:
    :param Km:
    :param Bm:
    :param sigma1:
    :param Q:
    :return:
    """

    #Calculate scaled variables:
    t = t1 * T1 / (S1 * L1 ** 2)
    Lambda = Lambda1 * L1 / T1
    K = (Km / Bm) * L1 ** 2 / T1
    epsilon = S1 / sigma1

    if t <= 0.:
        ret = 0.

    else:
        n = 8
        ret = 0  #

        for ii in range(1, n+1):
            ret = ret + stehcoef(ii,n) * g_4(ii * np.log(2) / t, K, epsilon, Lambda)

        ret = ret * np.log(2) / t

    return ret
