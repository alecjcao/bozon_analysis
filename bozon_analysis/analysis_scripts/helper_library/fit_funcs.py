import numpy as np

def line(x, m, b):
    return m*x + b

def parabola(x, x0, m, y0):
    return m*(x-x0)**2 + y0

def cosine(x, x0, f, A, y0):
    return A*np.cos(np.pi*f*(x-x0))**2+y0

def sine(x, x0, f, A, y0):
    return A*np.sin(np.pi*f*(x-x0))**2+y0

def exponential(t, tau, A, y0):
    return A*np.exp(-t/tau)+y0

def rabi_resonance(f, f0, A, rabi, T, y0):
    rabi_gen = np.sqrt(rabi**2 + (2*np.pi*(f-f0))**2)
    return A*(rabi/rabi_gen)**2*np.sin(rabi_gen*T/2)**2 + y0

def lorentzian(f, f0, A, gamma, y0):
    return A*gamma**2/(gamma**2 + (f-f0)**2) + y0

def triple_lorentzian(f, f0, deltaf, A, Al, Ar, gamma, gammas, y0):
    return lorentzian(f, f0-deltaf, Al, gammas, 0) + lorentzian(f, f0, A, gamma, y0) + lorentzian(f, f0+deltaf, Ar, gammas, 0)

def gaussian(x, x0, sigma, A, y0):
    return A*np.exp(-(x-x0)**2/(2*sigma**2)) + y0