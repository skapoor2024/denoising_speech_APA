import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import scipy.signal as sg
import pandas as pd
import math

from scipy.io import wavfile

def plott(sig,title,xlabel,figsize = (12,3)):
    fig = plt.figure(figsize=figsize)
    plt.plot(sig)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plott_2(sig1,sig2,title,xlabel,ylabel,figsize = (12,4)):
    fig = plt.figure(figsize = figsize)
    plt.plot(sig1,sig2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def wts_snr(wts_opt,wts_tf):
    
    l1 = len(wts_opt)
    l2 = len(wts_tf)
    
    if l1>l2:
        
        wts_tf = np.append(wts_tf,np.zeros(l1-l2))
        
    if l2>l1:
        
        wts_opt = np.append(wts_opt,np.zeros(l2-l1))
        
    wts_diff = wts_opt-wts_tf
    
    wts_tf = np.expand_dims(wts_tf,axis = 0)
    wts_diff = np.expand_dims(wts_diff,axis = 0)
        
    wsnr = 10*math.log((wts_tf@wts_tf.T)/(wts_diff@wts_diff.T))
    
    return wsnr

# calculate normalize error of the computed filters
def err_wav(wts,ip,op):
    
    op_e = sg.lfilter(wts,1,ip[0])
    
    eror = op_e - op[0]
    
    #ip = np.expand_dims(ip,axis = 0)
    eror = np.expand_dims(eror,axis = 0)
    
    nmse = np.divide(eror@eror.T,ip@ip.T)
    
    return nmse.item()

def rls(ip,op,l,alpha = 0.9999):
    
    ip_len = len(ip[0])
    ip_mean = np.sum(ip[0])/ip_len
    ip_std = np.sqrt(np.sum(np.square(ip[0]-ip_mean))/ip_len)
    
    i = l
    
    R_inv = 100*ip_std*np.identity(l+1)
    W = np.ones((l+1,1))
    
    while i<ip_len:
        
        x = ip[0,i]
        d = op[0,i]
        X = np.expand_dims(ip[0,i-l:i+1],axis = 1)
        e = (d - X.T@W)
        #print(e.shape)
        e = e.item()
        G = R_inv@X/(alpha+X.T@R_inv@X)
        R_inv = (R_inv-G@X.T@R_inv)/alpha
        W = W + e*G
        i = i+1
    
    return np.flip(W.T[0])

def rls_2(ip,op,l,alpha = 0.9999):
    
    w_len = l+1
    
    ip_len = len(ip[0])
    ip_mean = np.sum(ip[0])/ip_len
    ip_std = np.sqrt(np.sum(np.square(ip[0]-ip_mean))/ip_len)
    
    i = l
    
    R_inv = 100*ip_std*np.identity(w_len)
    W = np.ones((w_len,1))
    
    lms_e = []
    wts = []
    
    while i<ip_len:
        
        x = ip[0,i]
        d = op[0,i]
        X = np.expand_dims(ip[0,i-l:i+1],axis = 1)
        e = (d - X.T@W)
        #print(e.shape)
        e = e.item()
        G = R_inv@X/(alpha+X.T@R_inv@X)
        R_inv = (R_inv-G@X.T@R_inv)/alpha
        W = W + e*G
        i = i+1
        
        wts.append(np.flip(W.T[0]))
        lms_e.append(e)
    
    return wts,lms_e,np.flip(W.T[0])

def lms(ip,op,l,step_size=0):
    
    w_len = l+1
    #W = np.random.normal(0,1,(w_len,1))
    W = np.ones((w_len,1))
    ip_len = len(ip[0])
    
    eta = step_size
    
    if step_size == 0:
        
        sig_power = np.sum(np.square(ip[0]))/ip_len
        eta = 1/(10*w_len*sig_power)
    
    
    lms_e = []
    wts = []
    
    i = l
    
    while i<ip_len:
        
        X = np.expand_dims(ip[0,i-l:i+1],axis = 1)
        d = op[0,i]
        e = (d - X.T@W).item()
        W = W + eta*e*X
        wts.append(np.flip(W.T[0]))
        lms_e.append(e)
        
        i = i+1
    
    return wts,lms_e,np.flip(W.T[0])

def nlms(ip,op,l,step_size=0):
    
    w_len = l+1
    #W = np.random.normal(0,1,(w_len,1))
    W = np.ones((w_len,1))
    ip_len = len(ip[0])
    
    eta = step_size
    
    if step_size == 0:
        
        sig_power = np.sum(np.square(ip[0]))/ip_len
        eta = 1/(10*w_len*sig_power)
    
    
    lms_e = []
    wts = []
    
    i = l
    
    while i<ip_len:
        
        X = np.expand_dims(ip[0,i-l:i+1],axis = 1)
        d = op[0,i]
        e = (d - X.T@W).item()
        sig_pow = (X.T@X).item()
        W = W + eta*e*X/sig_pow
        wts.append(np.flip(W.T[0]))
        lms_e.append(e)
        
        i = i+1
    
    return wts,lms_e,np.flip(W.T[0])

def speech_norm(x):
    
    x_mean = np.mean(x)
    #print(x_mean)
    x_max = np.abs(np.max(x))
    #print(x_max)
    return (x - x_mean)/x_max

def err_wav_op(wts,ip,op):
    
    op_e = sg.lfilter(wts,1,ip[0])
    
    eror = op_e - op[0]
    
    #ip = np.expand_dims(ip,axis = 0)
    eror = np.expand_dims(eror,axis = 0)
    
    return eror

