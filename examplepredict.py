# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 03:22:02 2022

@author: dingxu
"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from astropy.timeseries import LombScargle
from tensorflow.keras.models import load_model
from scipy.fftpack import fft,ifft
import pandas as pd  

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model = load_model('classifymodel.hdf5')

def classifyfftdata(phases, resultmag, P):
    phases = np.copy(phases)
    resultmag = np.copy(resultmag)
    N = 100
    x = np.linspace(0,1,N)
    y = np.interp(x, phases, resultmag) 

    fft_y = fft(y) 
    half_x = x[range(int(N/2))]  #取一半区间
    abs_y = np.abs(fft_y) 
    normalization_y = abs_y/N            #归一化处理（双边频谱）                              
    normalization_half_y = normalization_y[range(int(N/2))] 
    normalization_half_y[0] = P
    sy1 = np.copy(normalization_half_y)

    sy1 = sy1[0:50]
    nparraydata = np.reshape(sy1,(1,50)) #mlpmodel
    prenpdata = model.predict(nparraydata)

    index = np.argmax(prenpdata[0])
    return index, np.max(prenpdata[0])

def readfits(fits_file):
    with fits.open(fits_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        print(hdulist[0].header['OBJECT'])
        print(hdulist[0].header['RA_OBJ'], hdulist[0].header['DEC_OBJ'])
        
        indexflux = np.argwhere(pdcsap_fluxes > 0)
#        print(sap_fluxes)
        time = tess_bjds[indexflux]
        time = time.flatten()
        flux = pdcsap_fluxes[indexflux]
        flux =  flux.flatten()
        objectname = hdulist[0].header['OBJECT']
        RA = hdulist[0].header['RA_OBJ']
        DEC = hdulist[0].header['DEC_OBJ']
        return time, flux, objectname, RA, DEC
    
def computeperiod(JDtime, targetflux):
   
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.01,maximum_frequency=40)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower

def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    
    lendata =  int((per/26)*2*len(times))
     
    time = times[0:lendata]
    mag = mags[0:lendata]
    phases = foldAt(time, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mag[sortIndi]
    return phases, resultmag

def stddata(timedata, fluxdata, P):
    yuanflux = np.copy(fluxdata)
    yuanmag = -2.5*np.log10(yuanflux)
    
    phases, resultmag = pholddata(P, timedata, fluxdata)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stddata = np.std(yuanmag)
    return stddata/datanoise

file = 'tess2018206045859-s0001-0000000025226885-0120-s_lc.fits'
path = ''
tbjd, fluxes, objectname, RA, DEC = readfits(path+file)
comper, wrongP, maxpower = computeperiod(tbjd, fluxes)
#计算两个周期的标准差
stdodata1 = stddata(tbjd, fluxes, comper)
stdodata2 = stddata(tbjd, fluxes, comper*2)
               
print('stdodata1= '+str(stdodata1))
print('stdodata2= '+str(stdodata2))
print('period= '+str(comper))

if (stdodata2/stdodata1)>1.5:
    P = comper*2
    phases, resultmag = pholddata(comper*2, tbjd, fluxes)
else:
    P = comper
    phases, resultmag = pholddata(comper, tbjd, fluxes)

index, prob = classifyfftdata(phases, resultmag, P)    


plt.figure(1)
plt.plot(phases, resultmag,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

if index == 0:
    plt.title('Prediction is ROT')
    print('probility is ROT '+str(prob))

if index == 1:
    plt.title('Prediction is DSCT')
    print('probility is DSCT '+str(prob))

if index == 2:
    plt.title('Prediction is EA')
    print('probility is EA '+str(prob))

if index == 3:
    plt.title('Prediction is EW')
    print('probility is EW '+str(prob))

if index == 4:
    plt.title('Prediction is MIRA')
    print('probility is MIRA '+str(prob))
    
if index == 5:
    plt.title('Prediction is RRAB')
    print('probility is RRAB '+str(prob))
    
if index == 6:
    plt.title('Prediction is RRC')
    print('probility is RRC '+str(prob))
    
if index == 7:
    plt.title('Prediction is SR')  
    print('probility is SR '+str(prob))
    
if index == 8:
    plt.title('Prediction is CEP') 
    print('probility is CEP '+str(prob))