import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import numpy as np
from scipy.signal import stft
import datetime
import copy

def plot_time_series(start_time_plot, end_time_plot, fgs_gsm_time_itp, dB_phi_zero, probe, 
                     filename_str='Events'):
    #Plot detrended B_phi time series
    fig, ax1 = plt.subplots(1,1,figsize=(12,5))
    plt.axis('off')
    ax1.margins(0,0)
    
    xlim=[start_time_plot,end_time_plot]
    ylim=[-20,20]
    
    fig.patch.set_facecolor('k')
    ax1.plot(fgs_gsm_time_itp,dB_phi_zero,c='w', linewidth=0.2)
    ax1.set(xlim=xlim,ylim=ylim)
    
    fn = Path('image_output/'+filename_str+'_'+probe.upper()+'_dBphi_ts_'+
              start_time_plot.strftime("%Y%m%d")+'_'+
              end_time_plot.strftime("%Y%m%d")+'.png').expanduser()
    fig.savefig(fn, dpi = 600, bbox_inches='tight',pad_inches = 0)
    
def plot_spectra(start_time_plot, end_time_plot, fgs_gsm_time_itp, dB_phi_zero, spacing, probe, 
                 filename_str='Events'):
    fig, ax2 = plt.subplots(1,1,figsize=(12,5))
    plt.axis('off')
    ax2.margins(0,0)
    
    time_index = (fgs_gsm_time_itp >= start_time_plot) & (fgs_gsm_time_itp <= end_time_plot)
    dB_phi_ti = dB_phi_zero[time_index]
    dB_times = fgs_gsm_time_itp[time_index]
    #calculate dB/dt
    dB_phi_dt = np.diff(dB_phi_ti)/spacing
    
    xlim=[start_time_plot,end_time_plot]
    
    #Compute the Short Time Fourier Transform (STFT)
    f, t, Zxx = stft(dB_phi_dt, fs=1/3.0, nperseg=1024,noverlap=768)
    dt_list = [dB_times[0]+datetime.timedelta(seconds=ii) for ii in t]
    
    mag = np.abs(Zxx)
    #print(np.amax(mag))
    
    mag_2mHz = mag[7:,:]
    max_mag_2mHz = np.amax(mag_2mHz)
    #print(max_mag_2mHz)
    
    #set bad data 0 (i.e., data gap) to black
    mag = np.ma.masked_where(mag == 0.0, mag)
    cmap = copy.copy(plt.get_cmap("inferno"))
    cmap.set_bad(color='black')
    
    im=ax2.pcolormesh(dt_list, f*1000., mag/max_mag_2mHz,cmap=cmap, 
                      norm=colors.LogNorm(vmin=0.001, vmax=1.), shading='auto')#
    ax2.set(xlim=xlim,ylim=[1,100])
    ax2.set_yscale('log')
    
    fn = Path('image_output/'+filename_str+'_'+probe.upper()+'_dBphidt_stft_0.001-1_'+
              start_time_plot.strftime("%Y%m%d")+'_'+end_time_plot.strftime("%Y%m%d")+
              '.png').expanduser()
    fig.savefig(fn, dpi = 600, bbox_inches='tight',pad_inches = 0)
