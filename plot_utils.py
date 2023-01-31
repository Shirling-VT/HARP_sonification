import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import numpy as np
from scipy.signal import stft
import datetime
import copy
import os


def plot_time_series(start_time, end_time, times, data, probe, ylim=[-20, 20], filename_str='Events'):
    # Plot detrended B_phi time series
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    plt.axis('off')
    ax1.margins(0, 0)

    xlim = [start_time, end_time]

    fig.patch.set_facecolor('k')
    ax1.plot(times, data, c='w', linewidth=0.4)
    ax1.set(xlim=xlim, ylim=ylim)

    directory = 'outputs/' + filename_str + '_' + start_time.strftime("%Y%m%d")
    if not os.path.exists(directory):
        os.mkdir(directory)

    fn = Path(directory + '/' + probe.upper() + '_ts_' +
              start_time.strftime("%Y%m%d") + '_' +
              end_time.strftime("%Y%m%d") + '_ylim' + str(ylim[1]) + '.png').expanduser()
    fig.savefig(fn, dpi=600, bbox_inches='tight', pad_inches=0)


def plot_spectra(start_time, end_time, times, data, spacing, probe, ylim=[1, 100],
                 filename_str='Events', fix_cb=True, dynamic_cb=True):
    time_index = (times >= start_time) & (times <= end_time)
    dB_phi_ti = data[time_index]
    dB_times = times[time_index]

    # calculate dB/dt
    dB_phi_dt = np.diff(dB_phi_ti) / spacing

    xlim = [start_time, end_time]

    # Compute the Short Time Fourier Transform (STFT)
    f, t, Zxx = stft(dB_phi_dt, fs=1 / spacing, nperseg=1024, noverlap=768)
    dt_list = [dB_times[0] + datetime.timedelta(seconds=ii) for ii in t]

    mag = np.abs(Zxx)
    # print(np.amax(mag))

    # creat a new array in the frequency array of >= 2 mHz and <= ylim[1] mHz
    difference_array_lb = np.absolute(f * 1000 - 2.)
    index_lb = difference_array_lb.argmin()
    if f[index_lb] * 1000 >= 2.:
        index_lb = index_lb + 1

    difference_array_hb = np.absolute(f * 1000 - ylim[1])
    index_hb = difference_array_hb.argmin()
    if f[index_hb] * 1000 >= ylim[1]:
        index_hb = index_hb - 1

    mag_freqb = mag[index_lb:index_hb, :]

    # find the maximum value in the stft amplitude array at >= 2mHz and <= ylim[1] mHz
    # and within the plotting range
    max_mag = np.amax(mag_freqb)
    mag_norm = mag / max_mag

    directory = 'outputs/' + filename_str + '_' + start_time.strftime("%Y%m%d")
    if not os.path.exists(directory):
        os.mkdir(directory)

    if dynamic_cb:
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        plt.axis('off')
        ax1.margins(0, 0)
        mag_norm_nan = mag_norm.copy()
        mag_norm_nan[mag_norm_nan == 0.0] = np.nan
        vmin = np.nanpercentile(mag_norm_nan, 50)
        vmax = np.nanpercentile(mag_norm_nan, 95)

        # set bad data 0 (i.e., data gap) to black
        mag_norm_nan = np.ma.masked_where(mag_norm_nan == np.nan, mag_norm_nan)
        cmap = copy.copy(plt.get_cmap("inferno"))
        cmap.set_bad(color='black')

        im = ax1.pcolormesh(dt_list, f * 1000., mag_norm_nan, cmap=cmap,
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto')  #
        ax1.set(xlim=xlim, ylim=ylim)
        ax1.set_yscale('log')
        fn = Path(directory + '/' + probe.upper() + '_stft_' +
                  start_time.strftime("%Y%m%d") + '_' + end_time.strftime("%Y%m%d") +
                  '_dynamic.png').expanduser()
        fig1.savefig(fn, dpi=600, bbox_inches='tight', pad_inches=0)
        fig1.cla()
        ax1.cla()

    if fix_cb:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
        plt.axis('off')
        ax2.margins(0, 0)

        vmin = 0.01
        vmax = 1.
        # set bad data 0 (i.e., data gap) to black
        mag_norm = np.ma.masked_where(mag_norm == 0, mag_norm)
        cmap = copy.copy(plt.get_cmap("inferno"))
        cmap.set_bad(color='black')
        im = ax2.pcolormesh(dt_list, f * 1000., mag_norm, cmap=cmap,
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto')
        ax2.set(xlim=xlim, ylim=ylim)
        ax2.set_yscale('log')
        fn = Path(directory + '/' + probe.upper() + '_stft_' +
                  start_time.strftime("%Y%m%d") + '_' + end_time.strftime("%Y%m%d") +
                  '_fix.png').expanduser()
        fig2.savefig(fn, dpi=600, bbox_inches='tight', pad_inches=0)
        fig2.clf()
        ax2.cla()
