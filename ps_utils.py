from sonificationMethods.paulstretch_mono import paulstretch
import numpy as np
import datetime

#apply paulstretch to a time series 
def thm_fgm_paulstretch(times,data,stretch=6,window=512./44100,samplerate=44100,return_time=False):
# Window for paulstretch is specifed so as to be equivalent to a window of 512 samples when using 
# the default sample rate of 44100
    paulStretch_data = paulstretch(data,stretch,window,samplerate=samplerate)
    
    if return_time == False:
        return paulStretch_data
    else:
        epochs = [ii.timestamp() for ii in times]
        epoch_stretch = np.linspace(epochs[0],epochs[-1],int(len(times) * stretch))
        epoch_stretch = epoch_stretch[:len(paulStretch_data)]
        times_interp_dt = np.array([datetime.datetime.fromtimestamp(ii) for ii in epoch_stretch])
        return times_interp_dt,paulStretch_data

def paulstretch_dBdt(fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch, spacing,
                     ps_window=512./44100,samplerate = 44100):
    #3-days, factor=6
    
    time_index = (fgs_gsm_time_itp >= start_time_plot) & (fgs_gsm_time_itp <= end_time_plot)
    times = fgs_gsm_time_itp[time_index]
    dB_phi_data = dB_phi_zero[time_index]
    
    paulStretch_dB_phi_zero = thm_fgm_paulstretch(times,dB_phi_data,stretch=stretch,window=ps_window,
                                                  samplerate=samplerate,return_time=False)
    #calculate dB/dt after time stretch 
    stretch_spacing = spacing/stretch
    dB_phi_dt_aft_stretch = np.diff(paulStretch_dB_phi_zero)/stretch_spacing
    return dB_phi_dt_aft_stretch
