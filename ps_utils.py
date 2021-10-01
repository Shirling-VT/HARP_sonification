from sonificationMethods.paulstretch_mono import paulstretch
import numpy as np
import datetime

#apply paulstretch to a time series 
def thm_fgm_paulstretch(times,data,stretch=8,window=512./44100,samplerate=44100,return_time=False):
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

def paulstretch_compressed(fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch, spacing):
    window=512./44100 #~0.0116

    #3-days, factor=6
    
    time_index = (fgs_gsm_time_itp >= start_time_plot) & (fgs_gsm_time_itp <= end_time_plot)
    times = fgs_gsm_time_itp[time_index]
    dB_phi_data = dB_phi_zero[time_index]
    
    samplerate = 44100
    paulStretch_dB_phi_zero = thm_fgm_paulstretch(times,dB_phi_data,samplerate=samplerate,
                                                  stretch=stretch,window=window,return_time=False)
    
    #calculate dB/dt after time stretch and save to ogg file
    stretch_spacing = spacing/stretch
    print(stretch_spacing)
    dB_phi_dt_aft_stretch = np.diff(paulStretch_dB_phi_zero)/stretch_spacing
    return dB_phi_dt_aft_stretch
