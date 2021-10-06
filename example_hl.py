#Package imports
import datetime
import numpy as np

#File imports
import themis as thm
import interpolations as interp
import utils
import ps_utils
import write_utils
import plot_utils

#High level file imports
import GSM
import ESA

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def process_data(start_time=datetime.datetime(2008, 12, 7), end_time=datetime.datetime(2008, 12, 10),
                 probe='thd', spacing = 3., pos_min = 5, stretch = 6, samplerate = 44100, filetype = 'wav'):
    #load magnetic field data from CDAWeb  
    Mag_data = GSM.Mag()
    Mag_data.load_data(start_time, end_time, probe, product='fgs', coord='gsm')
    Mag_data.interpolate(spacing)
    
    
    #load state/ephemeris data from CDAWeb
    State_data = GSM.State()
    State_data.load_data(start_time,end_time,probe,coord='gsm')
    
    #interpolate position data to the fgs time stamps
    State_data.interpolate(Mag_data.fgs_gsm_epoch_itp)
    
    #save position file (no interpolation)
    start_time_plot = start_time #datetime.datetime(2008, 11, 11,16)
    end_time_plot = end_time # datetime.datetime(2008, 11, 14,16)
    write_utils.write_pos_file(start_time_plot, end_time_plot, State_data.pos_time, 
                               State_data.pos_x, State_data.pos_y, State_data.pos_z, 
                               probe)
    
    #load electron moments data from CDAWeb for magnetosheath interval identification
    ESA_data = ESA.ESA();
    ESA_data.load_data(start_time_plot, end_time_plot, start_time, end_time, probe)
    
    #interpolate ESA data to the fgs time stamps
    ESA_data.interpolate(Mag_data.fgs_gsm_epoch_itp)
    
    #remove periods when spacecraft is in the magnetosheath
    #magnetosheath is when r>8 AND either (1) density is above 10/CC or (2) perp flux>2e7 or (3) tailward velocity>200km/s
    sheath_flag = (State_data.pos_r > 8) & ((ESA_data.interp_density > 10) | 
                                    (ESA_data.interp_velocity < -200) | (ESA_data.flux_perp > 2e7))
    
    #remove magnetosheath data points that are 9 sec on either side of the centered data point
    sheath_flag_update = utils.find_neighbors(sheath_flag,num=3)
    print(sum(sheath_flag_update)) #number of Trues in the list
    print(sum(sheath_flag))
    
    #Linear interpolation to fill “NAN” gaps in the magnetic field time series due to magnetosheath flags
    Mag_data.linear_interpolate(sheath_flag_update)
    
    #Convert mag data to a pandas dataframe and do running acerage detrend
    df = utils.convert_mag_data(Mag_data.Bx_itp, Mag_data.By_itp, Mag_data.Bz_itp, Mag_data.fgs_gsm_time_itp)
    print(df)
    
    #Detrending by substracting 30-min running average
    #This will result in nan at the edges, we will replace nan with zeros in the end
    Mag_data.detrend(df)
    
    
    #Rotate detrend B into field-aligned coordinates
    dB_phi = utils.detrend_rotate(State_data.interp_pos_x, State_data.interp_pos_y, 
                                  State_data.interp_pos_z, Mag_data.detrend_Bx,
                                  Mag_data.detrend_By, Mag_data.detrend_Bz, Mag_data.Bx_SMA,
                                  Mag_data.By_SMA, Mag_data.Bz_SMA)
    
    #replace periods when spacecraft position is at r<5 Re with zeros
    dB_phi_zero = utils.replace_periods(State_data.pos_r, dB_phi, pos_min)
    
    print(dB_phi_zero[5000:5010])
    print(dB_phi[5000:5010])
    
    #paulstretch
    
    dB_phi_dt_aft_stretch = ps_utils.paulstretch_compressed(Mag_data.fgs_gsm_time_itp, 
                                                start_time_plot, end_time_plot, 
                                                dB_phi_zero, stretch, spacing)
    
    #paulStretch_detrend_Bx_nan[r_flag] = 0 
    #paulStretch_detrend_Bx_nan[sheath_flag] = 0
    
    #Write sound file
    write_utils.write_sound_file(probe, start_time_plot, end_time_plot, stretch, 
                     dB_phi_dt_aft_stretch, samplerate, filetype)
    
    #Plot detrended B_phi time series
    plot_utils.plot_B_phi_series(start_time_plot, end_time_plot, Mag_data.fgs_gsm_time_itp, dB_phi_zero, probe)
    
    #Plot dB_phi/dt spectra
    plot_utils.plot_dB_phi_dt_spectras(start_time, start_time_plot, end_time_plot, 
                                       Mag_data.fgs_gsm_time_itp, dB_phi_zero, spacing, probe)
    
process_data()
