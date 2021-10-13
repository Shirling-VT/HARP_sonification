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

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

def process_data(start_time=datetime.datetime(2008, 12, 7), end_time=datetime.datetime(2008, 12, 10),probe='thd', 
                 spacing = 3., pos_min = 5, stretch = 6, samplerate = 44100, filetype = ['wav'],filename_str='Events'):
    #load magnetic field data from CDAWeb  
    Mag_data = GSM.Mag()
    Mag_data.load_data(start_time, end_time, probe, product='fgs', coord='gsm')
    Mag_data.despike(dBdt_th = 10.,num=10)#despike
    Mag_data.interpolate(spacing) #interpolate the mag data to be evenely spaced time series
    
    #load state/ephemeris data from CDAWeb
    State_data = GSM.State()
    State_data.load_data(start_time,end_time,probe,coord='gsm')
    
    #interpolate position data to the fgs time stamps
    State_data.interpolate(Mag_data.fgs_gsm_epoch_itp)
    
    #save original 1 min resolution position file
    write_utils.write_pos_file(start_time, end_time, State_data.pos_time, 
                               State_data.pos_x, State_data.pos_y, State_data.pos_z, 
                               probe,filename_str=filename_str)
    print('Write to satellite position file finished!')
    
    #load electron moments data from CDAWeb for magnetosheath interval identification
    ESA_data = ESA.ESA();
    ESA_data.load_data(start_time, end_time, probe)
    
    #interpolate ESA data to the fgs time stamps
    ESA_data.interpolate(Mag_data.fgs_gsm_epoch_itp)
    
    #remove periods when spacecraft is in the magnetosheath
    #magnetosheath is when r>8 AND either (1) density is above 10/CC or (2) perp flux>2e7 or (3) tailward velocity>200km/s
    sheath_flag = (State_data.pos_r > 8) & ((ESA_data.interp_density > 10) | 
                                    (ESA_data.interp_velocity < -200) | (ESA_data.flux_perp > 2e7))
    
    #remove magnetosheath data points that are 9 sec on either side of the centered data point
    sheath_flag_update = utils.find_neighbors(sheath_flag,num=3)
    
    #Linear interpolation to fill “NAN” gaps in the magnetic field time series due to magnetosheath flags
    Mag_data.linear_interpolate(sheath_flag_update)
    
    #Convert mag data to a pandas dataframe and do running acerage detrend
    df = utils.convert_mag_data(Mag_data.Bx_itp, Mag_data.By_itp, Mag_data.Bz_itp, Mag_data.fgs_gsm_time_itp)
    
    #Detrending by substracting 30-min running average
    #This will result in nan at the edges, we will replace nan with zeros in the end
    Mag_data.detrend(df,window=int(1800/spacing))
    
    #Rotate detrend B into field-aligned coordinates
    dB_phi = utils.detrend_rotate(State_data.interp_pos_x, State_data.interp_pos_y, 
                                  State_data.interp_pos_z, Mag_data.detrend_Bx,
                                  Mag_data.detrend_By, Mag_data.detrend_Bz, Mag_data.Bx_SMA,
                                  Mag_data.By_SMA, Mag_data.Bz_SMA)
    
    #replace periods when spacecraft position is at r<5 Re with zeros
    dB_phi_zero = utils.replace_periods(State_data.pos_r, dB_phi, pos_min)
    
    #paulstretch
    dB_phi_dt_aft_stretch = ps_utils.paulstretch_dBdt(Mag_data.fgs_gsm_time_itp, start_time, end_time, 
                                                      dB_phi_zero, stretch, spacing,ps_window=512./44100,
                                                      samplerate = samplerate)
    #Write sound file
    for ft in filetype:
        write_utils.write_sound_file(probe, start_time, end_time, stretch, 
                                     dB_phi_dt_aft_stretch, samplerate, ft,
                                     filename_str=filename_str)
        print('Write to %s sound file finished!' %(ft))
    
    #Plot detrended B_phi time series
    plot_utils.plot_time_series(start_time, end_time, Mag_data.fgs_gsm_time_itp, dB_phi_zero, 
                                probe,filename_str=filename_str)
    print('Plot time series finished!')
    
    #Plot dB_phi/dt spectra
    plot_utils.plot_spectra(start_time, end_time, Mag_data.fgs_gsm_time_itp, 
                            dB_phi_zero, spacing, probe, filename_str=filename_str)
    print('Plot spectrogram finished!')
    
#start_time = datetime.datetime(2008,11,18,3,40)
#end_time = datetime.datetime(2008,11,21,3,30)
#probe='the'
#process_data(start_time=start_time, end_time=end_time,probe=probe)