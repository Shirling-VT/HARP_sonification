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

#load magnetic field data from CDAWeb
start_time = datetime.datetime(2008, 11, 11,16)
end_time = datetime.datetime(2008, 11, 14,16)
probe='the'

start_time = datetime.datetime(2008, 12, 7)
end_time = datetime.datetime(2008, 12, 10)
probe='thd'

spacing = 3.

fgs_gsm = thm.load_thm_fgm(start_time,end_time,probe=probe,product='fgs',coord='gsm')

fgs_gsm_time_itp, fgs_gsm_epoch_itp, fx0, fy0, fz0 = interp.fgs_interpolate(fgs_gsm, spacing)

fgs_gsm_Bx_itp = fx0(fgs_gsm_epoch_itp)
fgs_gsm_By_itp = fy0(fgs_gsm_epoch_itp)
fgs_gsm_Bz_itp = fz0(fgs_gsm_epoch_itp)

#load state/ephemeris data from CDAWeb
state_gsm = thm.load_thm_state(start_time,end_time,probe=probe,coord='gsm')
pos_x = state_gsm['X'] #unit in Earth radii
pos_y = state_gsm['Y']
pos_z = state_gsm['Z']
pos_time = state_gsm['EPOCH']

#save position file (no interpolation)
start_time_plot = start_time #datetime.datetime(2008, 11, 11,16)
end_time_plot = end_time # datetime.datetime(2008, 11, 14,16)
write_utils.write_pos_file(start_time_plot, end_time_plot, pos_time, pos_x, pos_y, pos_z, probe)

#interpolate position data to the fgs time stamps
interp_pos_x, interp_pos_y, interp_pos_z = interp.position_interpolate(
                                            pos_time, pos_x, pos_y, 
                                            pos_z, fgs_gsm_epoch_itp)
pos_r = np.sqrt(interp_pos_x**2+interp_pos_y**2+interp_pos_z**2)


#load electron moments data from CDAWeb for magnetosheath interval identification
peem_time, density, velocity_x, flux_x, flux_y = utils.load_electron_moments(
                                                    start_time,end_time,probe)

#interpolate ESA data to the fgs time stamps
interp_velocity, interp_density, flux_perp = interp.em_interpolate(
    peem_time, density, fgs_gsm_epoch_itp, velocity_x, flux_x, flux_y)

#remove periods when spacecraft is in the magnetosheath
#magnetosheath is when r>8 AND either (1) density is above 10/CC or (2) perp flux>2e7 or (3) tailward velocity>200km/s
sheath_flag = (pos_r > 8) & ((interp_density > 10) | (interp_velocity < -200) | (flux_perp > 2e7))

#remove magnetosheath data points that are 9 sec on either side of the centered data point
sheath_flag_update = utils.find_neighbors(sheath_flag,num=3)
print(sum(sheath_flag_update)) #number of Trues in the list
print(sum(sheath_flag))

#Linear interpolation to fill “NAN” gaps in the magnetic field time series due to magnetosheath flags
Bx_itp, By_itp, Bz_itp = interp.linear_interpolate(fgs_gsm_Bx_itp, fgs_gsm_By_itp, 
                                                   fgs_gsm_Bz_itp, sheath_flag_update, 
                                                   fgs_gsm_epoch_itp, fx0, fy0, fz0)

#Convert mag data to a pandas dataframe and do running acerage detrend
df = utils.convert_mag_data(Bx_itp, By_itp, Bz_itp, fgs_gsm_time_itp)
print(df)

#Detrending by substracting 30-min running average
#This will result in nan at the edges, we will replace nan with zeros in the end
detrend_Bx, detrend_By, detrend_Bz, Bx_SMA, By_SMA, Bz_SMA = utils.detrend(
                                                            df, Bx_itp, By_itp, 
                                                            Bz_itp)

#Rotate detrend B into field-aligned coordinates
dB_phi = utils.detrend_rotate(interp_pos_x,interp_pos_y,interp_pos_z,detrend_Bx,
                             detrend_By,detrend_Bz,Bx_SMA,By_SMA,Bz_SMA)

#replace periods when spacecraft position is at r<5 Re with zeros
pos_min = 5
dB_phi_zero = utils.replace_periods(pos_r, dB_phi, pos_min)

print(dB_phi_zero[5000:5010])
print(dB_phi[5000:5010])

#paulstretch
stretch = 6
dB_phi_dt_aft_stretch = ps_utils.paulstretch_compressed(fgs_gsm_time_itp, 
                                            start_time_plot, end_time_plot, 
                                            dB_phi_zero, stretch, spacing)

#paulStretch_detrend_Bx_nan[r_flag] = 0 
#paulStretch_detrend_Bx_nan[sheath_flag] = 0
samplerate = 44100

#Write sound file
filetype = 'wav'
write_utils.write_sound_file(probe, start_time_plot, end_time_plot, stretch, 
                 dB_phi_dt_aft_stretch, samplerate, filetype)

#Plot detrended B_phi time series
plot_utils.plot_B_phi_series(start_time_plot, end_time_plot, fgs_gsm_time_itp, dB_phi_zero, probe)

#Plot dB_phi/dt spectra
plot_utils.plot_dB_phi_dt_spectras(start_time, start_time_plot, end_time_plot, 
                                   fgs_gsm_time_itp, dB_phi_zero, spacing, probe)
