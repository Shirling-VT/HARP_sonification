from scipy import interpolate
import numpy as np
import datetime

def fgs_interpolate(fgs_gsm, spacing):
    fgs_gsm_Bx = fgs_gsm['BX_FGS-D']
    fgs_gsm_By = fgs_gsm['BY_FGS-D']
    fgs_gsm_Bz = fgs_gsm['BZ_FGS-D']
    fgs_gsm_time = fgs_gsm['UT']
    
    fgs_gsm_epoch = [ii.timestamp() for ii in fgs_gsm_time]
    
    num = int((fgs_gsm_epoch[-1] - fgs_gsm_epoch[0])/spacing)
    fgs_gsm_epoch_itp = fgs_gsm_epoch[0] + np.arange(0,num+1) * spacing
    
    fx0 = interpolate.interp1d(fgs_gsm_epoch, fgs_gsm_Bx, kind='linear',fill_value="extrapolate")
    fy0 = interpolate.interp1d(fgs_gsm_epoch, fgs_gsm_By, kind='linear',fill_value="extrapolate")
    fz0 = interpolate.interp1d(fgs_gsm_epoch, fgs_gsm_Bz, kind='linear',fill_value="extrapolate")
    
    fgs_gsm_time_itp = np.array([datetime.datetime.fromtimestamp(ii) for ii in fgs_gsm_epoch_itp])
    
    return fgs_gsm_time_itp, fgs_gsm_epoch_itp, fx0, fy0, fz0


def position_interpolate(pos_time, pos_x, pos_y, pos_z, fgs_gsm_epoch_itp):
    pos_epoch = [ii.timestamp() for ii in pos_time]
    fx = interpolate.interp1d(pos_epoch, pos_x, kind='linear',fill_value="extrapolate")
    fy = interpolate.interp1d(pos_epoch, pos_y, kind='linear',fill_value="extrapolate")
    fz = interpolate.interp1d(pos_epoch, pos_z, kind='linear',fill_value="extrapolate")
    
    interp_pos_x = fx(fgs_gsm_epoch_itp)
    interp_pos_y = fy(fgs_gsm_epoch_itp)
    interp_pos_z = fz(fgs_gsm_epoch_itp)
    
    return interp_pos_x, interp_pos_y, interp_pos_z

def em_interpolate(peem_time, density, fgs_gsm_epoch_itp, velocity_x, flux_x, flux_y):
    peem_epoch = [ii.timestamp() for ii in peem_time]
    fx1 = interpolate.interp1d(peem_epoch, density, kind='linear', fill_value="extrapolate")
    interp_density = fx1(fgs_gsm_epoch_itp)
    
    fx2 = interpolate.interp1d(peem_epoch, velocity_x, kind='linear',fill_value="extrapolate")
    interp_velocity = fx2(fgs_gsm_epoch_itp)
    
    fx3 = interpolate.interp1d(peem_epoch, flux_x, kind='linear',fill_value="extrapolate")
    fy3 = interpolate.interp1d(peem_epoch, flux_y, kind='linear',fill_value="extrapolate")
    interp_flux_x = fx3(fgs_gsm_epoch_itp)
    interp_flux_y = fy3(fgs_gsm_epoch_itp)
    flux_perp = np.sqrt(interp_flux_x**2+interp_flux_y**2)
    return interp_velocity, interp_density, flux_perp

def linear_interpolate(fgs_gsm_Bx_itp, fgs_gsm_By_itp, fgs_gsm_Bz_itp, sheath_flag_update,
                       fgs_gsm_epoch_itp):
    Bx_no_ms = fgs_gsm_Bx_itp[~sheath_flag_update]
    By_no_ms = fgs_gsm_By_itp[~sheath_flag_update]
    Bz_no_ms = fgs_gsm_Bz_itp[~sheath_flag_update]
    epoch_no_ms = fgs_gsm_epoch_itp[~sheath_flag_update]
    
    fx4 = interpolate.interp1d(epoch_no_ms, Bx_no_ms, kind='linear',fill_value="extrapolate")
    fy4 = interpolate.interp1d(epoch_no_ms, By_no_ms, kind='linear',fill_value="extrapolate")
    fz4 = interpolate.interp1d(epoch_no_ms, Bz_no_ms, kind='linear',fill_value="extrapolate")
    
    Bx_itp = fx4(fgs_gsm_epoch_itp)
    By_itp = fy4(fgs_gsm_epoch_itp)
    Bz_itp = fz4(fgs_gsm_epoch_itp)
    
    return Bx_itp, By_itp, Bz_itp