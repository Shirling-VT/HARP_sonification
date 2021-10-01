import pandas as pd
import themis as thm
import numpy as np
from sonificationMethods.paulstretch_mono import paulstretch

    
def load_electron_moments(start_time,end_time,probe):
    peem = thm.load_thm_mom(start_time,end_time,probe=probe,coord='gsm')
    peem_time = peem['UT']
    density=peem['N_ELEC_MOM_ESA-'+probe[2].upper()]
    velocity_x=peem['VX_ELEC_GSM_MOM_ESA-'+probe[2].upper()]
    flux_x = peem['FX_ELEC_MOM_ESA-'+probe[2].upper()] 
    flux_y = peem['FY_ELEC_MOM_ESA-'+probe[2].upper()]
    
    return peem_time, density, velocity_x, flux_x, flux_y

#Find neighbor values of a list
def find_neighbors(data,num=3):
    #data is a 1-D list with Tures and Falses, the Tures are the data points that need to be updated
    #num is the number of neighboring data points on either side of the data point that needs to be updated
    updated_data = data.copy() 
    len_arr = len(data)
    for ind, TF in enumerate(data):
            if TF:
                min_ind = max(0,ind-num) #lower boundary cannot be lower than 0
                max_ind = min(ind+num,len_arr) #upper boundary cannot be larger than the length of the array
                updated_data[min_ind:max_ind] = True
    return updated_data

def convert_mag_data(Bx_itp, By_itp, Bz_itp, fgs_gsm_time_itp):
    #Convert mag data to a pandas dataframe and do running acerage detrend
    Bint = np.stack((Bx_itp,By_itp,Bz_itp),axis=-1)
    df = pd.DataFrame(Bint, columns = ['Bx_gsm', 'By_gsm','Bz_gsm'])
    df['datetime']=fgs_gsm_time_itp
    return df

def detrend(df, Bx_itp, By_itp, Bz_itp):
    Bx_SMA = df.iloc[:,0].rolling(window=600, center=True).mean()
    detrend_Bx = Bx_itp - Bx_SMA
    
    By_SMA = df.iloc[:,1].rolling(window=600, center=True).mean()
    detrend_By = By_itp - By_SMA
    
    Bz_SMA = df.iloc[:,2].rolling(window=600, center=True).mean()
    detrend_Bz = Bz_itp - Bz_SMA
    return detrend_Bx, detrend_By, detrend_Bz, Bx_SMA, By_SMA, Bz_SMA

def detrend_rotate(interp_pos_x,interp_pos_y,interp_pos_z,detrend_Bx,detrend_By,detrend_Bz,Bx_SMA,By_SMA,Bz_SMA):
    #Rotate detrend B into field-aligned coordinates
    pos_gsm = np.stack((interp_pos_x,interp_pos_y,interp_pos_z),axis=-1)
    mag_field = np.stack((detrend_Bx,detrend_By,detrend_Bz),axis=-1)
    mean_field = np.stack((Bx_SMA,By_SMA,Bz_SMA),axis=-1)
    B_fac = fac_transformation(mag_field, mean_field, pos_gsm)
    
    dB_phi = B_fac[:,1]
    
    print(mag_field[5000:5010,1])
    print(B_fac[5000:5010,1])
    return dB_phi

def replace_periods(pos_r, dB_phi, pos_min):
    r_flag = pos_r < pos_min
    dB_phi_zero = dB_phi.copy()
    dB_phi_zero[r_flag] = 0
    
    #replace any nan values with zeros
    dB_phi_zero = np.nan_to_num(dB_phi_zero)
    return dB_phi_zero

def fac_transformation(mag_field, mean_field, pos_gsm):
    """
    Transfer to mean field coordinate
    from an input B vector array, mean field array, and a position vector array
    Inputs
    ----------
        mag_field: the B-field data to be transfered into FAC (N*3 2D array)
        mean_field: the mean B-field data, should have the same shape as mag_field 
        pos_gsm: spacecraft position data in GSM coord, should have the same shape as mag_field 
    Returns
    ----------
        The FAC transformation B vector: [pol, tor, com] or [radial,azimuthal,field-aligned]
    """   
    #make mean_field unit vector
    B_mean_amp = mean_field[:,0] ** 2 + mean_field[:,1] ** 2 + mean_field[:,2]**2
    B_mean_amp = B_mean_amp ** (1/2)
    #unit vector in mean field direction
    Uz = mean_field/np.stack((B_mean_amp,B_mean_amp,B_mean_amp),axis=-1)
    
    #unit vector in azimuthal direction
    Uy = np.cross(mean_field,pos_gsm)
    Uy_amp = Uy[:,0] ** 2 + Uy[:,1] ** 2 + Uy[:,2]**2
    Uy_amp = Uy_amp ** (1/2)
    Uy = Uy/np.stack((Uy_amp,Uy_amp,Uy_amp),axis=-1) 
    
    #unit vector in radial direction
    Ux = np.cross(Uy,Uz)
    
    #rotate mag_field in the FAC coord
    fac_field = mag_field.copy()
    fac_field[:,0] = mag_field[:,0]*Ux[:,0] + mag_field[:,1]*Ux[:,1] + mag_field[:,2]*Ux[:,2]
    fac_field[:,1] = mag_field[:,0]*Uy[:,0] + mag_field[:,1]*Uy[:,1] + mag_field[:,2]*Uy[:,2]
    fac_field[:,2] = mag_field[:,0]*Uz[:,0] + mag_field[:,1]*Uz[:,1] + mag_field[:,2]*Uz[:,2]
    
    return fac_field

