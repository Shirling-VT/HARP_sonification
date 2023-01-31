import themis as thm
import interpolations as interp
import numpy as np
import utils


class Mag:
    def __init__(self):
        self.fgs_gsm = None
        self.fgs_gsm_time_itp = None
        self.fgs_gsm_epoch_itp = None
        self.fx0 = None
        self.fy0 = None
        self.fz0 = None
        self.fgs_gsm_Bx_itp = None
        self.fgs_gsm_By_itp = None
        self.fgs_gsm_Bz_itp = None
        self.Bx_itp = None
        self.By_itp = None
        self.Bz_itp = None
        self.detrend_Bx = None
        self.detrend_By = None
        self.detrend_Bz = None
        self.Bx_SMA = None
        self.By_SMA = None
        self.Bz_SMA = None

    def load_data(self, start_time, end_time, probe, product, coord):
        self.fgs_gsm = thm.load_thm_fgm(start_time, end_time, probe=probe, product=product, coord=coord)

    def despike(self, dBdt_th=5., num=10):
        fgs_gsm_Bx = self.fgs_gsm['BX_FGS-D']
        fgs_gsm_By = self.fgs_gsm['BY_FGS-D']
        fgs_gsm_Bz = self.fgs_gsm['BZ_FGS-D']
        fgs_gsm_time = self.fgs_gsm['UT']

        Bdt = [(t2 - t1).total_seconds() for t1, t2 in zip(fgs_gsm_time[:-1], fgs_gsm_time[1:])]
        dBxdt = np.diff(fgs_gsm_Bx) / Bdt
        dBydt = np.diff(fgs_gsm_By) / Bdt
        dBzdt = np.diff(fgs_gsm_Bz) / Bdt

        spike_flag1 = np.absolute(dBxdt) > dBdt_th
        flag1 = np.append(spike_flag1, [False])
        flag1_nb = utils.find_neighbors(flag1, num=num)

        spike_flag2 = np.absolute(dBydt) > dBdt_th
        flag2 = np.append(spike_flag2, [False])
        flag2_nb = utils.find_neighbors(flag2, num=num)

        spike_flag3 = np.absolute(dBzdt) > dBdt_th
        flag3 = np.append(spike_flag3, [False])
        flag3_nb = utils.find_neighbors(flag3, num=num)

        spike_flags = flag1_nb + flag2_nb + flag3_nb
        self.fgs_gsm['UT'] = fgs_gsm_time[~spike_flags]
        self.fgs_gsm['BX_FGS-D'] = fgs_gsm_Bx[~spike_flags]
        self.fgs_gsm['BY_FGS-D'] = fgs_gsm_By[~spike_flags]
        self.fgs_gsm['BZ_FGS-D'] = fgs_gsm_Bz[~spike_flags]

    def interpolate(self, spacing, start_time, end_time):
        self.fgs_gsm_time_itp, self.fgs_gsm_epoch_itp, self.fx0, self.fy0, self.fz0 \
            = interp.fgs_interpolate(self.fgs_gsm, spacing, start_time, end_time)

        self.fgs_gsm_Bx_itp = self.fx0(self.fgs_gsm_epoch_itp)
        self.fgs_gsm_By_itp = self.fy0(self.fgs_gsm_epoch_itp)
        self.fgs_gsm_Bz_itp = self.fz0(self.fgs_gsm_epoch_itp)

    def linear_interpolate(self, sheath_flag_update):
        self.Bx_itp, self.By_itp, self.Bz_itp = interp.linear_interpolate(
            self.fgs_gsm_Bx_itp, self.fgs_gsm_By_itp,
            self.fgs_gsm_Bz_itp, sheath_flag_update,
            self.fgs_gsm_epoch_itp)

    def detrend(self, df, window=600):
        self.detrend_Bx, self.detrend_By, self.detrend_Bz, self.Bx_SMA, self.By_SMA, self.Bz_SMA \
            = utils.detrend(df, self.Bx_itp, self.By_itp, self.Bz_itp, window=window)


class State():
    def __init__(self):
        self.state_gsm = None
        self.pos_x = None
        self.pos_y = None
        self.pos_z = None
        self.pos_time = None
        self.pos_r = None
        self.interp_pos_x = None
        self.interp_pos_y = None
        self.interp_pos_z = None

    def load_data(self, start_time, end_time, probe, coord):
        self.state_gsm = thm.load_thm_state(start_time, end_time, probe=probe, coord=coord)
        self.pos_x = self.state_gsm['X']  # unit in Earth radii
        self.pos_y = self.state_gsm['Y']
        self.pos_z = self.state_gsm['Z']
        self.pos_time = self.state_gsm['EPOCH']

    def interpolate(self, fgs_gsm_epoch_itp):
        self.interp_pos_x, self.interp_pos_y, self.interp_pos_z = interp.position_interpolate(
            self.pos_time, self.pos_x, self.pos_y,
            self.pos_z, fgs_gsm_epoch_itp)
        self.pos_r = np.sqrt(self.interp_pos_x ** 2 + self.interp_pos_y ** 2 + self.interp_pos_z ** 2)
