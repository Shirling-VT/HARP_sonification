import utils
import interpolations as interp

class ESA():
    def __init__(self):
        self.peem_time = None
        self.density = None 
        self.velocity_x = None
        self.flux_x = None
        self.flux_y = None
    
    def load_data(self, start_time, end_time, probe):
        self.peem_time, self.density, self.velocity_x, self.flux_x, self.flux_y = \
                        utils.load_electron_moments(start_time,end_time,probe)
                        
    def interpolate(self, fgs_gsm_epoch_itp):
        self.interp_velocity, self.interp_density, self.flux_perp = interp.em_interpolate(
            self.peem_time, self.density, fgs_gsm_epoch_itp, self.velocity_x, self.flux_x, self.flux_y)