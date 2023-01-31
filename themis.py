from ai import cdas


# load magnetic field data from CDAWeb
def load_thm_fgm(start_time, end_time, probe='thd', product='fgs', coord='gsm'):
    fgm = cdas.get_data('sp_phys', probe.upper() + '_L2_FGM',
                        start_time, end_time, [probe + '_' + product + '_' + coord])
    return fgm


# load state/ephemeris data from CDAWeb
def load_thm_state(start_time, end_time, probe='thd', coord='gsm'):
    state_gsm = cdas.get_data('sp_phys', probe.upper() + '_OR_SSC',
                              start_time, end_time, ['XYZ_' + coord.upper()])
    return state_gsm


# load electron moments data from CDAWeb
def load_thm_mom(start_time, end_time, probe='thd', coord='gsm'):
    peem = cdas.get_data('sp_phys', probe.upper() + '_L2_MOM', start_time, end_time,
                         [probe + '_peem_density', probe + '_peem_velocity_' + coord, probe + '_peem_flux'])
    return peem
