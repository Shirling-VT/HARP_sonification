# Package imports
import datetime
import numpy as np

# File imports
import themis as thm
import interpolations as interp
import utils
import ps_utils
import write_utils
import plot_utils

# High level file imports
import GSM
import ESA

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

light_mode = False


def process_data(
        start_time=datetime.datetime(2008, 12, 7), end_time=datetime.datetime(2008, 12, 10), probe='the',
        spacing=3., pos_min=5, stretch=6, samplerate=44100, filetype=None,
        filename_str='Events', stretchMethod='wavelets', process_method='equal_loudness', enable_time_series=True
):
    """Principal data processing code.
    
    :param stretchMethod:
        The method to use for time stretching. Can be 'paulstretch_dBdt' 'wavelets' 'wavelets_dBdt'
        'phaseVocoder' 'phaseVocoder_dBdt' 'wsola' or 'wsola_dBdt'
    """
    # load magnetic field data from CDAWeb
    Mag_data = GSM.Mag()
    Mag_data.load_data(start_time, end_time, probe, product='fgs', coord='gsm')
    Mag_data.despike(dBdt_th=3., num=10)  # despike
    Mag_data.interpolate(spacing, start_time, end_time)  # interpolate the mag data to be evenely spaced time series

    # load state/ephemeris data from CDAWeb
    State_data = GSM.State()
    State_data.load_data(start_time, end_time, probe, coord='gsm')

    # interpolate position data to the fgs time stamps
    State_data.interpolate(Mag_data.fgs_gsm_epoch_itp)

    # save original 1 min resolution position file
    write_utils.write_pos_file(start_time, end_time, State_data.pos_time,
                               State_data.pos_x, State_data.pos_y, State_data.pos_z,
                               probe, filename_str=filename_str)
    print('Write to satellite position file finished!')

    # load electron moments data from CDAWeb for magnetosheath interval identification
    ESA_data = ESA.ESA()
    ESA_data.load_data(start_time, end_time, probe)

    # interpolate ESA data to the fgs time stamps
    ESA_data.interpolate(Mag_data.fgs_gsm_epoch_itp)

    # remove periods when spacecraft is in the magnetosheath
    sheath_flag = (State_data.pos_r > 8) & ((ESA_data.interp_density > 10) |
                                            (ESA_data.interp_velocity < -200) |
                                            (ESA_data.flux_perp > 2e7) |
                                            (ESA_data.interp_density < 0.02))

    # remove magnetosheath data points that are 9 sec on either side of the centered data point
    sheath_flag_update = utils.find_neighbors(sheath_flag, num=3)

    # Linear interpolation to fill “NAN” gaps in the magnetic field time series due to magnetosheath flags
    Mag_data.linear_interpolate(sheath_flag_update)

    # Convert mag data to a pandas dataframe and do running acerage detrend
    df = utils.convert_mag_data(Mag_data.Bx_itp, Mag_data.By_itp, Mag_data.Bz_itp, Mag_data.fgs_gsm_time_itp)

    # Detrending by substracting 30-min running average
    # This will result in nan at the edges, we will replace nan with zeros in the end
    Mag_data.detrend(df, window=int(1800 / spacing))

    # Rotate detrend B into field-aligned coordinates
    dB_phi = utils.detrend_rotate(State_data.interp_pos_x, State_data.interp_pos_y,
                                  State_data.interp_pos_z, Mag_data.detrend_Bx,
                                  Mag_data.detrend_By, Mag_data.detrend_Bz, Mag_data.Bx_SMA,
                                  Mag_data.By_SMA, Mag_data.Bz_SMA)

    # Replace periods when spacecraft position is at r<5 Re with zeros
    dB_phi_zero = utils.replace_periods(State_data.pos_r, dB_phi, pos_min)

    # Time stretching of data
    timeStretchingInputs = (Mag_data.fgs_gsm_time_itp, start_time, end_time, dB_phi_zero, stretch)

    if stretchMethod == 'paulstretch':
        ts_aft_stretch = ps_utils.paul_stretch(
            *timeStretchingInputs, spacing,
            ps_window=512. / 44100, samplerate=samplerate,
            process_method=process_method
        )

    if stretchMethod == 'wavelets':
        ts_aft_stretch = ps_utils.wavelet_stretch(
            *timeStretchingInputs, spacing,
            interpolateBefore=None, interpolateAfter=None,
            scaleLogSpacing=0.12, process_method=process_method,
            samplerate=samplerate
        )

    if stretchMethod == 'phaseVocoder':
        ts_aft_stretch = ps_utils.phaseVocoder_stretch(
            *timeStretchingInputs, spacing,
            frameLength=512, synthesisHop=None, process_method=process_method,
            samplerate=samplerate
        )

    if stretchMethod == 'wsola':
        ts_aft_stretch = ps_utils.WSOLA_stretch(
            *timeStretchingInputs, spacing,
            frameLength=512, synthesisHop=None, process_method=process_method,
            samplerate=samplerate
        )

    # Write sound file
    for ft in filetype:
        # Normalises the data set by the maximum in the interval.
        ts_aft_stretch = ts_aft_stretch / np.max(np.abs(ts_aft_stretch))
        write_utils.write_sound_file(probe, start_time, end_time, stretch,
                                     ts_aft_stretch, samplerate, ft,
                                     filename_str=filename_str, algorithm=stretchMethod)
        print('Write to %s sound file finished!' % (ft))

    # Plot detrended B_phi time series
    if enable_time_series:
        plot_utils.plot_time_series(start_time, end_time, Mag_data.fgs_gsm_time_itp, dB_phi_zero,
                                    probe, ylim=[-20, 20], filename_str=filename_str)
        plot_utils.plot_time_series(start_time, end_time, Mag_data.fgs_gsm_time_itp, dB_phi_zero,
                                    probe, ylim=[-5, 5], filename_str=filename_str)
        print('Plot time series finished!')

    # Plot dB_phi/dt spectra
    fix_cb = True
    dynamic_cb = True
    if light_mode:
        fix_cb = False
        dynamic_cb = True
    plot_utils.plot_spectra(start_time, end_time, Mag_data.fgs_gsm_time_itp,
                            dB_phi_zero, spacing, probe, filename_str=filename_str, fix_cb=fix_cb,
                            dynamic_cb=dynamic_cb)
    print('Plot spectrogram finished!')


start_time = datetime.datetime(2008, 11, 18, 3, 40)
end_time = datetime.datetime(2008, 11, 21, 3, 30)
probe = 'the'

stretchMethods = ['paulstretch', 'wavelets', 'phaseVocoder', 'wsola']
process_methods = ['equal_loudness', 'dBdt_aft_stretch', 'original_ts']
stretchMethod = stretchMethods[1]
process_method = process_methods[0]


def start_from_orbit_file(file: str):
    orbit_file = open(file, 'r')

    all_lines = orbit_file.readlines()

    count_of_orbits = 0

    for line in all_lines:
        count_of_orbits += 1
        start_and_end_dates = line.split(" ")

        if len(start_and_end_dates) < 2:
            print("An error as occurred with the orbit number " + str(count_of_orbits))
        else:
            start_date_as_string = start_and_end_dates[0]
            end_date_as_string = start_and_end_dates[1].replace('\n', str())

            date_format = "%Y-%m-%d/%H:%M:%S"

            start_date = None
            end_date = None

            try:
                start_date = datetime.datetime.strptime(start_date_as_string, date_format)
                end_date = datetime.datetime.strptime(end_date_as_string, date_format)

            except ValueError as error:
                print("An error as occurred with the format of one of the date of the orbit number " +
                      str(count_of_orbits) + " " + str(error))

            if start_date is not None and end_date is not None:
                print("Processing from " + start_date_as_string + " to " + end_date_as_string)

                file_types = ['wav', 'ogg']
                enable_time_series = True
                if light_mode:
                    file_types = []
                    enable_time_series = False

                process_data(start_time=start_date, end_time=end_date, probe=probe, filetype=file_types,
                             filename_str='Dawn_Active_' + stretchMethod + '_' + process_method,
                             stretchMethod=stretchMethod,
                             process_method=process_method, enable_time_series=enable_time_series)

                print("End of processing")

    print("End of the orbit file execution")


def start_from_default():
    process_data(start_time=start_time, end_time=end_time, probe=probe, filetype=['wav', 'ogg'],
                 filename_str='Dawn_Active_' + stretchMethod + '_' + process_method, stretchMethod=stretchMethod,
                 process_method=process_method)


import sys

if len(sys.argv) > 0:
    if "--light" in sys.argv or "--LIGHT" in sys.argv:
        print("Enabling light mode. Only the spectrograms will be generated")
        light_mode = True

    if sys.argv[1].upper() == "--ORBIT_FILE":
        if len(sys.argv) < 2:
            raise Exception("You must specify an orbit file")
        else:
            start_from_orbit_file(sys.argv[2])
    if sys.argv[1].upper() == "--DEFAULT":
        start_from_default()
    if sys.argv[1].upper() == "-HELP":
        print("Args:\n --orbit_file [path] (Executing sonification with orbit date as start date)\n--default (execute "
              "sonification with default values")
else:
    start_from_default()
