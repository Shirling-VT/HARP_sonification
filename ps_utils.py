from sonificationMethods.paulstretch_mono import paulstretch
from sonificationMethods import wavelets
import numpy as np
from scipy.interpolate import interp1d
import datetime

_audiotsm_tools = None

def get_audiotsm():
    """Utility to only import audiotsm if it is used.
    """
    global _audiotsm_tools
    if _audiotsm_tools is None:
        import audiotsm
        from audiotsm.io.array import ArrayReader, ArrayWriter
        _audiotsm_tools = (audiotsm, ArrayReader, ArrayWriter)
    return _audiotsm_tools

def equal_loudness_normalization(y,samplerate,L_N=40):
    """To obtain equal loudness contours and develop a transfer function 
    (assuming a red noise background) to be applied to time series data so 
    it sounds better to the human ear and removes harsh sounds
    
    Input: 
        y: data to be normalized
        samplerate: sample rate of y
        L_N: phon loudness level of the loaded equal loudness contour, default is 40
    Output: 
        equal loudnss normalized time series of y
    """
    from scipy.interpolate import CubicSpline
    from scipy.fft import fft, fftfreq, fftshift
    from scipy.fft import ifft, ifftshift
    import pydsm

    n=len(y)
    #Load equal loudness contour
    freq, spl = pydsm.iso226.iso226_spl_contour(L_N=L_N)
    #Calculate amplitude transfer function
    h=10.**(spl/20)/ 10.**(spl[0]/20)*freq/freq[0]
    #compute 2-sided FFT of stretched data to give 2-sided freqs and Fourier coefficents
    Fy=fftshift(fft(y))
    fshift = fftshift(fftfreq(y.shape[-1],d=1/samplerate))

    fx = CubicSpline(freq, np.log10(h))#,fill_value="extrapolate"
    h2 = 10**fx(abs(fshift))

    h2[abs(fshift)<20]=1
    y2=ifft(ifftshift(Fy*h2))
    y2_real = y2.real
    return y2_real

#apply paulstretch to a time series 
def thm_fgm_paulstretch(times,data,stretch=6,window=512./44100,samplerate=44100,return_time=False):
# Window for paulstretch is specifed so as to be equivalent to a window of 512 samples when using 
# the default sample rate of 44100
    paulStretch_data = paulstretch(data,stretch,window,samplerate=samplerate)
    
    if return_time == False:
        return paulStretch_data
    else:
        times_interp_dt = _interpolateTimes(times, stretch, paulStretch_data)
        return times_interp_dt,paulStretch_data

def _interpolateTimes(times, stretch, data=None):
    """Interpolates a list of times, decreasing the spacing by ``stretch`` times. Crops
    the list to times to ensure it has the same length as ``data``.
    """
    epochs = [ii.timestamp() for ii in times]
    epoch_stretch = np.linspace(epochs[0],epochs[-1],int(len(times) * stretch))
    if data is not None:
        epoch_stretch = epoch_stretch[:len(data)]
    times_interp_dt = np.array([datetime.datetime.fromtimestamp(ii) for ii in epoch_stretch])
    return times_interp_dt

def paulstretch_dBdt(fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch, spacing,
                     ps_window=512./44100,samplerate = 44100):
    #3-days, factor=6
    
    times, dB_phi_data = _select_data_between_start_and_end_times(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero
    )
    
    paulStretch_dB_phi_zero = thm_fgm_paulstretch(times,dB_phi_data,stretch=stretch,window=ps_window,
                                                  samplerate=samplerate,return_time=False)
    #calculate dB/dt after time stretch 
    stretch_spacing = spacing/stretch
    dB_phi_dt_aft_stretch = np.diff(paulStretch_dB_phi_zero)/stretch_spacing
    return dB_phi_dt_aft_stretch

def _select_data_between_start_and_end_times(fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero):
    time_index = (fgs_gsm_time_itp >= start_time_plot) & (fgs_gsm_time_itp <= end_time_plot)
    times = fgs_gsm_time_itp[time_index]
    dB_phi_data = dB_phi_zero[time_index]
    return times,dB_phi_data

def _waveletPitchShift(
    times,
    data,
    shift=1,
    scaleLogSpacing=0.125,
    interpolateFactor = None,
    maxNumberSamples = 1200,
    wavelet=wavelets.Morlet(),
    preserveScaling=True
) -> None:
    """Pitch shifts the data provided by ``shift`` times using 
    the continous wavlet transform.

    The attributes ``.coefficients`` and ``.coefficients_shifted`` are populated with
    the coefficents produced by the CWT, before and after interpolation.
    The attribute ``scales`` is populated with the scales used for the forward CWT.
    
    :param shift:
        The multiple by which to shift the pitch of the input field.
    :param scaleLogSpacing:
        Scale spacing in log space, a lower value leads to higher
        resolution in frequency space and more processing time.
    :param interpolateFactor:
        If not None, specifies the facator by which the density of the coefficients should be
        increased. Used when generating time stretched audio.
    :param maxNumberSamples:
        The maximum number of samples for the largest scale in the wavelet wavelets.transform, used to
        prevent computations for inaubidle frequencies.
    :param wavelet:
        Wavelet function to use. If none is given, the Morlet wavelet will be used by default.
    :param preserveScaling:
        Whether to preserve the scaling of the data when outputing.
    """
    sampleSeperation = (np.max(times)-np.min(times))/len(times)
    sampleSeperation = sampleSeperation.total_seconds()
    
    scales = wavelets.transform.generateCwtScales(
        maxNumberSamples,
        len(data),
        scaleLogSpacing,
        sampleSeperation,
        wavelet,
    )
    coefficients = wavelets.transform.cwt(data,scales,sampleSeperation,wavelet)

    # Rescale the coefficients as in
    #   A Wavelet-based Pitch-shifting Method, Alexander G. Sklar
    #   https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.5079&rep=rep1&type=pdf

    magnitude = np.abs(coefficients)
    phase = np.unwrap(np.angle(coefficients),axis=1)

    if interpolateFactor is not None:
        magnitude, phase = wavelets.transform.interpolateCoeffsPolar(
            magnitude,phase,interpolateFactor
        )

    coefficients_shifted = magnitude * np.exp(1j * phase * shift)

    # Scaling constants are generally redudant if generating audio as data will be normalised
    if preserveScaling:
        rx = wavelets.transform.icwt(
            coefficients_shifted,
            scaleLogSpacing,
            sampleSeperation,
            wavelet.C_d,
            wavelet.time(0)
        )
    else:
        rx = wavelets.transform.icwt(coefficients_shifted)
    return np.real(rx)

def _wavelet_stretch(times,data,stretch=6,interpolateBefore=None,interpolateAfter=None,scaleLogSpacing=0.12) -> np.ndarray:
    """Time stretches the data using wavelet transforms.
    
    :param stretch:
        The factor by which to stretch the data.
    :param interpolateBefore:
        Interpolation factor prior to forward CWT.
    :param interpolateAfter:
        Interpolation factor after the forward CWT. Default is ``stretch`` if both
        ``interpolateBefore`` and ``interpolateAfter`` are ``None``.
    :param scaleLogSpacing:
        Spacing between scales for the CWT. Lower values improve frequency resolution
        at the cost of increasing computation time.
    """
    if interpolateBefore is None and interpolateAfter is None:
        interpolateAfter = stretch
        newTimes = times
    if interpolateBefore is not None:
        newTimes = _interpolateTimes(times,interpolateBefore)
        fd = interp1d(times,data,kind="cubic",fill_value="extrapolate")
        data = fd(newTimes)
    return _waveletPitchShift(newTimes,data,stretch,scaleLogSpacing,interpolateAfter)

def wavelet_stretch(
    fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch,
    interpolateBefore=None,interpolateAfter=None,scaleLogSpacing=0.12
):
    """Extracts data in the specified time window and performs a wavelet time stretch
    on the data.
    """
    times, dB_phi_data = _select_data_between_start_and_end_times(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero
    )

    wavelets_dB_phi_zero = _wavelet_stretch(
        times,dB_phi_data,stretch,interpolateBefore,interpolateAfter,scaleLogSpacing
    )

    return wavelets_dB_phi_zero

def wavelet_stretch_dBdt(
    fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch, spacing,
    interpolateBefore=None,interpolateAfter=None,scaleLogSpacing=0.12
):  
    """Extracts data in the specified time window, performs a wavelet time stretch
    on the data, finally, spectrally whitens the data using a time-wise difference (diff).
    """
    wavelets_dB_phi_zero = wavelet_stretch(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch,
        interpolateBefore,interpolateAfter,scaleLogSpacing
    )
    stretch_spacing = spacing/stretch
    dB_phi_dt_aft_stretch = np.diff(wavelets_dB_phi_zero)/stretch_spacing
    return dB_phi_dt_aft_stretch

def _phaseVocoder_stretch(times,data,stretch,frameLength=512,synthesisHop=None) -> None:
        """Time stretches the data using a phase vocoder
        
        See also: `audiotsm.phasevocoder <https://audiotsm.readthedocs.io/en/latest/tsm.html#audiotsm.phasevocoder>`_

        :param frameLength: the length of the frames
        :type frameLength: int
        :param synthesisHop: 
            the number of samples between two consecutive synthesis frames (``frameLength // 16`` by default).
        :type synthesisHop: int

        .. note::

            Some samples may be clipped at the end of the data set.
        """
        audiotsm, ArrayReader, ArrayWriter = get_audiotsm()

        if synthesisHop is None:
            synthesisHop = frameLength//16
        reader = ArrayReader(np.array((data,)))
        writer = ArrayWriter(reader.channels)
        timeSeriesModification = audiotsm.phasevocoder(
            reader.channels,
            speed = 1/stretch,
            frame_length=frameLength,
            synthesis_hop=synthesisHop,
        )
        timeSeriesModification.run(reader, writer)
        return writer.data.flatten()

def phaseVocoder_stretch(
    fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch,
    frameLength=512,synthesisHop=None
):
    """Extracts data in the specified time window and performs a phase vocoder time stretch
    on the data.
    """
    times, dB_phi_data = _select_data_between_start_and_end_times(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero
    )

    phaseVocoder_dB_phi_zero = _phaseVocoder_stretch(
        times,dB_phi_data,stretch,frameLength,synthesisHop
    )

    return phaseVocoder_dB_phi_zero

def phaseVocoder_stretch_dBdt(
    fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch, spacing,
    frameLength=512,synthesisHop=None
):  
    """Extracts data in the specified time window, performs a phase-vocoder time stretch
    on the data, finally, spectrally whitens the data using a time-wise difference (diff).
    """
    phaseVocoder_dB_phi_zero = phaseVocoder_stretch(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch,
        frameLength,synthesisHop
    )
    stretch_spacing = spacing/stretch
    dB_phi_dt_aft_stretch = np.diff(phaseVocoder_dB_phi_zero)/stretch_spacing
    return dB_phi_dt_aft_stretch

def _wsolaStretch(times,data,stretch,frameLength=512,synthesisHop=None,tolerance=None) -> None:
        """Time stretches the data using WSOLA

        See also: `audiotsm.wsola <https://audiotsm.readthedocs.io/en/latest/tsm.html#audiotsm.wsola>`_

        :param frame_length: the length of the frames
        :type frame_length: int
        :param synthesis_hop: 
            the number of samples between two consecutive synthesis frames (``frame_length // 8`` by default).
        :type synthesis_hop: int

        .. note::

            Some samples may be clipped at the end of the data set.
        """
        audiotsm, ArrayReader, ArrayWriter = get_audiotsm()

        if synthesisHop is None:
            synthesisHop = frameLength//8
        reader = ArrayReader(np.array((data,)))
        writer = ArrayWriter(reader.channels)
        timeSeriesModification = audiotsm.wsola(
            reader.channels,
            speed = 1/stretch,
            frame_length=frameLength,
            synthesis_hop=synthesisHop,
            tolerance=tolerance
        )
        timeSeriesModification.run(reader, writer)
        return writer.data.flatten()

def WSOLA_stretch(
    fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch,
    frameLength=512,synthesisHop=None,tolerance=None
):
    """Extracts data in the specified time window and performs a WOLSA time stretch
    on the data.
    """
    times, dB_phi_data = _select_data_between_start_and_end_times(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero
    )

    WSOLA_dB_phi_zero = _wsolaStretch(
        times,dB_phi_data,stretch,frameLength,synthesisHop,tolerance
    )

    return WSOLA_dB_phi_zero

def WSOLA_stretch_dBdt(
    fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch, spacing,
    frameLength=512,synthesisHop=None,tolerance=None
):  
    """Extracts data in the specified time window, performs a WSOLA time stretch
    on the data, finally, spectrally whitens the data using a time-wise difference (diff).
    """
    wsola_dB_phi_zero = WSOLA_stretch(
        fgs_gsm_time_itp, start_time_plot, end_time_plot, dB_phi_zero, stretch,
        frameLength,synthesisHop
    )
    stretch_spacing = spacing/stretch
    dB_phi_dt_aft_stretch = np.diff(wsola_dB_phi_zero)/stretch_spacing
    return dB_phi_dt_aft_stretch
