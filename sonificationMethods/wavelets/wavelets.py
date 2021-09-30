from __future__ import division

import numpy as np
import scipy
import scipy.signal
import scipy.optimize
import scipy.special
from scipy.special import factorial

__all__ = ['Morlet',]


class Morlet(object):
    """ Definition of Morlet wavelet function

    :param int w0:
        Nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.
    """
    def __init__(self, w0=6):
        
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=False):
        """
        Complex Morlet wavelet, centred at zero.

        :param float t:
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        :param float s:
            Scaling factor. Default is 1.
        :param bool complete:
            Whether to use the complete or the standard version.

        :return:
            Value of the Morlet wavelet at the given time

        The standard version:

        .. math::

            \pi^{-0.25} \, \\text{exp}(iwx) \, \\text{exp}(-0.5(x^2))

        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of `w`.

        The complete version:

        .. math::

            \pi^{-0.25} \, ( \\text{exp}(iwx) - \\text{exp}(-0.5(w^2))) \, \\text{exp}(-0.5(x^2))

        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For `w` greater than 5, the
        correction term is negligible.

        Note that the energy of the return wavelet is not normalised
        according to `s`.

        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.
        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** .5)

    def scale_from_period(self, period) -> np.array:
        """Compute the scale from the fourier period.
        """
        # Solve 4 * np.pi * scale / (w0 + (2 + w0 ** 2) ** .5)
        #  for s to obtain this formula
        coeff = np.sqrt(self.w0 * self.w0 + 2)
        return (period * (coeff + self.w0)) / (4. * np.pi)

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Morlet.

        :param float w:
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        :param float s:
            Scaling factor. Default is 1.

        :return:
            Value of the Morlet wavelet at the given frequency
        """
        x = w * s
        # Heaviside mock
        Hw = np.array(w)
        Hw[w <= 0] = 0
        Hw[w > 0] = 1
        return np.pi ** -.25 * Hw * np.exp((-(x - self.w0) ** 2) / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.

        This can be worked out analytically by solving

        .. math::

            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** .5 * s