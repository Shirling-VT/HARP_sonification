#
# Paul's Extreme Sound Stretch (Paulstretch) - Python version
#
# by Nasca Octavian PAUL, Targu Mures, Romania
# http://www.paulnasca.com/
#
# http://hypermammut.sourceforge.net/paulstretch/
#
# this file is released under Public Domain

# Modified version by Marek Cottingham

import numpy as np


def paulstretch(
    audioSample: np.array,
    stretch: float, 
    windowsize_seconds: float, 
    samplerate=44100, 
    enableDebugOutput=False
) -> np.array:
    """ Implementation of paulstretch.

    :param debugOutput:
        Returns a tuple as output, where the first element is the standard output, the second
        element is a 2D numpy array containing the amplitude component of the fft of each window,
        the third is the start index of each window and the fourth is the window function.
    """
    
    smp = audioSample

    #make sure that windowsize is even and larger than 16
    windowsize=int(windowsize_seconds*samplerate)
    if windowsize<16:
        windowsize=16
    windowsize=int(windowsize/2)*2
    half_windowsize=int(windowsize/2)

    #correct the end of the smp
    end_size=int(samplerate*0.05)
    if end_size<16:
        end_size=16
    smp[len(smp)-end_size:len(smp)]*=np.linspace(1,0,end_size)

    
    #compute the displacement inside the input file
    start_pos=0.0
    displace_pos=(windowsize*0.5)/stretch

    #create Hann window
    window=0.5-np.cos(np.arange(windowsize,dtype='float')*2.0*np.pi/(windowsize-1))*0.5

    old_windowed_buf=np.zeros(windowsize)
    hinv_sqrt2=(1+np.sqrt(0.5))*0.5
    hinv_buf=hinv_sqrt2-(1.0-hinv_sqrt2)*np.cos(np.arange(half_windowsize,dtype='float')*2.0*np.pi/half_windowsize)

    finalOutput = []
    debugOutput = []
    intervalStarts = []

    while True:

        #get the windowed buffer
        istart_pos=int(np.floor(start_pos))
        buf=smp[istart_pos:istart_pos+windowsize]
        if len(buf)<windowsize:
            buf=np.append(buf,np.zeros(windowsize-len(buf)))
        buf=buf*window
    
        #get the amplitudes of the frequency components and discard the phases
        freqs=abs(np.fft.rfft(buf))

        debugOutput.append(freqs.copy())
        intervalStarts.append(istart_pos)

        #randomize the phases by multiplication with a random complex number with modulus=1
        ph=np.random.uniform(0,2*np.pi,len(freqs))*1j
        freqs=freqs*np.exp(ph)

        #do the inverse FFT 
        buf=np.fft.irfft(freqs)

        #window again the output buffer
        buf*=window


        #overlap-add the output
        output=buf[0:half_windowsize]+old_windowed_buf[half_windowsize:windowsize]
        old_windowed_buf=buf

        #remove the resulted amplitude modulation
        output*=hinv_buf

        start_pos+=displace_pos
        if start_pos>=len(smp):
            break
        finalOutput.append(output)

    finalOutput = np.concatenate(finalOutput)
    if enableDebugOutput:
        return finalOutput, np.array(debugOutput), np.array(intervalStarts), window
    else:
        return finalOutput



