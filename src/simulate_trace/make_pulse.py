from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import jax
from src.utilities import get_sk_rn, do_fft, do_ifft, MyNamespace, do_interpolation_1d, generate_random_continuous_function, get_com


def check_dataclass_types(self):
    for (name, field) in self.__dataclass_fields__.items():
        field_type = field.type
        if not isinstance(self.__dict__[name], field_type):
            current_type = type(self.__dict__[name])
            raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")

class SpectralAmplitude:
    def __post_init__(self):
        check_dataclass_types(self)
        

class SpectralPhase:
    def __post_init__(self):
        check_dataclass_types(self)


@dataclass
class GaussianAmplitude(SpectralAmplitude): # can be used for single or multiple gaussians
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    fwhm: Union[np.ndarray,tuple,float,int]


@dataclass
class SuperGaussianAmplitude(SpectralAmplitude): # can be used for single or multiple gaussians
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    fwhm: Union[np.ndarray,tuple,float,int]
    p: Union[np.ndarray,tuple,float,int]


@dataclass
class LorentzianAmplitude(SpectralAmplitude): # can be used for single or multiple lorentzians
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    fwhm: Union[np.ndarray,tuple,float,int]


@dataclass
class SuperLorentzianAmplitude(SpectralAmplitude): # can be used for single or multiple lorentzians
    amplitude: Union[np.ndarray,tuple,float,int]
    central_frequency: Union[np.ndarray,tuple,float,int]
    fwhm: Union[np.ndarray,tuple,float,int]
    p: Union[np.ndarray,tuple,int]


@dataclass
class PolynomialPhase(SpectralPhase):
    """ Coefficients are for all taylor-orders -> for a third order pulse, zeros have to be present for the other orders."""
    central_frequency: Union[float,int,None]
    coefficients: Union[np.ndarray,tuple,float,int]


@dataclass
class SinusoidalPhase(SpectralPhase):
    amplitude: Union[np.ndarray,tuple,float,int]
    periodicity: Union[np.ndarray,tuple,float,int]
    phase_shift: Union[np.ndarray,tuple,float,int]


@dataclass
class RandomPhase(SpectralPhase):
    number_of_points: int = 10
    minval_phase: Union[float,int] = -0.5*np.pi
    maxval_phase: Union[float,int] = 0.5*np.pi
    sigma: Union[float,int] = 10




@dataclass
class CustomPulse:
    frequency: np.ndarray
    amplitude: Union[np.ndarray,SpectralAmplitude,list]
    phase: Union[np.ndarray,SpectralPhase]



@dataclass
class MultiPulse:
   """ Constructs/Defines a pulse sequence in the time domain. I should do this in the frequency domain. Is much cleaner. """
   
   delay: Union[np.ndarray,tuple,float,int]
   duration: Union[np.ndarray,tuple,float,int]
   central_frequency: Union[np.ndarray,tuple,float,int]
   amplitude: Union[np.ndarray,tuple,float,int]
   phase: Union[list,tuple]





class MakePulse:
    """ 
    Is supposed to generate pulses based on input parameters from GaussianAmplitude, PolynomialPhase, SinusoidalPhase, RandomPhase, CustomPulse or MultiPulse.
    Does not work with Jax.

    Attributes:
        N (int): the number of points on the time/frequency-grid
        Delta_f (float): the frequency range which is used. Extends from -Delta_f to +Delta_f (so it should actually be named f_max?)

    """
    def __init__(self, N=256, Delta_f=None):
        self.N = N
        self.Delta_f = Delta_f


    def check_and_correct_if_scalar(self, parameters):
        if all([type(j)!=i for i in (int,float) for j in parameters]):
            temp = parameters
        else:
            temp = []
            for param in parameters:
                temp.append(np.squeeze(np.asarray([param])))
        return temp


    def gaussian(self, x, amp, fwhm, shift, p=1):
        #idx = np.argmin(np.abs(x-shift))
        #shift = x[idx] # makes sure that shift lies on top of grid point, prevents issues when fwhm is to small
        return amp*np.exp(-np.log(2)*(4*(x-shift)**2/fwhm**2)**p)
    
    def lorentzian(self, x, amp, fwhm, shift, p=1):
        assert type(p)==int, "things get problematic when the power isnt an integer."
        return amp/(1+(2*(x-shift)/fwhm)**(2*p))
    

    def generate_gaussian_lorentzian_amplitude(self, x, parameters, amp_func):
        amp, fwhm, shift = parameters.amplitude, parameters.fwhm, parameters.central_frequency
        amp, fwhm, shift = self.check_and_correct_if_scalar((amp, fwhm, shift))
        assert len(amp)==len(fwhm)==len(shift)

        a = 0
        for i in range(len(amp)):
            a = a + amp_func(x, amp[i], fwhm[i], shift[i])
        return a
    
    def generate_super_gaussian_lorentzian_amplitude(self, x, parameters, amp_func):
        amp, fwhm, shift, p = parameters.amplitude, parameters.fwhm, parameters.central_frequency, parameters.p
        amp, fwhm, shift, p = self.check_and_correct_if_scalar((amp, fwhm, shift, p))
        assert len(amp)==len(fwhm)==len(shift)==len(p)

        a = 0
        for i in range(len(amp)):
            a = a + amp_func(x, amp[i], fwhm[i], shift[i], p=p[i])
        return a
    

    def generate_polynomial_phase(self, frequency, parameters, amp_f):
        central_frequency, coefficients = parameters.central_frequency, np.squeeze(np.asarray([parameters.coefficients]))
        if central_frequency==None:
            idx_f0 = get_com(amp_f, np.arange(len(self.frequency))).astype(int)
            central_frequency = self.frequency[idx_f0]
        
        n = len(coefficients)
        phase = 0
        for i in range(n):
            phase = phase + coefficients[i]*(2*np.pi*(frequency-central_frequency))**i
        return phase
    
    
    def generate_sinusoidal_phase(self, frequency, phase_parameters):
        a,b,c = phase_parameters.amplitude, phase_parameters.periodicity, phase_parameters.phase_shift
        a,b,c = self.check_and_correct_if_scalar((a,b,c))
        assert len(a)==len(b)==len(c)

        phase = 0
        for i in range(len(a)):
            phase = phase + a[i]*np.sin(2*np.pi*b[i]*frequency-c[i])
        return 2*np.pi*phase
    

    def get_spectral_phase(self, phase_parameters, amp_f):
        if isinstance(phase_parameters, PolynomialPhase):
            phase = self.generate_polynomial_phase(self.frequency, phase_parameters, amp_f)

        elif isinstance(phase_parameters, SinusoidalPhase):
            phase = self.generate_sinusoidal_phase(self.frequency, phase_parameters)

        elif isinstance(phase_parameters, RandomPhase):
            N_points, sigma = phase_parameters.number_of_points, phase_parameters.sigma
            min_val, max_val = phase_parameters.minval_phase, phase_parameters.maxval_phase

            idx = np.where(amp_f>0.01)
            xf = self.frequency[idx[0][0]], self.frequency[idx[0][-1]]
            phase = generate_random_continuous_function(jax.random.PRNGKey(np.random.randint(0,1e9)), N_points, self.frequency, 
                                                        min_val, max_val, gaussian_filter1d(amp_f, sigma=sigma), forced_vals=xf, extrap=True)
            
        elif isinstance(phase_parameters, CustomPulse):
            if isinstance(phase_parameters.phase, SpectralPhase):
                phase = self.get_spectral_phase(phase_parameters.phase, amp_f)
            else:
                phase = do_interpolation_1d(self.frequency, phase_parameters.frequency, phase_parameters.phase)

        else:
            raise NotImplementedError(f"Phase type {phase_parameters} is not implemented.")

        return phase
    


    def get_spectral_amplitude(self, amp_parameters):
        if isinstance(amp_parameters, GaussianAmplitude):
            amp_f = self.generate_gaussian_lorentzian_amplitude(self.frequency, amp_parameters, self.gaussian)
        
        elif isinstance(amp_parameters, SuperGaussianAmplitude):
            amp_f = self.generate_gaussian_lorentzian_amplitude(self.frequency, amp_parameters, self.gaussian)

        elif isinstance(amp_parameters, LorentzianAmplitude):
            amp_f = self.generate_gaussian_lorentzian_amplitude(self.frequency, amp_parameters, self.lorentzian)
        
        elif isinstance(amp_parameters, SuperLorentzianAmplitude):
            amp_f = self.generate_gaussian_lorentzian_amplitude(self.frequency, amp_parameters, self.lorentzian)

        elif isinstance(amp_parameters, MultiPulse):
            print("is broken")
            amp, fwhm, shift = amp_parameters.amplitude, amp_parameters.duration, amp_parameters.delay
            shift = np.concatenate((np.zeros(1), shift))
            shift = shift - np.mean(shift)
            amp_t = self.generate_gaussian_lorentzian_amplitude(self.time, amp, fwhm, shift)
            amp_f = do_fft(amp_t, self.sk, self.rn)
            amp_f = np.abs(amp_f)

        elif isinstance(amp_parameters, CustomPulse):
            if isinstance(amp_parameters.amplitude, SpectralAmplitude):
                amp_f = self.get_spectral_amplitude(amp_parameters.amplitude)
            elif isinstance(amp_parameters.amplitude, list):
                amp_f = 0
                for params in amp_parameters.amplitude:
                    amp_f = amp_f + self.get_spectral_amplitude(params)
            else:
                amp_f = do_interpolation_1d(self.frequency, amp_parameters.frequency, amp_parameters.amplitude)
            
        else:
            raise NotImplementedError(f"Amplitude type {amp_parameters} is not implemented.")

        return amp_f
    
        
        
    def init_generation(self, parameters):
        if isinstance(parameters, tuple)==True:
            amplitude, phase = parameters
            assert isinstance(amplitude, SpectralAmplitude) and isinstance(phase, SpectralPhase)

            if self.Delta_f == None:
                f = amplitude.fwhm + amplitude.central_frequency
                fmax = np.maximum(np.abs(np.min(f)), np.abs(np.max(f)))
                fmin, fmax = -4*fmax, 4*fmax
            else:
                fmin, fmax = -1*self.Delta_f, self.Delta_f

        else:
            assert self.Delta_f != None, "For CustomPulse or MultiPulse you should predefine a good Delta_f."
            fmin, fmax = -1*self.Delta_f, self.Delta_f

            amplitude = parameters
            if isinstance(parameters, MultiPulse):
                phase = parameters.phase # this could be a list of phase_classes
            else:
                phase = parameters


        self.frequency = np.linspace(fmin, fmax, self.N)
        self.df = np.mean(np.diff(self.frequency))
        self.time = np.fft.fftshift(np.fft.fftfreq(self.N, self.df))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        return amplitude, phase



    def generate_pulse(self, parameters):
        amplitude, phase = self.init_generation(parameters)

        amp_f = self.get_spectral_amplitude(amplitude)

        if isinstance(parameters, MultiPulse): 
            shift = np.concatenate((np.zeros(1), parameters.delay))
            shift = shift - np.mean(shift)
        
            temp = 0
            for i in range(len(shift)):
                amp_t = self.gaussian(self.time, parameters.amplitude[i], parameters.duration[i], shift[i])
                pulse_t = amp_t * np.exp(1j*2*np.pi*parameters.central_frequency[i]*self.time)
                pulse_f = do_fft(pulse_t, self.sk, self.rn)

                phase = self.get_spectral_phase(parameters.phase[i], np.abs(pulse_f))
                pulse_f = pulse_f*np.exp(1j*phase)

                temp = temp + pulse_f
            pulse_f = temp

        else:
            phase = self.get_spectral_phase(phase, amp_f)
            pulse_f = amp_f*np.exp(1j*phase)

        pulse_t = do_ifft(pulse_f, self.sk, self.rn)

        self.pulses = MyNamespace(time=self.time, frequency=self.frequency, pulse_t=pulse_t, pulse_f=pulse_f)
        return self.time, pulse_t, self.frequency, pulse_f
    


    def plot_pulses(self):
        fig = plt.figure(figsize=(9,4))

        plt.subplot(1,2,1)
        plt.plot(self.pulses.time, np.abs(self.pulses.pulse_t))
        plt.xlabel("Time [fs]")
        plt.ylabel("Amplitude [arb. u.]")


        plt.subplot(1,2,2)
        plt.plot(self.pulses.frequency, np.abs(self.pulses.pulse_f))
        plt.xlabel("Frequency [pHz]")
        plt.ylabel("Amplitude [arb. u.]")
