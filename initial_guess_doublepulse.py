import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


from utilities import center_signal, get_sk_rn, do_fft
import jax




def gaussian(x,a,b,c):
    return a*np.exp(-0.5*(x-c)**2/b**2)
 

def get_double_pulse_amps(trace, sigma=10, init_std=0.001):
    m = np.mean(trace, axis=1)

    x1 = find_peaks(gaussian_filter1d(m,sigma=sigma))[0]
    a0 = m[x1]
    x0 = x1/len(m)

    if len(x1)>3:
        x1 = find_peaks(gaussian_filter1d(m, sigma=2*sigma))[0]
        a0 = m[x1]
        x0 = x1/len(m)

    elif len(x1)<=1:
        print("reduce sigma")


    x = np.linspace(0, 1, len(m))
    y = m
    
    dx0 = x0[2] - x0[0]
    da = (np.max(a0)-np.min(a0))/3
    b_max = 0.5/2.355

    amp = []
    for i in [0,2]:
        a0_min, a0_max = np.maximum(a0[i]-da, 0), a0[i]+da
        x0_min, x0_max = x0[i]-dx0/3, x0[i]+dx0/3
        sol = curve_fit(gaussian, x,y, p0=[a0[i], init_std, x0[i]], 
                        bounds=([a0_min, init_std/2, x0_min], 
                                [a0_max, b_max, x0_max]))

        if i==0:
            sol[0][2] = sol[0][2]+dx0/4
        elif i==2:
            sol[0][2] = sol[0][2]-dx0/4
        
        amp.append(gaussian(x, sol[0][0], sol[0][1], sol[0][2]))
    return np.array(amp)



def get_flat_spectral_phase(time, freq, trace, nonlinear_method, monochromatic=True, sigma=10):
    m = np.mean(trace, axis=0)

    if nonlinear_method=="shg":
        factor = 2
    elif nonlinear_method=="thg":
        factor = 3
    else:
        factor = 1

    if monochromatic==True:
        x1 = np.argmax(gaussian_filter1d(m, sigma=sigma))
        f1 = freq[x1]/factor
        return np.array([2*np.pi*f1*time])

    elif monochromatic==False:
        x1 = find_peaks(gaussian_filter1d(m,sigma=sigma))[0]

        if len(x1)!=2:
            max1 = np.max(x1)
            max2 = np.max(x1[x1!=max1])
            print(f"Didnt find exactly two central frequencies. Found {len(x1)}. Picked the strongest two.")

            f1 = freq[max1]
            f2 = freq[max2]

        else:
            f1 = freq[x1[0]]
            f2 = freq[x1[1]]

        return np.array([2*np.pi*f1*time, 2*np.pi*f2*time])
    


def get_double_pulse_initial_guess(tau_arr, frequency, measured_trace, nonlinear_method, monochromatic_double_pulse=True, sigma=3, init_std=0.001, 
                                   i_want_control=False):
    # find an initial guess for an autocorrelation frog trace
    # works by finding the delays, amplitudes and central frequencies through peak_finder
    # the actual guess is obtained from a gaussian fit
    # doesnt use/work with jax

    assert (nonlinear_method!="shg" and monochromatic_double_pulse==False) and i_want_control==False, "shg with multichromatic pulse doesnt really work"


    sk, rn = get_sk_rn(tau_arr, frequency)

    amp = get_double_pulse_amps(measured_trace, sigma=sigma, init_std=init_std)
    phase = get_flat_spectral_phase(tau_arr, frequency, measured_trace, nonlinear_method, monochromatic=monochromatic_double_pulse, sigma=sigma)

    if len(phase)==1:
        amp_time = np.sum(amp, axis=0)
        pulse_t = amp_time*np.exp(1j*phase[0])

    elif len(phase)==2:
        pulse_t = 0
        for i in range(2):
            p = amp[i]*np.exp(1j*phase[i])
            pulse_t = pulse_t + p
    else:
        print("something went wrong")

    pulse_t = center_signal(pulse_t)
    pulse_t = pulse_t/np.max(np.abs(pulse_t))
    return pulse_t, do_fft(pulse_t, sk, rn)






def make_population_doublepulse(key, population_size, tau_arr, frequency, measured_trace, nonlinear_method, monochromatic_double_pulse=True, sigma=3, 
                                init_std=0.001, i_want_control=False):
    pulse_t, pulse_f = get_double_pulse_initial_guess(tau_arr, frequency, measured_trace, nonlinear_method, monochromatic_double_pulse=monochromatic_double_pulse, 
                                                      sigma=sigma, init_std=init_std, i_want_control=i_want_control)
    
    shape = (population_size, np.size(frequency))
    noise = jax.random.uniform(key, shape, minval=-0.1, maxval=0.1)
    pulse_t = pulse_t + noise
    return jax.numpy.asarray(pulse_t)