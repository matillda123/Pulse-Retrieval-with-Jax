import jax.numpy as jnp
from src.utilities import do_fft



def Z_gradient_shg_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    term1 = do_fft(deltaS*jnp.conjugate(gate_pulses), sk, rn)
    term2 = do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    grad = term1 + (spectral_filter1 + spectral_filter2*exp_arr)*term2
    return -2*grad


def Z_gradient_thg_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    term1 = do_fft(deltaS*jnp.conjugate(gate_pulses)**2, sk, rn)
    term2 = do_fft(deltaS*jnp.conjugate(pulse_t*gate_pulses), sk, rn)
    grad = term1 + 2*(spectral_filter1 + spectral_filter2*exp_arr)*term2
    return -2*grad


def Z_gradient_pg_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    term1 = do_fft(deltaS*jnp.abs(gate_pulses)**2, sk, rn)
    term2 = do_fft(jnp.real(jnp.conjugate(deltaS)*(pulse_t*gate_pulses)), sk, rn)
    grad = term1 + 2*(spectral_filter1 + spectral_filter2*exp_arr)*term2
    return -2*grad


def Z_gradient_sd_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    term1 = do_fft(deltaS*gate_pulses**2, sk, rn)
    term2 = do_fft(pulse_t*jnp.conjugate(deltaS*gate_pulses), sk, rn)
    grad = term1 + 2*(spectral_filter1 + spectral_filter2*exp_arr)*term2
    return -2*grad







def Z_gradient_cross_correlation_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    # gradient with respect to pulse, is the same for all nonlinear methods
    grad = do_fft(deltaS*jnp.conjugate(gate), sk, rn)
    return -2*grad





def Z_gradient_shg_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    grad = (spectral_filter1 + spectral_filter2*exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    return -2*grad



def Z_gradient_thg_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    grad = 2*(spectral_filter1 + spectral_filter2*exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t*gate_pulses), sk, rn)
    return -2*grad



def Z_gradient_pg_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    grad = 2*(spectral_filter1 + spectral_filter2*exp_arr)*do_fft(jnp.real(jnp.conjugate(pulse_t)*deltaS)*gate_pulses, sk, rn)
    return -2*grad



def Z_gradient_sd_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2):
    grad = 2*(spectral_filter1 + spectral_filter2*exp_arr)*do_fft(deltaS*pulse_t*jnp.conjugate(gate_pulses), sk, rn)
    return -2*grad








def calculate_Z_gradient_pulse(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    spectral_filter1, spectral_filter2 = jnp.conjugate(measurement_info.spectral_filter1), jnp.conjugate(measurement_info.spectral_filter2)
    nonlinear_method = measurement_info.nonlinear_method

    omega_arr = 2*jnp.pi*frequency
    exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    deltaS = signal_t_new - signal_t


    if measurement_info.doubleblind==True or measurement_info.cross_correlation==True:
        xcorr = True
    else:
        xcorr = False

    grad_func_ac = {"shg": Z_gradient_shg_pulse, "thg": Z_gradient_thg_pulse, 
                    "pg": Z_gradient_pg_pulse, "sd": Z_gradient_sd_pulse}
    
    grad_func_xcorr = {"shg": Z_gradient_cross_correlation_pulse, "thg": Z_gradient_cross_correlation_pulse, 
                       "pg": Z_gradient_cross_correlation_pulse, "sd": Z_gradient_cross_correlation_pulse}
    
    grad_func = {True: grad_func_xcorr,
                 False: grad_func_ac}

    grad = grad_func[xcorr][nonlinear_method](deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2)
    return grad




def calculate_Z_gradient_gate(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    nonlinear_method = measurement_info.nonlinear_method
    spectral_filter1, spectral_filter2 = jnp.conjugate(measurement_info.spectral_filter1), jnp.conjugate(measurement_info.spectral_filter2)

    omega_arr = 2*jnp.pi*frequency
    exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    deltaS = signal_t_new-signal_t

    grad_func = {"shg": Z_gradient_shg_cross_correlation_gate, "thg": Z_gradient_thg_cross_correlation_gate, 
                 "pg": Z_gradient_pg_cross_correlation_gate, "sd": Z_gradient_sd_cross_correlation_gate}

    grad = grad_func[nonlinear_method](deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn, spectral_filter1, spectral_filter2)
    return grad






def calculate_Z_gradient(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info, pulse_or_gate):
    """
    Calculates the Z-error gradient with respect to the pulse or the gate-pulse for a given 2DSI measurement. 
    The gradient is calculated in the frequency domain.

    Args:
        signal_t: jnp.array, the current signal field
        signal_t_new: jnp.array, the current signal field projected onto the measured intensity
        pulse_t: jnp.array, the current guess
        gate_pulses: jnp.array, the current gate-pulse guess
        gate: jnp.array, the current gate
        tau_arr: jnp.array, the delays
        measurement_info: Pytree, contains measurement data and parameters
        pulse_or_gate: str, whether the gradient is calculated with respect to the pulse or the gate-pulse

    Returns:
        jnp.array, the Z-error gradient
    """
        
    calculate_Z_gradient_dict={"pulse": calculate_Z_gradient_pulse,
                               "gate": calculate_Z_gradient_gate}
    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info)