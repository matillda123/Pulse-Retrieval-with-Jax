from frog_z_error_gradients import calculate_Z_gradient as calculate_Z_gradient_FROG




def calculate_Z_gradient(signal_t, signal_t_new, pulse_t, pulse_t_shifted, gate_shifted, tau_arr, measurement_info, pulse_or_gate, is_tdp=True):
    """
    Calculates the Z-error gradient with respect to the pulse or the gate-pulse for a given FROG measurement. 
    The gradient is calculated in the frequency domain.

    Args:
        signal_t: jnp.array, the current signal field
        signal_t_new: jnp.array, the current signal field projected onto the measured intensity
        pulse_t: jnp.array, the current guess
        pulse_t_shifted: jnp.array, the current guess translated on the time axis
        gate_shifted: jnp.array, the current gate translated on the time axis
        tau_arr: jnp.array, the delays
        measurement_info: Pytree, contains measurement data and parameters
        pulse_or_gate: str, whether the gradient is calculated with respect to the pulse or the gate-pulse

    Returns:
        jnp.array, the Z-error gradient
    """
    return calculate_Z_gradient_FROG(signal_t, signal_t_new, pulse_t, pulse_t_shifted, gate_shifted, tau_arr, measurement_info, is_tdp)