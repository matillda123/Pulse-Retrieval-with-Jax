import jax.numpy as jnp

from equinox import tree_at

from src.utilities import MyNamespace, get_sk_rn, do_interpolation_1d, calculate_gate_with_Real_Fields
from src.core.base_classes_methods import RetrievePulsesFROG, RetrievePulsesCHIRPSCAN, RetrievePulses2DSI





# this is meant to be a parent to the general_algorithms for real fields
# needs to come in first position in order for construct trace to override original one.
class RetrievePulsesRealFields:
    """  
    A Base-Class for reconstruction via real fields. This is needed if multiple nonlinear signals are present in the same trace.
    A complex signal does not inherently express difference frequency generation, because complex signals do not possess negative frequencies.
    This attempt can only be used with general solvers, because for classical solvers analytic gradients/hessians are required. 
    Does not inherit from any class. But is supposed to be used via composition of its child classes with solver classes.

    Attributes:
        frequency_exp: jnp.array, the frequencies corresponding to the measured trace
        frequency: jnp.array, the frequencies which are used in the reconstruction

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.measurement_info = self.measurement_info.expand(frequency_exp = self.frequency_exp, 
                                                             fcut = jnp.argmin(jnp.abs(self.frequency)))
        
        fmin, fmax = jnp.min(self.frequency_exp), jnp.max(self.frequency_exp)
        self.transfer_matrix = self.get_nonlinear_transfer_matrix((fmin, fmax), self.measurement_info)
        self.descent_info = self.descent_info.expand(nonlinear_transfer_matrix = self.transfer_matrix)

        self.sk, self.rn = get_sk_rn(self.time, jnp.concatenate((jnp.flip(self.frequency[1:]), self.frequency)))
        self.measurement_info = tree_at(lambda x: x.sk, self.measurement_info, self.sk)
        self.measurement_info = tree_at(lambda x: x.rn, self.measurement_info, self.rn)



    def get_data(self, x_arr, frequency, measured_trace):
        measured_trace = measured_trace/jnp.linalg.norm(measured_trace)

        self.x_arr = jnp.array(x_arr)

        self.frequency_exp = jnp.array(frequency)
        f = jnp.abs(jnp.array(frequency))
        df = jnp.mean(jnp.diff(jnp.array(frequency)))
        self.frequency = jnp.arange(0, jnp.max(f)+df, df)

        self.time = jnp.fft.fftshift(jnp.fft.fftfreq(2*jnp.size(self.frequency)-1, df))
        self.measured_trace = jnp.array(measured_trace)

        return self.x_arr, self.time, self.frequency, self.measured_trace
    

    def get_nonlinear_transfer_matrix(self, parameters, measurement_info):
        frequency = measurement_info.frequency
        fmin, fmax = parameters

        frequency = jnp.concatenate((jnp.flip(-1*frequency[1:]), frequency))

        idxmin, idxmax = jnp.argmin(jnp.abs(frequency-fmin)), jnp.argmin(jnp.abs(frequency-fmax))
        transfer_matrix = jnp.zeros(jnp.size(frequency))
        transfer_matrix = transfer_matrix.at[idxmin:idxmax+1].set(1.0)
        return transfer_matrix
    

    def apply_nonlinear_transfer_matrix(self, signal_f, descent_info):
        return signal_f*descent_info.nonlinear_transfer_matrix



    def construct_trace(self, individual, measurement_info, descent_info):
        x_arr, frequency, trace = super().construct_trace(individual, measurement_info, descent_info)
        trace = self.apply_nonlinear_transfer_matrix(trace, descent_info)

        frequency_exp = measurement_info.frequency_exp
        frequency = jnp.concatenate((jnp.flip(-1*frequency[1:]), frequency))
        trace = do_interpolation_1d(frequency_exp, frequency, trace.T, method="linear").T
        return x_arr, frequency_exp, trace
    



    def make_pulse_f_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        pulse_f = super().make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        p_flip = jnp.flip(pulse_f[1:])
        p_flip = p_flip*jnp.exp(-2j*jnp.angle(p_flip))
        pulse_f = jnp.concatenate((p_flip, pulse_f))
        return pulse_f
    



    def post_process(self, descent_state, error_arr):
        final_result = super().post_process(descent_state, error_arr)

        frequency_exp, frequency = self.measurement_info.frequency_exp, self.measurement_info.frequency
        frequency = jnp.concatenate((jnp.flip(-1*frequency[1:]), frequency))
        trace = final_result.trace
        trace = self.apply_nonlinear_transfer_matrix(trace, self.descent_info)
        trace = do_interpolation_1d(frequency_exp, frequency, trace.T, method="linear").T
        trace = trace/jnp.linalg.norm(trace)

        final_result = final_result.expand(frequency_exp = self.measurement_info.frequency_exp, 
                                           frequency = frequency,
                                           trace = trace)
        return final_result
    







class RetrievePulsesFROGwithRealFields(RetrievePulsesFROG):
    """ 
    Inherits from RetrievePulsesFROG. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




    def calculate_shifted_signal(self, signal, frequency, tau_arr, time, in_axes=(None, 0, None, None, None)):
        frequency = jnp.concatenate((jnp.flip(-1*frequency[1:]), frequency))
        return super().calculate_shifted_signal(signal, frequency, tau_arr, time, in_axes=in_axes)


        
    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        time, frequency = measurement_info.time, measurement_info.frequency
        cross_correlation, doubleblind, ifrog = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.ifrog
        frogmethod = measurement_info.nonlinear_method

        pulse, gate = individual.pulse, individual.gate


        pulse_t_shifted = self.calculate_shifted_signal(pulse, frequency, tau_arr, time)

        if cross_correlation==True:
            gate_pulse_shifted = self.calculate_shifted_signal(measurement_info.gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted = None
            gate_shifted = calculate_gate_with_Real_Fields(pulse_t_shifted, frogmethod)


        if ifrog==True and cross_correlation==False and doubleblind==False:
            signal_t = jnp.real(pulse + pulse_t_shifted)*calculate_gate_with_Real_Fields(pulse + pulse_t_shifted, frogmethod)
        elif ifrog==True:
            signal_t = jnp.real(pulse + gate_pulse_shifted)*calculate_gate_with_Real_Fields(pulse + gate_pulse_shifted, frogmethod)
        else:
            signal_t = jnp.real(pulse)*gate_shifted
            

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted)
        return signal_t





class RetrievePulsesCHIRPSCANwithRealFields(RetrievePulsesCHIRPSCAN):
    """ 
    Inherits from RetrievePulsesCHIRPSCAN. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def calculate_signal_t(self, individual, phase_matrix, measurement_info):
        pulse = individual.pulse

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse, phase_matrix, measurement_info)
        gate_disp = calculate_gate_with_Real_Fields(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = jnp.real(pulse_t_disp)*gate_disp

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_disp = pulse_t_disp, 
                               gate_disp = gate_disp)
        return signal_t







class RetrievePulses2DSIwithRealFields(RetrievePulses2DSI):
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        time, frequency, nonlinear_method = measurement_info.time, measurement_info.frequency, measurement_info.nonlinear_method

        pulse_t = individual.pulse

        if measurement_info.cross_correlation==True:
            gate1, gate2 = measurement_info.anc_1, measurement_info.anc_2

        elif measurement_info.doubleblind==True:
            gate1 = gate2 = individual.gate

        else:
            gate1 = gate2 = self.apply_phase(pulse_t, measurement_info)
            

        gate2_shifted = self.calculate_shifted_signal(gate2, frequency, tau_arr, time)
        gate_pulses = gate1 + gate2_shifted
        gate = calculate_gate_with_Real_Fields(gate_pulses, nonlinear_method)
        signal_t = jnp.real(pulse_t)*gate

        signal_t = MyNamespace(signal_t=signal_t, gate_pulses=gate_pulses, gate=gate)
        return signal_t
