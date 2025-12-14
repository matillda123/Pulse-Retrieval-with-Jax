from src.real_fields.base_classes_methods import RetrievePulsesFROGwithRealFields, RetrievePulsesRealFields
from src.frog.general_algorithms_frog import DifferentialEvolution as DifferentialEvolutionFROG, Evosax as EvosaxFROG, LSF as LSFFROG, AutoDiff as AutoDiffFROG
from src.utilities import MyNamespace, do_interpolation_1d
import jax



class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolutionFROG, RetrievePulsesFROGwithRealFields):
    """ The Differential Evolution Algorithm applied to FROG with real fields."""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)



class Evosax(RetrievePulsesRealFields, EvosaxFROG, RetrievePulsesFROGwithRealFields):
    """ The Evosax package applied to FROG with real fields. """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)



class LSF(RetrievePulsesRealFields, LSFFROG, RetrievePulsesFROGwithRealFields):
    """ The LSF Algorithm applied to FROG with real fields. """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)

    
    
    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """
        Real fields LSF does not use make_pulse_f_from_individual()/make_pulse_t_from_individual().
        It stores the fields as complex values instead of amp/phase.
        This function interpolates the individuals onto time_big.
        """
        pulse_t_arr, gate_t_arr = super().get_pulses_from_population(population, measurement_info, descent_info)

        pulse_t_arr, pulse_f_arr = self.interpolate_signal_to_big(pulse_t_arr, measurement_info)
        if measurement_info.doubleblind==True:
            gate_t_arr, gate_f_arr = self.interpolate_signal_to_big(gate_t_arr, measurement_info)
        else:
            gate_t_arr = pulse_t_arr

        return pulse_t_arr, gate_t_arr



    def convert_population(self, population, measurement_info, descent_info):
        """
        convert_population() does use make_pulse_f_from_individual()/make_pulse_t_from_individual().
        Thus the pulses are converted onto time_big. 
        The optimization is likely more efficient when the pulses are defined on time. Thus the interpolation.
        """
        population = super().convert_population(population, measurement_info, descent_info)

        pulse_t_arr, pulse_f_arr = self.interpolate_signal_from_big(population.pulse, measurement_info)
        if measurement_info.doubleblind==True:
            gate_t_arr, gate_f_arr = self.interpolate_signal_from_big(population.gate, measurement_info)
        else:
            gate_t_arr = pulse_t_arr

        return MyNamespace(pulse=pulse_t_arr, gate=gate_t_arr)
    


    def make_population_bisection_search(self, E_arr, population, measurement_info, descent_info, pulse_or_gate):
        """
        For the generation of the signal field the pulses need to be interpolated onto time_big.
        """
        population = super().make_population_bisection_search(E_arr, population, measurement_info, descent_info, pulse_or_gate)

        pulse_t_arr, pulse_f_arr = self.interpolate_signal_to_big(population.pulse, measurement_info)
        if measurement_info.doubleblind==True:
            gate_t_arr, gate_f_arr = self.interpolate_signal_to_big(population.gate, measurement_info)
        else:
            gate_t_arr = pulse_t_arr

        return MyNamespace(pulse=pulse_t_arr, gate=gate_t_arr)
    


    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ Another interpolation onto time_big/frequency_big to calculate the signal field for the final trace. """
        pulse_t, gate_t, pulse_f, gate_f = super().post_process_get_pulse_and_gate(descent_state, measurement_info, descent_info, idx=idx)
        
        pulse_t, pulse_f = self.interpolate_signal_to_big(pulse_t, measurement_info)
        if measurement_info.doubleblind==True:
            gate_t, gate_f = self.interpolate_signal_from_big(gate_t, measurement_info)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f



class AutoDiff(RetrievePulsesRealFields, AutoDiffFROG, RetrievePulsesFROGwithRealFields):
    """ The Optimistix package applied to FROG with real fields."""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
