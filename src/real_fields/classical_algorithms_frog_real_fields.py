import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from equinox import tree_at

from src.real_fields.base_classes_methods import RetrievePulsesRealFields, RetrievePulsesFROGwithRealFields
from src.frog import GeneralizedProjection as GeneralizedProjectionFROG, PtychographicIterativeEngine as PtychographicIterativeEngineFROG, COPRA as COPRAFROG

from src.core.gradients.z_error_gradients_via_AD import calc_Z_grad_AD










class GeneralizedProjection(RetrievePulsesRealFields, GeneralizedProjectionFROG, RetrievePulsesFROGwithRealFields):
    """
    The Generalized Projection Algorithm for FROG.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        pass


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        raise NotImplementedError("The Z-error hessian could be calculated via AD. But thats very expensive.")















class PtychographicIterativeEngine(RetrievePulsesRealFields, PtychographicIterativeEngineFROG, RetrievePulsesFROGwithRealFields):
    """
    The Ptychographic Iterative Engine (PIE) for FROG.

    Attributes:
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, pie_method="rPIE", cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "Dont use interferometric with PIE. its not meant or made for that"

        self.pie_method=pie_method




    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """
        pass




    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        raise NotImplementedError("the hessian could be obtained via AD. But thats very expensive.")
        
    
















class COPRA(RetrievePulsesRealFields, COPRAFROG, RetrievePulsesFROGwithRealFields):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for FROG.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)



    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        pass



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        raise NotImplementedError(f"The z-error hessian could be calculated via AD. But thats very expensive.")

