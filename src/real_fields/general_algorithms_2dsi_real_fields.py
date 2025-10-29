from src.core.base_classes_methods import RetrievePulses2DSIwithRealFields, RetrievePulsesRealFields
from ..twodsi.general_algorithms_2dsi import DifferentialEvolution as DifferentialEvolution2DSI, Evosax as Evosax2DSI, LSF as LSF2DSI, AutoDiff as AutoDiff2DSI




class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolution2DSI, RetrievePulses2DSIwithRealFields):
    """ The Differential Evolution Algorithm applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, DifferentialEvolution2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



class Evosax(RetrievePulsesRealFields, Evosax2DSI, RetrievePulses2DSIwithRealFields):
    """ The Evosax package applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, Evosax2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



class LSF(RetrievePulsesRealFields, LSF2DSI, RetrievePulses2DSIwithRealFields):
    """ The LSF Algorithm applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, LSF2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



class AutoDiff(RetrievePulsesRealFields, AutoDiff2DSI, RetrievePulses2DSIwithRealFields):
    """ The Optimistix package applied to 2DSI with real fields. 
    Inherits from  RetrievePulsesRealFields, AutoDiff2DSI and RetrievePulses2DSIwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)
