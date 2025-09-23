from BaseClasses import RetrievePulsesCHIRPSCANwithRealFields
from general_algorithms_chirpscan import DifferentialEvolution as DifferentialEvolutionDSCAN, Evosax as EvosaxDSCAN, LSF as LSFDSCAN, AutoDiff as AutoDiffDSCAN




class DifferentialEvolution(DifferentialEvolutionDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)







class Evosax(EvosaxDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)





class LSF(LSFDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)




class AutoDiff(AutoDiffDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
