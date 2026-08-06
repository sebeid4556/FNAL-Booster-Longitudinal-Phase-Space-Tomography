from model_2 import Model, State
from maps_4 import MapGenerator
from TomographyEngine_5 import TomographyEngine

from dataclasses import dataclass
import pandas as pd
import os
import numpy as np
from scipy import constants
from scipy.optimize import brentq, newton

# symbols: ϕΔαηβπδ

DEFAULT_NPASSES = 10

DEFAULT_NPPP = 100
DEFAULT_NITER = 25
DEFAULT_L = 64

# a bunch of constants
π = np.pi
c = constants.speed_of_light
e = constants.elementary_charge # charge of proton
m0 = constants.proton_mass # kg - proton rest mass
E0 = (m0*c**2)/e # proton rest energy

#**************************************************************************
# MachineParamters: set certain machine parameters
#**************************************************************************

@dataclass
class MachineParameters:
    ϕs: float # synchronous phase
    V: float # total RF voltage over a turn
    Es: float # synchronous energy
    h: int = 84 # harmonic number for the Booster
    C: float = 474.2 # Booster circumfrence in meters
    αp: float =  0.03219 # momentum compaction factor for the Booster (assumed constant here)
    Npasses: int = DEFAULT_NPASSES # RF divisor (i.e. evenly distribute the RF kick into [Npasses] sub-kicks)

#**************************************************************************
# Paramters: set reconstruction parameters and hold machine parameters
#**************************************************************************

@dataclass
class Parameters:
    machine: MachineParameters
    L: int = DEFAULT_L # square resolution (i.e. LxL image)
    Nppp: int = DEFAULT_NPPP # number of (test) particles per pixel/cell
    N_iter: int = DEFAULT_NITER # number of reconstruction algorithm iterations

#**************************************************************************
# InputSinogram: loading and various utilities for the input data
#**************************************************************************

class InputSinogram:
    def __init__(self, csv_path, parameters):
        self.path = csv_path
        self.parameters = parameters

        # bring the goods out of the walk-in
        self.data = self._loadCSV(self.path)

        # cook and season the data
        self.data = self.clipAndInterpolate(
            self.data, 
            parameters.machine.ϕs, 
            parameters.machine.Es, 
            Nbins=parameters.L
        )

    def _loadCSV(self, csv_path):
        assert os.path.exists(csv_path) == True, '[Error]: path \"%s\" does not exist'
    
        df = pd.read_csv(csv_path, header=None)
        nrows, ncols = df.shape
        data = df.to_numpy()
    
        return data

    def clipToHalfSynchrotronPeriod(self):
        nrows, _ = self.data.shape
        self.data = self.data[:int(np.floor(nrows/2))]
    
    def clipAndInterpolate(self, proj_data, ϕs, Es, Nbins, plot=False):
        nrows, ncols = proj_data.shape

        #--------------------------------------------------------

        def ω_RF(Es):
            h = self.parameters.machine.h
            C = self.parameters.machine.C
            prefactor = (2*π*h*c)/C
            sqrt = np.sqrt
            return prefactor * sqrt(1 - (E0/Es)**2)
        
        def separatrixZeros(ϕs):
            cos = np.cos
            sin = np.sin
            
            def separatrixZeroExpr(ϕ):
                return cos(π - ϕs) + (π - ϕs)*sin(ϕs) - cos(ϕ) - ϕ*sin(ϕs)
            
            def separatrixZeroExprPrime(ϕ):
                return sin(ϕ) - sin(ϕs)
            
            left = newton(separatrixZeroExpr, fprime=separatrixZeroExprPrime, x0=ϕs-π/2)
            right = newton(separatrixZeroExpr, fprime=separatrixZeroExprPrime, x0=ϕs+π/2)
            return (left, right)

        def interpolateToN(arr, N):
            x = np.arange(len(arr))
            x_new = np.linspace(0, len(arr) - 1, N)
            interpolated = np.interp(x_new, x, arr)
            return interpolated
        
        def normalizeDataProjections(proj):
            return proj / np.sum(proj, axis=1, keepdims=True)

        #--------------------------------------------------------
        
        ω_rf = ω_RF(Es)
        ϕ_left, ϕ_right = separatrixZeros(ϕs)
        
        Δt = 0.2 # ns
        
        t_s = ϕs/ω_rf*1e9
        t_left = ϕ_left/ω_rf*1e9
        t_right = ϕ_right/ω_rf*1e9
        
        t_s_shifted = ((Δt*proj_data.shape[1]) / 2) + t_s
        
        Δt_left = t_left - t_s
        Δt_right = t_right - t_s
        
        Δn_left = Δt_left / Δt
        Δn_right = Δt_right / Δt
        
        n_centered_f = int(np.floor(ncols/2)) # assuming sinogram is centered on ϕs
        #--------------------------------------------------------
        
        n_left_centered = int(np.floor(n_centered_f+Δn_left))
        n_right_centered = int(np.floor(n_centered_f+Δn_right))

        if plot:
            plt.imshow(proj_data, cmap='inferno')
            plt.title('bucket boundaries using center')
            plt.axvline(n_centered_f, color='green')
            plt.axvline(n_left_centered, color='white', linestyle='--')
            plt.axvline(n_right_centered, color='white', linestyle='--')
            plt.show()
    
        clipped_data = proj_data[:, n_left_centered:n_right_centered]
        interpolated_data_centered = np.apply_along_axis(interpolateToN, 1, clipped_data, Nbins)

        if plot:
            plt.imshow(interpolated_data_centered, cmap='inferno')
            plt.title('clipped/interpolated to bucket using center')
            plt.show()
    
        return normalizeDataProjections(interpolated_data_centered)

#**************************************************************************
# BoosterTomography: perform reconstruction
#**************************************************************************

class BoosterTomography:
    def __init__(self, parameters, sinogram):
        self.setParameters(parameters)
        self.setInputSinogram(sinogram)

        self._validateArguments()

    # perform checks to ensure the supplied data is good
    def _validateArguments(self):
        pass

    def getParameters(self):
        return self.parameters
    
    def setParameters(self, new_parameters):
        self.parameters = new_parameters

    def setInputSinogram(self, sinogram):
        self.sinogram = sinogram
        self.N = self.sinogram.data.shape[0]

    # perform the tomographic reconstruction and return the result as a ReconstructionResult object
    def reconstruct(self):
        # create a machine model with current parameter set
        machineModel = Model(
            self.parameters.machine.ϕs, 
            self.parameters.machine.V, 
            self.parameters.machine.Npasses
        )

        num_maps = self.N - 1 # number of transport maps to create
        
        # create tomography engine with current parameter set
        tomoEngine = TomographyEngine(
            machineModel, 
            self.sinogram.data, 
            self.parameters.machine.Es, 
            self.parameters.L, 
            self.parameters.Nppp, 
            1, # number of turns between frames = 1
            num_maps,
            self.parameters.N_iter
        )
    
        return tomoEngine.reconstruct(plot=False, _dbg=False)