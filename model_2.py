#=========================================================================================
# Model 2 - test model for accelerating buckets using multiple RF passes
#=========================================================================================
# Description:
# Models accelerating buckets below transition using multiple RF cavities
#
# Assumptions:
# - constant ϕs over duration of simulation
# - constant V
# - multiple RF cavities evenly distributed along the ring (totals up to V)
# - below transition
#
#
#=========================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython import display
import time
from scipy.optimize import brentq, newton # for finding roots of Hamiltonian
import sympy as sp

#=================================================================================================
# Constants
#=================================================================================================

c = 299792458 # m/s
π = np.pi
q = 1.602e-19 # charge of proton
m0 = 1.673e-27 # kg - proton rest mass
E0 = (m0*c**2)/q # proton rest energy

#=================================================================================================
# Bunch State
#=================================================================================================

class State:
    def __init__(self, Es, ϕ, ΔE):
        self.Es = Es
        self.ϕ = ϕ
        self.ΔE = ΔE

        # the machine model is responsible for calculating and setting this value
        self.ΔE_max = None
        # the map generator is reponsible for these ones
        self.ΔX = None
        self.ΔY = None

#=================================================================================================
# Machine model
#=================================================================================================

class Model():
    #=================================================================================================
    # Machine Parameters
    #=================================================================================================
    def __init__(self, ϕs, Vmax, Npasses=1):
        self.NAME = 'Model_2'
        
        self.E_inj = 0.4e9 #0.4 GeV - initial energy at injection from LINAC
        self.η0 = -0.2 # slip factor (changes over turns)
        
        self.h = 84 # harmonic
        self.β0 = 0.7 # speed factor (changes over turns)
        
        self.ϕs = ϕs # sync. phase
        
        #self.Vmax = 250e3 #250 keV
        self.Vmax = Vmax

        #=================================================================================================
        # reduce the original parameters by this amount to make the motion more continuous
        self.REDUCTION_FACTOR = 1
        
        self.η0 = self.η0*self.REDUCTION_FACTOR
        self.Vmax = self.Vmax*self.REDUCTION_FACTOR

        #=================================================================================================

        self.Npasses = Npasses # number of RF passes per turn
        self.Vpass = self.Vmax / self.Npasses # evenly distributed RF cavity voltage
        #print('Vpass = %f kV' % (self.Vpass/1e3))

        self.C = 474.2 # Booster circumfrence [m]
        
        #self.αp = self.η0 + 1 - (self.β0**2) # αp is constant in an ideal accelerator; compute using intiial values given
        self.αp = 0.03219 # use constant momentum compaction factor
        #self.αp = 0.03272 # taken from Jeff's plots
        
        # constants to specify simulation direction
        self.FORWARD = 0
        self.BACKWARD = 1

        self.time_elapsed = 0 # we will calculate the time elapsed using values of Es

        #=================================================================================================

        def separatrixZeroExpr(ϕ):
            ϕs = self.ϕs
            cos = np.cos
            sin = np.sin
            return cos(π - ϕs) + (π - ϕs)*sin(ϕs) - cos(ϕ) - ϕ*sin(ϕs)

        def separatrixZeroExprPrime(ϕ):
            cos = np.cos
            sin = np.sin
            return sin(ϕ) - sin(ϕs)

        # NOTE: these values for (left,right) will only work for below transition
        #self.left = brentq(separatrixZeroExpr, a=-π, b=self.ϕs)
        #self.right = π - self.ϕs
        #self.right = brentq(separatrixZeroExpr, a=self.ϕs, b=π)

        # this seems to work for all ϕs (that have been tested)
        self.left = newton(separatrixZeroExpr, fprime=separatrixZeroExprPrime, x0=self.ϕs-π/2)
        self.right = newton(separatrixZeroExpr, fprime=separatrixZeroExprPrime, x0=self.ϕs+π/2)

    #=================================================================================================
    # Compute various quantities
    #=================================================================================================

    # computes revolution period of the sync. particle for the turn corresponding to its energy
    def T(self, state):
        Es = state.Es
        C = self.C
        v = c*np.sqrt( 1 - (E0/(E0+Es))**2 )
        return C/v
    
    # computes the energy gain from a pass through the RF gap based on the arrival phase ϕ
    def δE_pass(self, ϕ: float):
        V = self.Vpass
        sin = np.sin
        #return self.Vmax*np.sin(ϕ) # [eV]
        return V*sin(ϕ)

    def δE_turn(self, ϕ: float):
        return self.δE_pass(ϕ)*self.Npasses

    # assume Es is the TOTAL RELATIVISTIC ENERGY, not kinetic energy
    def β(self, Es: float) -> float:
        sqrt = np.sqrt
        return sqrt( 1 - (E0**2)/(Es**2) )
    
    # computes η for the n-th turn using β (assuming constant αp)
    def η(self, β: float) -> float:
        return (self.αp - 1 + (β**2)) / self.Npasses
    
    # computes ΔE for the (n+1)-th turn based on ϕs, ϕn, and ΔEn
    def ΔE_next(self, _ΔE_last: float, _ϕ_last: float) -> float:
        ϕs = self.ϕs
        V = self.Vpass
        sin = np.sin
        #return _ΔE_last + self.Vmax*(np.sin(_ϕ_last) - np.sin(self.ϕs)) # [eV]
        return _ΔE_last + V*(sin(_ϕ_last) - sin(ϕs)) # [eV]
    
    # computes ϕ for the (n+1)-th turn based on ΔE and η for the n
    def ϕ_next(self, _ΔE_next: float, _ϕ_last: float, Es: float) -> float:
        βs = self.β(Es) # compute new βs for given Es
        η = self.η(βs)
        #print('η=%f' % η)
        h = self.h
        #E = Es + E0 # total relativistic energy, not just kinetic energy
        #return _ϕ_last + _ΔE_next*((2*π*self.h*self.η(βs))/((βs**2)*Es))
        return _ϕ_last + _ΔE_next*((2*π*h*η)/((βs**2)*Es))
        #return _ϕ_last + _ΔE_next*((2*π*h*η)/((βs**2)*E))

    #==================================================================
    # Turning the machine backwards
    #==================================================================
    def Es_last(self, Es_next):
        V = self.Vmax
        ϕs = self.ϕs
        sin = np.sin
        #return Es_next - self.Vmax*np.sin(self.ϕs)
        return Es_next - V*sin(ϕs)

    def ϕ_last(self, ΔE_next, ϕ_next, Es_last):
        βs = self.β(Es_last) # on last turn
        η = self.η(βs) # on last turn
        h = self.h
        #E_last = Es_last + E0
        return ϕ_next - ( (2*π*h*η) / ((βs**2)*Es_last) )*ΔE_next
        #return ϕ_next - ( (2*π*h*η) / ((βs**2)*E_last) )*ΔE_next
    
    def ΔE_last(self, ΔE_next, ϕ_last):
        V = self.Vpass
        ϕs = self.ϕs
        sin = np.sin
        #return ΔE_next - self.Vmax*(np.sin(ϕ_last) - np.sin(self.ϕs))
        return ΔE_next - V*( sin(ϕ_last) - sin(ϕs) )
    
    #=================================================================================================
    # do N turns - returns new state object
    #=================================================================================================
    def turnOnce(self, state, direction):
        __state = state
        
        Es_n = __state.Es # EDITED (for forward sim)
        Es_last = self.Es_last(__state.Es) # calculate Es on the previous turn (for backward sim)
        
        for i in range(0, self.Npasses):
            if direction == self.FORWARD:
                #self.time_elapsed += self.T(state) # add Trev to time elapsed before passing RF cavity
        
                _ΔE_next = self.ΔE_next(__state.ΔE, __state.ϕ)
                _ϕ_next = self.ϕ_next(_ΔE_next, __state.ϕ, Es_n)
                
                __state = State(Es_n, _ϕ_next, _ΔE_next)
                if i == self.Npasses-1:
                    __state.Es += self.δE_turn(self.ϕs)
            elif direction == self.BACKWARD:
                
                ϕ_last = self.ϕ_last(__state.ΔE, __state.ϕ, Es_last)
                ΔE_last = self.ΔE_last(__state.ΔE, ϕ_last)
                
                __state = State(Es_last, ϕ_last, ΔE_last)
            else:
                assert False, 'invalid simulation direction \'%s\'' % str(direction)
        # it is this machine model's responsibility to set ΔE_max before returning
        __state.ΔE_max = self.getBucketHeightFast(__state.Es) # OPTIMIZED
        return __state
    
    def turnFor(self, state, N, direction):
        for i in range(0, N):
            state = self.turnOnce(state, direction)
        return state

    def outputParameters(self, state):
        Es = state.Es
        β = self.β(Es)
        η = self.η(β)
        V = self.Vmax
        h = self.h
        ϕs = self.ϕs
        ΔE_max = self.getBucketHeightFast(state.Es)
        t_elapsed = self.time_elapsed
        print('='*50)
        print('ϕs=%f (%fπ rad)' % (ϕs, ϕs/π))
        print('Es=%f MeV' % (Es/1e6))
        print('β=%f' % β)
        print('η=%f' % η)
        print('V=%f kV' % (V/1e3))
        print('ΔE_max=%f MeV' % (ΔE_max/1e6))
        #print('t_elapsed=%f ms' % (t_elapsed/1e3))
        print('='*50)

    # returns the synchrotron oscillation period in number of turns
    def n_period(self, state):
        Es = state.Es
        β = self.β(Es)
        η = self.η(β)
        V = self.Vmax
        h = self.h
        ϕs = self.ϕs
        cos = np.cos
        return np.sqrt( (π * β**2 * Es) / (h * V * abs(η * cos(ϕs))) )

    # returns the Hamiltonian for a given set of phase space coordinates (and Es)
    def H(self, Es, ΔE, ϕ):
        β = self.β(Es)
        η = self.η(β)*self.Npasses
        V = self.Vmax
        h = self.h
        ϕs = self.ϕs
        return ((π * h * η)/(β**2 * Es))*(ΔE**2) + V*(np.cos(ϕ) + ϕ*np.sin(ϕs))

    def getBucketHeightFast(self, Es):
        β = self.β(Es)
        η = self.η(β)*self.Npasses # need to multiply by Npasses to get the η for the whole turn
        V = self.Vmax
        h = self.h
        ϕs = self.ϕs
        sqrt = np.sqrt
        cos = np.cos
        sin = np.sin

        prefactor = ((V * β**2 * Es) / (π * h * η))

        ϕ = ϕs # this is where the bucket height is at its maximum
        
        return sqrt( prefactor * (cos(π - ϕs) + (π - ϕs)*sin(ϕs) - cos(ϕ) - ϕ*sin(ϕs)) )

    def ω_RF(self, Es):
        h = self.h
        C = self.C
        prefactor = (2*π*h*c)/C
        sqrt = np.sqrt
        return prefactor * sqrt(1 - (E0/Es)**2)

    def _generateSeparatrixPoints(self, Es):
        β = self.β(Es)
        η = self.η(β)*self.Npasses # need to multiply by Npasses to get the η for the whole turn
        V = self.Vmax
        h = self.h
        ϕs = self.ϕs
        cos = np.cos
        sin = np.sin

        # the prefactor cannot be zero (assuming η != 0), therefore the roots/zeros of this expression are the left and right endpoints of the separatrix
        def separatrixZeroExpr(ϕ):
            return cos(π - ϕs) + (π - ϕs)*sin(ϕs) - cos(ϕ) - ϕ*sin(ϕs)
        
        ϕ = np.linspace(self.left, self.right, 1000)

        prefactor = ((V * β**2 * Es) / (π * h * η))        
        radicand = prefactor * separatrixZeroExpr(ϕ)
        # remove erroneous radicands (likely due to floating point round offs)
        radicand[radicand < 0] = 0

        ΔE = np.sqrt(radicand)
        return ΔE

    def plotSeparatrix(self, state):
        ϕ = np.linspace(self.left, self.right, 1000)
        ΔE = self._generateSeparatrixPoints(state.Es)
        plt.plot(ϕ, ΔE/1e6, color='gray') # top half
        plt.plot(ϕ, -ΔE/1e6, color='gray') # bottom half
    
    def plot(self, state, show=False, fixed=False):
        self.plotSeparatrix(state)
        plt.scatter(state.ϕ, state.ΔE/1e6, s=0.5)
        plt.axvline(self.ϕs, color='black', linestyle='--')
        plt.axhline(0, color='black', linestyle='--')
        if show: 
            if fixed:
                plt.xlim(-π, π)
                plt.ylim(-1, 1)
            else:
                #height = self.getBucketHeightFast(state.Es)/1e6
                assert state.ΔE_max is not None # check just in case
                #print('state.ΔE_max = %f MeV' % (state.ΔE_max/1e6))
                height = state.ΔE_max/1e6 # this should have been be set when state was created
                plt.xlim(self.left, self.right)
                plt.ylim(-height, height)
            plt.title('%s, $\phi_{s}=$%f$\pi$' % (self.NAME, self.ϕs/π))
            plt.xlabel('$\phi$ (rad)')
            plt.ylabel('$\Delta E$ (MeV)')
            plt.grid()
            plt.show()