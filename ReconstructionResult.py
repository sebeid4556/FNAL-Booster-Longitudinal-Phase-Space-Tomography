# imports here
import numpy as np
import matplotlib.pyplot as plt
import os

# symbols: ϕΔαηβπδωϵ

class ReconstructionResult:
    def __init__(self, 
                 D, # 2D binned reconstruction image
                 projections, # forward projections of reconstruction (i.e. reconstruction sinogram)
                 L, # square resolution (i.e. # of cells hori/vert)
                 N_iter, # number of reconstruction iterations
                 Nppp, # number of (test) particles per pixel
                 Es, # synchronous energy at reconstruction
                 ϕs, # synchronous phase at reconstruction
                 ϕ_left, ϕ_right, # left and right zeros of separatrix
                 ΔE_max, # acceptance
                 ω_RF, # RF angular frequency
                 emittance68, # emittance (one standard deviation or 68%)
                 emittance95,
                 emittance99,
                 separatrix_points, # for plotting the separatrix
                 d_history, # (final) discrepancy over iterations,
                 name='' # used for saving figures (ex. 15p0ms_b_03)
                ):
        self.D = D
        self.projections = projections
        self.N = len(self.projections)
        self.L = L
        self.N_iter = N_iter
        self.Nppp = Nppp
        self.Es = Es
        self.ϕs = ϕs
        self.ϕ_left = ϕ_left
        self.ϕ_right = ϕ_right
        self.ΔE_max = ΔE_max
        self.emittance68 = emittance68
        self.emittance95 = emittance95
        self.emittance99 = emittance99
        self.separatrix_points = separatrix_points
        self.d_history = d_history

        self.name = name
        self.save_dir = '.' # default to current working directory
        self.save_filename = name if (name != '') else 'N%d_L%d_Niter%d_Nppp%d_Es%fMeV_phis%fpi' % (
            self.N, self.L, self.N_iter, self.Nppp, self.Es, self.ϕs
        ) # default to this

    # plot reconstruction with sepratrix
    def plot(self, title='', cmap='inferno', save_figure=False):
        if title == '':
            title = 'Reconstruction, ITER=%d' % (self.N_iter)

        extent_frame = [self.ϕ_left, self.ϕ_right, -self.ΔE_max/1e6, self.ΔE_max/1e6]

        im = plt.imshow(self.D, extent=extent_frame, aspect='auto', cmap=cmap)
        
        plt.colorbar(im)
        plt.title(title)
        plt.xlabel('ϕ [rad]')
        plt.ylabel('ΔE [MeV]')
        
        ϕ = np.linspace(self.ϕ_left, self.ϕ_right, 1000)
        plt.plot(ϕ, self.separatrix_points/1e6, color='gray') # top half
        plt.plot(ϕ, -self.separatrix_points/1e6, color='gray') # bottom half

        plt.tight_layout()

        if save_figure:
            save_path = os.path.join(self.save_dir, 'Reconstruction_' + self.save_filename)
            save_path += '.png'
            plt.savefig(save_path)
        
        plt.show()

    # plot the reconstruction sinogram
    def plotSinogram(self, title='', cmap='inferno', save_figure=False):
        if title == '':
            title = 'Reconstruction Sinogram, ITER=%d' % (self.N_iter)
        
        plt.imshow(self.projections, cmap=cmap, extent=[self.ϕ_left, self.ϕ_right, self.N, 0])
        plt.title(title)
        plt.xlabel('ϕ [rad]')
        plt.xticks([self.ϕ_left, self.ϕs, self.ϕ_right], ['ϕ_left', 'ϕs', 'ϕ_right'])
        plt.ylabel('Turn')
        plt.yticks(np.linspace(0, self.N, self.N+1), np.arange(self.N+1))
        plt.colorbar()
        plt.tight_layout()

        if save_figure:
            save_path = os.path.join(self.save_dir, 'Sinogram_' + self.save_filename)
            save_path += '.png'
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()

    def plotDiscrepancy(self, save_figure=False):
        assert len(self.d_history) == (self.N_iter+1)
        plt.plot(np.arange(0, self.N_iter+1), self.d_history)
        plt.grid()
        plt.title('Discrepancy over %d iterations' % (self.N_iter))
        plt.xlabel('Iterations')
        plt.ylabel('Discrepancy')

        if save_figure:
            save_path = os.path.join(self.save_dir, 'Discrepancy_' + self.save_filename)
            save_path += '.png'
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()

    def getDiscrepancy(self):
        return self.d_history[-1]