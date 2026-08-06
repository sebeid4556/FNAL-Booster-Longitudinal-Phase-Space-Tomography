from model_2 import *
from maps_4 import *
from ReconstructionResult import *

import matplotlib.colors as colors

# TomographyEngine:
# input: data projections, machine parameters, maximum acceptable discrepancy (i.e. halt condition)
# output: reconstruction

class TomographyEngine:
    def __init__(self, machineModel, proj_data, Es_recon, RESOLUTION, Nppp, N, FRAMES, ITERATIONS):
        self.machineModel = machineModel

        #self.CMAP = 'viridis'
        #self.CMAP = 'hot'
        #self.CMAP = 'hot_r'
        #self.CMAP = 'plasma'
        #self.CMAP = 'magma' # yuck
        #self.CMAP = 'cividis'
        self.CMAP = 'inferno'
        #self.CMAP = 'bone'
        #self.CMAP = 'seismic'

        self.RESOLUTION = RESOLUTION
        self.Nppp = Nppp
        self.N = N
        self.FRAMES = FRAMES
        self.ITERATIONS = ITERATIONS

        self.Np = 16**2**2 # only used for generating test distribution

        self.λ = 1 # relaxation factor

        self.Es_recon = Es_recon # the energy at the reconstruction frame

        self.mapsForward = MapGenerator(self.machineModel, \
                                        self.Es_recon, \
                                        self.RESOLUTION, \
                                        self.Nppp, \
                                        self.N, \
                                        self.FRAMES, \
                                        MapGenerator.FORWARD)

        self.mapsBackward = MapGenerator(self.machineModel, \
                                        self.Es_recon, \
                                        self.RESOLUTION, \
                                        self.Nppp, \
                                        self.N, \
                                        self.FRAMES, \
                                        MapGenerator.BACKWARD)

        #========================================================================
        # Build maps and BP matrices
        #========================================================================
        self.mapsForward.createAllFrameMaps()
        self.mapsForward.createAllFramesBackProjectionMatrices()
    
        self.mapsBackward.createAllFrameMaps()
        self.mapsBackward.createAllFramesBackProjectionMatrices()

        #========================================================================
        # Use test data if no proj_data is supplied
        #========================================================================
        if proj_data is None:
            self.proj_data = self._generateTestDataProjections()
            self.plotSinogram(self.proj_data, 'Generated Test Data')
            #self.proj_data = self._generateDensityGradientTestDataProjections()
        else:
            self.proj_data = proj_data

    def _generateDistribution(self, turns, shift=False):
        #state = State(self.machineModel.E_inj + E0, np.zeros(self.Np), np.zeros(self.Np))
        state = State(self.Es_recon, np.zeros(self.Np), np.zeros(self.Np))
        state.ΔE_max = self.machineModel.getBucketHeightFast(state.Es)
        state.ΔX = self.mapsForward.ΔX_0
        state.ΔY = (2*state.ΔE_max)/self.RESOLUTION
        #return state # test no guess
    
        # place particles uniformly in a rectangle
        N_side = int(np.sqrt(self.Np))
        for i in range(0, N_side): state.ΔE[i*N_side:(i+1)*N_side] = np.linspace(-1*state.ΔE_max/2, state.ΔE_max/2, num=N_side)[i]
        if shift:         
            # use these ones for testing - generate one rectangular distribution
            #left = self.machineModel.ϕs - abs(self.machineModel.ϕs - self.machineModel.left)/2
            left = self.machineModel.ϕs
            right = self.machineModel.ϕs + abs(self.machineModel.ϕs - self.machineModel.right)/2
            for i in range(0, self.Np): state.ϕ[i] = np.linspace(left, right, num=N_side)[i%N_side]

            # do two spiraling distributions
            '''left = self.machineModel.ϕs - abs(self.machineModel.ϕs - self.machineModel.left)/2
            right = self.machineModel.ϕs - abs(self.machineModel.ϕs - self.machineModel.left)/4
            for i in range(0, int(self.Np/2)): state.ϕ[i] = np.linspace(left, right, num=N_side)[i%N_side]
            left = self.machineModel.ϕs + abs(self.machineModel.ϕs - self.machineModel.right)/4
            right = self.machineModel.ϕs + abs(self.machineModel.ϕs - self.machineModel.right)/2
            for i in range(int(self.Np/2), self.Np): state.ϕ[i] = np.linspace(left, right, num=N_side)[i%N_side]'''
        else:
            for i in range(0, self.Np): state.ϕ[i] = np.linspace(-π/2, π/2, num=N_side)[i%N_side] # bad
    
        #self.machineModel.plot(state, show=True)
        
        state = self.machineModel.turnFor(state, turns, self.machineModel.FORWARD) # turnFor() sets state.ΔE_max
        state.ΔX = self.mapsForward.ΔX_0
        state.ΔY = (2*state.ΔE_max)/self.mapsForward.RESOLUTION
    
        return state

    def _createUniformDistribution(self, frame):
        P_uni = np.ones(self.RESOLUTION)
        D_uni = self.mapsForward.getBackProjection(P_uni, frame)
        D_uni[D_uni > 0] = 1 # set all pixels inside bucket to 1
        D_uni = self.mapsForward.normalizeBinnedDistribution(D_uni) # normalize -> uniform
        return D_uni

    def _generateTestDataProjections(self):
        # right now we only have 10 frames, so turns has to be 10 so that the uniform distribution is also at frame 10
        #state = self._generateDistribution(turns=100, shift=True)
        state = self._generateDistribution(turns=0, shift=True)
        #self.Es_recon = state.Es

        #print('_generateTestDataProjections(): state.ΔE_max = %f' % state.ΔE_max)
        #print('_generateTestDataProjections(): state.Es = %f' % state.Es)

        #self.machineModel.plot(state, show=True)

        #D_0 = self.mapsForward.getBinnedProbabilityDistributionMap(state)
        #print('_generateTestDataProjections(): D_0, _ = self.mapsForward.getBinnedDistributionMap(state)')
        #D_0, _ = self.mapsForward.getBinnedDistributionMap(state)
        #print('_generateTestDataProjections(): Done.')
        #plt.imshow(D_0)
        #plt.title('DEBUG')
        #plt.show()
        #D_0 = self.mapsForward.getBinnedProbabilityDistributionMap(state)
        
        proj = []
        for i in range(0, self.FRAMES+1):
            #D_i, _ = mapsForward.getBinnedDistributionMap(state)
            D_i = self.mapsForward.getBinnedProbabilityDistributionMap(state)

            # debug
            if i == 0:
                '''plt.imshow(D_i, cmap=self.CMAP)
                plt.title('Test Data, frame=%d'%(i))
                plt.show()'''
                self.plotBinnedDistributionWithSeparatrix(self.Es_recon, \
                                                  D_i, \
                                                  title='Test Data, frame=%d, Es=%f MeV' % (i, state.Es/1e6))
            
            #print('D_i = %s' % str(D_i))
            proj_i = self.mapsForward.getBinnedProjection(D_i)
            #print('proj_data[%2d] = %s' % (i, str(proj_i)))
            #print('Data: D_i.sum() = %f' % D_i.sum())
            proj.append(proj_i)
            '''if i == 0:
                #plt.imshow(D_i, cmap=self.CMAP)
                #plt.title('Test Data, frame=%d'%(i))
                #plt.show()
                self.plotBinnedDistributionWithSeparatrix(self.machineModel.E_inj + E0, \
                                                  D_i, \
                                                  title='Test Data, frame=%d' % (i))'''
            state = self.machineModel.turnFor(state, self.N, self.machineModel.FORWARD)
            state.ΔX = self.mapsForward.ΔX_0
            state.ΔY = (2*state.ΔE_max)/self.RESOLUTION
        assert len(proj) == (self.FRAMES + 1) # +1 including the reconstruction frame
        return np.array(proj)

    #=============================================================================================
    # DEBUG
    #=============================================================================================

    # create an initial guess that is the same for ALL sync phases 
    # (i.e. uniformly distribute particles filling the whole frame)
    def dbgGenerateGuess(self):
        '''NUM_PARTICLES = self.Nppp * (self.RESOLUTION**2)
        state = State(self.Es_recon, np.zeros(NUM_PARTICLES), np.zeros(NUM_PARTICLES))
        state.ΔE_max = self.machineModel.getBucketHeightFast(state.Es)

        for y_abs in range(0, self.RESOLUTION):
            for x_abs in range(0, self.RESOLUTION):
                y_rel, x_rel = self.mapsForward._absCoordsToRelCoords(y_abs, x_abs)
                state = self.mapsForward.placeTestParticlesFast(state, y_rel, x_rel)

        return self.mapsForward.getBinnedProbabilityDistributionMap(state)'''
        return np.ones((self.RESOLUTION, self.RESOLUTION))/(self.RESOLUTION**2)

    def dbgGenerateGuessProjections(self, D_guess):
        proj = []

        for i in range(0, self.FRAMES+1):
            if i == 0:
                D_i = D_guess
                Es_frame = self.Es_recon
            else:
                D_i = self.mapsForward.getNextFrame(i-1, D_i)
                D_i = self.mapsForward.normalizeBinnedDistribution(D_i)
            proj_i = self.mapsForward.getBinnedProjection(D_i)
            proj.append(proj_i)
            
        assert len(proj) == (self.FRAMES + 1) # +1 including the reconstruction frame
        return np.array(proj)

    #=============================================================================================
    #
    #=============================================================================================

    def _generateDensityGradientTestDataProjections(self):
        num_blocks = 6
        state = State(self.machineModel.E_inj + E0, np.zeros(self.Np*num_blocks), np.zeros(self.Np*num_blocks))
        state.ΔE_max = self.machineModel.getBucketHeightFast(state.Es)
        ΔE_max = state.ΔE_max
    
        div = 4
    
        # Each block has same ΔE range but narrower/wider phi range
        ϕ_divs = [2, 2.5, 3, 3.5, 4, 4.5]
    
        N_side = int(np.sqrt(self.Np))
        assert N_side**2 == self.Np
    
        # Same ΔE values for every block
        ΔE_vals = np.linspace(-ΔE_max/div, ΔE_max/div, num=N_side)
    
        # This creates the row-wise ΔE layout:
        # [ΔE_0 repeated N_side times, ΔE_1 repeated N_side times, ...]
        ΔE_block = np.repeat(ΔE_vals, N_side)
    
        for k, ϕ_div in enumerate(ϕ_divs):
            start = k * self.Np
            end = (k + 1) * self.Np
    
            ϕ_vals = np.linspace(-π/ϕ_div, π/ϕ_div, num=N_side)
    
            # This creates the column-wise phi layout:
            # [ϕ_0, ϕ_1, ..., ϕ_N, ϕ_0, ϕ_1, ..., ϕ_N, ...]
            ϕ_block = np.tile(ϕ_vals, N_side)
    
            state.ΔE[start:end] = ΔE_block
            state.ϕ[start:end] = ϕ_block
    
        #state = self.machineModel.turnFor(state, 200, self.machineModel.FORWARD)
        state = self.machineModel.turnFor(state, 35, self.machineModel.FORWARD)
        state.ΔE_max = self.machineModel.getBucketHeightFast(state)
        state.ΔX = self.mapsForward.ΔX_0
        state.ΔY = (2*state.ΔE_max)/self.RESOLUTION

        #self.machineModel.plot(state, show=True)
    
        proj = []
    
        for i in range(0, self.FRAMES + 1):
            D_i = self.mapsForward.getBinnedProbabilityDistributionMap(state)
            proj_i = self.mapsForward.getBinnedProjection(D_i)
            proj.append(proj_i)
    
            if i == 0:
                '''plt.imshow(D_i, cmap=self.CMAP)
                plt.title('Test Data, frame=%d' % i)
                plt.show()'''
                self.plotBinnedDistributionWithSeparatrix(self.machineModel.E_inj + E0, \
                                                  D_i, \
                                                  title='Test Data, frame=%d' % i)
    
            state = self.machineModel.turnFor(state, self.N, self.machineModel.FORWARD)
            state.ΔE_max = self.machineModel.getBucketHeightFast(state.Es)
            state.ΔX = self.mapsForward.ΔX_0
            state.ΔY = (2*state.ΔE_max)/self.RESOLUTION
    
        assert len(proj) == (self.FRAMES + 1)
        return np.array(proj)

    def _generateGuessProjections(self, D0):
        proj = []

        #print('_generateGuessProjections(): D0 = %s' % str(D0))
    
        #proj.append(mapsForward.getBinnedProjection(D0))
        D_i = self.mapsForward.normalizeBinnedDistribution(D0) # D0 is not assumed to be normalized
        proj_i = self.mapsForward.getBinnedProjection(D_i)
        proj.append(proj_i)
        
        for i in range(0, self.FRAMES):
            D_i = self.mapsForward.normalizeBinnedDistribution(self.mapsForward.getNextFrame(i, D_i))
            proj_i = self.mapsForward.getBinnedProjection(D_i)
            proj.append(proj_i)
    
        assert len(proj) == (self.FRAMES + 1) # +1 including the reconstruction frame
        return np.array(proj)

    def _getErrorProjections(self, data, guess):
        assert data.shape == guess.shape
        return np.array(data - guess)

    def _discrepancy(self, ΔP, N, L):
        # ORIGINAL (from literature)
        return np.sqrt(sum(ΔP.flatten()**2)/((N+1)*L))
        
        #return np.sqrt(sum(ΔP[N].flatten()**2)/(L)) # test: only measure the discrepancy of one frame

        # RMS of RMS of error
        assert len(ΔP) == self.FRAMES + 1
        assert N == self.FRAMES
        
        d_overall = 0
        d_frames = []
        for i in range(0, N+1):
            d_i = np.sqrt(sum(ΔP[i].flatten()**2)/(L))
            d_frames.append(d_i)
        d_frames = np.array(d_frames)
        d_overall = np.sqrt(sum(d_frames.flatten()**2)/(N+1))
        #d_overall = sum(d_bins.flatten()) / (N+1)
        return d_overall

    '''def _getEmittance(self, D, p_obj, N=100):
        assert D.sum() == 1

        Nppp_ϵ = 4**2 # we don't need that many particles per cell to approximate the fraction
        
        H_s = self.machineModel.H(self.Es_recon, 0, self.machineModel.ϕs) # origin (i.e. sync. particle) to start from
        H_0 = self.machineModel.H(self.Es_recon, 0, self.machineModel.right) # separatrix

        # for determining above/below transition
        β = self.machineModel.β(self.Es_recon)
        η = self.machineModel.η(β)*self.machineModel.Npasses

        H_list = np.linspace(H_s, H_0, N) # list of H values to try

        ϵ_matrix = None # used to indicate portion of each cell contained within action contour

        Y,X = D.shape
        
        ΔX = self.mapsForward.ΔX_0
        ΔY = (2*self.machineModel.getBucketHeightFast(self.Es_recon))/self.RESOLUTION

        p_actual = 0

        def _isParticleWithinActionContour(H_ref, ϕ, ΔE):
            H = self.machineModel.H(self.Es_recon, ΔE, ϕ)
            if η < 0: # below transition
                insideBucket = (H > H_ref)
            elif η > 0: # above transition
                insideBucket = (H < H_ref)
            else: # precisely at transition
                raise ValueError('η is zero, halting.')

            return insideBucket

        # for each action contour
        for i in tqdm(range(N), 'Calculating %d%% emittance' % int(p_obj*100)):
            H_i = H_list[i] # Hamitonian for this action contour

            if H_i == H_0:
                print('[DEBUG]: current action contour is separatrix')

            # reset on each iteration
            ϵ_matrix = np.zeros_like(D, dtype=float)

            # for each cell
            for y in range(Y):
                for x in range(X):
                    num_in = 0

                    # state of uniformly dist. ptcls
                    uni_state = State(self.Es_recon, np.zeros(Nppp_ϵ), np.zeros(Nppp_ϵ)) 
                    uni_state.ΔE_max = self.machineModel.getBucketHeightFast(uni_state.Es)
                    uni_state.ΔX = ΔX
                    uni_state.ΔY = ΔY

                    uni_state = self.mapsForward.placeTestParticlesFast(uni_state, y, x)

                    # for each particle in the cell
                    for j in range(Nppp_ϵ):
                        # particle coordinates in phase space
                        ϕ_j = uni_state.ϕ[j]
                        ΔE_j = uni_state.ΔE[j]
                        if not _isParticleWithinActionContour(H_i, ϕ_j, ΔE_j):
                            continue
                        
                        num_in += 1

                    # calculate fraction of cell inside action contour
                    p_in = num_in/Nppp_ϵ

                    assert (0 <= p_in <= 1)

                    # assign the fraction to the corresponding cell in ϵ_matrix
                    ϵ_matrix[y, x] = p_in

            # element-wise multiply nonzero cells in D by ϵ_matrix
            p_actual = np.multiply(D, ϵ_matrix).sum()
            if p_actual >= p_obj:
                break # computed fraction is at or above objective fraction (e.g. 68%, 95%)
            # no - repeat

        #-----------------------------
        # Loop done (objective fraction reached or completed loop without reaching objective)

        if not p_actual >= p_obj:
            print('[ERROR]: see below; p_actual=%f, p_obj=%f' % (p_actual, p_obj))
        assert p_actual >= p_obj, '[ERROR]: fraction within max action contour (p_actual) less than p_obj; %d%% of particles could not be found within bucket'
        
        # multiply ϵ_matrix by ΔX and then ΔY
        ϵ_matrix *= (ΔX * ΔY)

        # this is in the wrong units
        emittance_eV_rad = ϵ_matrix.sum()

        # so fix it to the more conventional one (eV * s)
        ω_RF = self.machineModel.ω_RF(self.Es_recon)
        emittance_eV_s = emittance_eV_rad / ω_RF

        return emittance_eV_s'''

    # locate the cells that lie on the separatrix boundary
    def _getBoundaryCells(self, D):
        Y,X = D.shape
        assert (Y,X) == (self.RESOLUTION, self.RESOLUTION)

        # for determining above/below transition
        β = self.machineModel.β(self.Es_recon)
        η = self.machineModel.η(β)*self.machineModel.Npasses

        H_0 = self.machineModel.H(self.Es_recon, 0, self.machineModel.right) # separatrix

        D_boundary = np.zeros_like(D) # this is how we will mark the boundary cells

        def _isParticleWithinBucket(ϕ, ΔE):
            H = self.machineModel.H(self.Es_recon, ΔE, ϕ)
            if η < 0: # below transition
                insideBucket = (H > H_0)
            elif η > 0: # above transition
                insideBucket = (H < H_0)
            else: # precisely at transition
                raise ValueError('η is zero, halting.')

            return insideBucket

        # for each cell
        for y in range(Y):
            for x in range(X):
                # state of uniformly dist. ptcls
                uni_state = State(self.Es_recon, np.zeros(self.Nppp), np.zeros(self.Nppp))
                # I have to do this or it crashes (assertion will fail)
                uni_state.ΔE_max = self.machineModel.getBucketHeightFast(uni_state.Es)

                uni_state = self.mapsForward.placeTestParticlesFast(uni_state, y, x)

                # for each particle
                for j in range(self.Nppp):
                    ϕ_j = uni_state.ϕ[j]
                    ΔE_j = uni_state.ΔE[j]

                    # if even a single particle is outside the bucket, mark the cell
                    if not _isParticleWithinBucket(ϕ_j, ΔE_j):
                        D_boundary[y, x] = 1
                        break

        return D_boundary

    def _getEmittance(self, D, p_obj, N=100):
        #assert D.sum() == 1, 'D.sum() = %s' % str(D.sum())

        Nppp_ϵ = 4**2 # we don't need that many particles per cell to approximate the fraction
        
        H_s = self.machineModel.H(self.Es_recon, 0, self.machineModel.ϕs) # origin (i.e. sync. particle) to start from
        H_0 = self.machineModel.H(self.Es_recon, 0, self.machineModel.right) # separatrix

        # for determining above/below transition
        β = self.machineModel.β(self.Es_recon)
        η = self.machineModel.η(β)*self.machineModel.Npasses

        H_list = np.linspace(H_s, H_0, N) # list of H values to try

        ϵ_matrix = None # used to indicate portion of each cell contained within action contour

        Y,X = D.shape
        
        ΔX = self.mapsForward.ΔX_0
        ΔY = (2*self.machineModel.getBucketHeightFast(self.Es_recon))/self.RESOLUTION

        p_actual = 0

        print('[DFBUG]: marking boundary cells')
        D_boundary = self._getBoundaryCells(D)

        def _isParticleWithinActionContour(H_ref, ϕ, ΔE):
            H = self.machineModel.H(self.Es_recon, ΔE, ϕ)
            if η < 0: # below transition
                insideBucket = (H > H_ref)
            elif η > 0: # above transition
                insideBucket = (H < H_ref)
            else: # precisely at transition
                raise ValueError('η is zero, halting.')

            return insideBucket

        # for each action contour
        for i in tqdm(range(N), 'Calculating %d%% emittance' % int(p_obj*100)):
            H_i = H_list[i] # Hamitonian for this action contour

            if H_i == H_0:
                print('[DEBUG]: current action contour is separatrix')

            # reset on each iteration
            ϵ_matrix = np.zeros_like(D, dtype=float)

            # for each cell
            for y in range(Y):
                for x in range(X):
                    num_in = 0

                    # state of uniformly dist. ptcls
                    uni_state = State(self.Es_recon, np.zeros(Nppp_ϵ), np.zeros(Nppp_ϵ)) 
                    uni_state.ΔE_max = self.machineModel.getBucketHeightFast(uni_state.Es)
                    uni_state.ΔX = ΔX
                    uni_state.ΔY = ΔY

                    uni_state = self.mapsForward.placeTestParticlesFast(uni_state, y, x)

                    # for each particle in the cell
                    for j in range(Nppp_ϵ):
                        # particle coordinates in phase space
                        ϕ_j = uni_state.ϕ[j]
                        ΔE_j = uni_state.ΔE[j]
                        if not _isParticleWithinActionContour(H_i, ϕ_j, ΔE_j):
                            continue
                        
                        num_in += 1

                    # calculate fraction of cell inside action contour
                    p_in = num_in/Nppp_ϵ

                    assert (0 <= p_in <= 1)

                    # assign the fraction to the corresponding cell in ϵ_matrix
                    ϵ_matrix[y, x] = p_in

            # element-wise multiply nonzero cells in D by ϵ_matrix
            p_actual = np.multiply(D, ϵ_matrix).sum()
            if p_actual >= p_obj:
                break # computed fraction is at or above objective fraction (e.g. 68%, 95%)
            # no - repeat

        #-----------------------------
        # Loop done (objective fraction reached or completed loop without reaching objective)

        if not p_actual >= p_obj:
            print('[ERROR]: see below; p_actual=%f, p_obj=%f' % (p_actual, p_obj))
        assert p_actual >= p_obj, '[ERROR]: fraction within max action contour (p_actual) less than p_obj; %d%% of particles could not be found within bucket'
        
        # multiply ϵ_matrix by ΔX and then ΔY
        ϵ_matrix *= (ΔX * ΔY)

        # this is in the wrong units
        emittance_eV_rad = ϵ_matrix.sum()

        # so fix it to the more conventional one (eV * s)
        ω_RF = self.machineModel.ω_RF(self.Es_recon)
        emittance_eV_s = emittance_eV_rad / ω_RF

        return emittance_eV_s

    def _reconstruct(self, proj_data, D_guess, dbg=False):
        #proj_guess = self._generateGuessProjections(D_guess)
        proj_guess = self.dbgGenerateGuessProjections(D_guess)
        #print('proj_data.shape=%s, proj_guess.shape=%s' % (str(proj_data.shape), str(proj_guess.shape)))
        proj_error = self._getErrorProjections(proj_data, proj_guess)

        if dbg:
            self.plotSinogram(proj_guess, title='Guess Projections')
            self.plotSinogram(proj_error, title='Error Projections', cmap='bwr')
    
        d = self._discrepancy(proj_error, self.FRAMES, self.RESOLUTION)
    
        D_BP = [] # straight BP'd error
    
        # BP errors
        for i in range(0, self.FRAMES+1):
            D_error_i = self.mapsForward.getBackProjection(proj_error[i], frame=i)
            D_BP.append(D_error_i)

            if dbg:
                plt.imshow(D_error_i, cmap='bone_r')
                plt.title('Back-Projected Error, frame #%d' % (i))
                plt.show()
            
        D_correction = np.zeros((self.RESOLUTION, self.RESOLUTION))
        
        # reorient BPs
        for i in range(0, self.FRAMES+1):
            D_reoriented_i = D_BP[i]
            
            # turn BP'd error back in time
            for n in range(0, i):
                D_reoriented_i = self.mapsBackward.getNextFrame(i-n, D_reoriented_i)

            if dbg:
                plt.imshow(D_reoriented_i, cmap='bone_r')
                plt.title('Reoriented Error, frame #%d' % (i))
                plt.show()
    
            # add to correction
            D_correction += D_reoriented_i
        # average the correction
        D_correction /= (self.FRAMES+1) # should it be +1? Yes but it also doesnt matter much
    
        D_recon = D_guess + (self.λ * D_correction) # apply correction
        D_recon = self.mapsForward.normalizeBinnedDistribution(D_recon)

        if dbg:
            plt.imshow(D_recon, cmap=self.CMAP)
            plt.title('Reconstruction, ITER=%d' % (1))
            plt.show()

        self.D_recon = D_recon # save it so we can plot it later
        
        return D_recon, d

    #************************************************
    # RECONSTRUCTION STEPS:
    # 1) Get projections of the guess
    # 2) Get error profiles
    # 3) BP errors
    # 4) Reorient BP'd errors
    # 5) Sum reoriented errors
    #************************************************
    def reconstruct(self, plot=True, _dbg=False):
        # generate initial guess
        D_guess = self._createUniformDistribution(frame=0)
        #D_guess = self.dbgGenerateGuess()
        
        '''plt.imshow(D_guess, cmap=self.CMAP)
        plt.title('Initial Guess Distribution, frame #%d' % 0)
        plt.show()'''
        '''if plot:
            self.plotBinnedDistributionWithSeparatrix(self.Es_recon, \
                                                      D_guess, \
                                                      title='Initial Guess Distribution, frame #%d' % 0)'''
    
        #print('D_guess.sum() = %f' % D_guess.sum())

        #===================================================================

        d = [] # discrepancy over iterations

        D_recon = D_guess
        for c in tqdm(range(0, self.ITERATIONS), 'Reconstructing (ITERATIONS=%d)' % self.ITERATIONS):
            # originally meant for debug
            if c == 0:
                D_recon, d_c = self._reconstruct(self.proj_data, D_recon, dbg=_dbg) # set to true if you want the intemediates
            else:
                D_recon, d_c = self._reconstruct(self.proj_data, D_recon, dbg=_dbg)
            '''plt.imshow(D_recon, cmap=CMAP)
            plt.title('Reconstruction, ITER=%d' % (c+1))
            plt.show()'''

            # debug
            '''self.plotBinnedDistributionWithSeparatrix(self.Es_recon, \
                                                  D_recon, \
                                                  title='Reconstruction, ITER=%d' % (c+1))'''
            
            d.append(d_c)

        # add discrepancy of final frame
        proj_guess = self.dbgGenerateGuessProjections(D_recon)
        proj_error = self._getErrorProjections(self.proj_data, proj_guess)
        d_fin = self._discrepancy(proj_error, self.FRAMES, self.RESOLUTION)
        d.append(d_fin)

        #===================================================================
    
        # show result
        '''plt.imshow(D_recon, cmap=self.CMAP)
        plt.title('Reconstruction, ITER=%d' % (self.ITERATIONS))
        plt.show()'''
        if plot:
            self.plotBinnedDistributionWithSeparatrix(self.Es_recon, \
                                                      D_recon, \
                                                      title='Reconstruction, ITER=%d' % (self.ITERATIONS))

        '''self.plotBinnedDistributionWithSeparatrix(self.Es_recon, \
                                                  self.D_0, \
                                                  title='Initial Distribution, same density scale', \
                                                  _vmin=D_recon.min(), \
                                                  _vmax=D_recon.max())'''

        if plot:
            pass
            #self.plotSinogram(self.dbgGenerateGuessProjections(D_recon), 'REconstruction Sinogram')

        

        if plot:
            plt.plot(np.arange(0, self.ITERATIONS+1), d)
            plt.title('Discrepancy over %d iterations' % self.ITERATIONS)
            plt.ylabel('Discrepancy')
            plt.xlabel('Iteration')
            plt.show()

        #-------------------------------------------------------------------------------
        # Pack the reconstruction result into a neat little object

        D = D_recon
        projections = self.dbgGetForwardProjections(D_recon)
        L = self.RESOLUTION
        N_iter = self.ITERATIONS
        Nppp = self.Nppp
        Es = self.Es_recon
        ϕs = self.machineModel.ϕs
        ϕ_left = self.machineModel.left
        ϕ_right = self.machineModel.right
        ΔE_max = self.machineModel.getBucketHeightFast(self.Es_recon)
        separatrix_points = self.machineModel._generateSeparatrixPoints(self.Es_recon)
        d_history = d
        ω_RF = self.machineModel.ω_RF(self.Es_recon)
        #emittance68 = self._getEmittance(D_recon, p_obj=0.68, N=100)
        #emittance95 = self._getEmittance(D_recon, p_obj=0.95, N=100)
        #emittance99 = self._getEmittance(D_recon, p_obj=0.99, N=100)
        name = ''
        result = ReconstructionResult(
            D, 
            projections, 
            L, 
            N_iter, 
            Nppp, 
            Es, 
            ϕs, 
            ϕ_left, 
            ϕ_right, 
            ΔE_max,
            ω_RF,
            0, # DEBUGGING
            0, # DEBUGGING
            0, # DEBUGGING
            separatrix_points,
            d_history,
            name
        )

        return result
        #return d

    def plotBinnedDistributionWithSeparatrix(self, Es_frame, D, title='', _vmin=None, _vmax=None):
        state_frame = State(Es_frame, None, None)
        ΔE_max_MeV = self.machineModel.getBucketHeightFast(state_frame.Es) / 1e6
        
        #extent_frame = [-π, π, -ΔE_max_MeV, ΔE_max_MeV]
        extent_frame = [self.machineModel.left, self.machineModel.right, -ΔE_max_MeV, ΔE_max_MeV]

        if (_vmin is not None) and (_vmax is not None):
            assert _vmax > _vmin
            im = plt.imshow(D, extent=extent_frame, aspect='auto', cmap=self.CMAP, vmin=_vmin, vmax=_vmax)
        else:
            im = plt.imshow(D, extent=extent_frame, aspect='auto', cmap=self.CMAP)
        plt.colorbar(im)
        plt.title(title)
        plt.xlabel('ϕ [rad]')
        plt.ylabel('ΔE [MeV]')
        self.machineModel.plotSeparatrix(state_frame)
        plt.tight_layout()
        plt.show()

    def plotSinogram(self, proj, title='', cmap=None, save=False, center=False, showScale=False):
        if cmap == None:
            cmap = self.CMAP
        if center:
            plt.imshow(proj, cmap=cmap, extent=[-π, π, self.FRAMES+1, 0], norm=colors.CenteredNorm(vcenter=0))
        else:
            plt.imshow(proj, cmap=cmap, extent=[-π, π, self.FRAMES+1, 0])
        plt.title(title)
        plt.xlabel('ϕ [rad]')
        plt.xticks([-π, 0, π], ['-π', '0', 'π'])
        plt.ylabel('Frame')
        plt.yticks(np.linspace(0, self.FRAMES+1, self.FRAMES+1+1), np.arange(self.FRAMES+1+1))
        if showScale:
            plt.colorbar()
        if save:
            plt.savefig('figures/measuredDataSinogram.png', bbox_inches='tight')
        plt.show()

    # ϕΔαηβπδ
    def sweepParameters(self, ϕs_range, V_range, step_count):
        pass

    def dbgGetForwardProjections(self, D_correct):
        # this method below actually only generates forward projections using the current machine model so it'll do
        return self.dbgGenerateGuessProjections(D_correct)

    # ONLY CALL FOR PROJ_DATA
    def dbgGetCorrectDistribution(self):
        # generate the "correct" distribution
        state = self._generateDistribution(turns=100, shift=True)
        D_correct = self.mapsForward.getBinnedProbabilityDistributionMap(state)
        return D_correct

    def dbgPlotReconstructedSinogram(self):
        self.plotSinogram(self.dbgGetForwardProjections(self.D_recon), title='Reconstructed Sinogram', save=True)

    def dbgPlotErrorSinogram(self, cmap='RdBu'):
        original_cmap = self.CMAP
        self.CMAP = cmap
        self.plotSinogram(self.proj_data - self.dbgGetForwardProjections(self.D_recon), \
                          title='Error Sinogram', \
                          save=True, 
                          center=True,
                          showScale=True)
        self.CMAP = original_cmap

    def dbgReconstruct(self):
        pass