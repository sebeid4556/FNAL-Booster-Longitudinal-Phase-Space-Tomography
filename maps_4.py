#==================================================================================
# maps_4: this version builds on the last version by utilizing a more spatially
# efficient transport map storage method (both in RAM and on disk)
#==================================================================================
import numpy as np
import matplotlib as plt
from model_2 import * # for State object
import os
from tqdm.auto import tqdm
import multiprocessing
import time
from functools import wraps

# projection operator - returns a Nx1 matrix
def P(W):
    return np.transpose(W) @ np.ones(W.shape[1])

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        #print(f"Function '{func.__name__}' took {end - start:.6f} seconds")
        return result
    return wrapper

class MapGenerator:
    FORWARD = 0
    BACKWARD = 1
    def __init__(self, machineModel, Es, resolution, Nppp, N, FRAMES, direction):
        #===============================
        # [!] WARNING: DO NO TOUCH [!]
        #===============================
        self.__MAX_RESOLUTION = 100 # set a limit to make sure the hard drive isn't destroyed
        #===============================
        
        self.machineModel = machineModel
        self.Es = Es # this is the Es from which the model runs - for a stationary bucket E_inj=400MeV is fine
        self.RESOLUTION = resolution
        self.Nppp = Nppp
        self.N = N # number of turns between frames
        self.FRAMES = FRAMES # number of frames to generate maps for
        
        self.NUM_PIXELS = self.RESOLUTION**2

        # sanity checks
        assert self.RESOLUTION <= self.__MAX_RESOLUTION # this line is what stands between a functioning PC and a brick
        assert self.RESOLUTION > 0
        assert self.RESOLUTION % 2 == 0 # ensure even number
        assert int(np.sqrt(Nppp))**2 == Nppp # ensure perfect square

        #=============================================================
        self.Np_side = int(np.sqrt(self.Nppp))
        assert self.Np_side**2 == self.Nppp
        
        # Normalized interior coordinates in one pixel.
        # Values are in (0, 1), not including edges.
        u = (np.arange(self.Np_side) + 0.5) / self.Np_side
        
        # Match meshgrid(...).flatten() ordering:
        # phi varies fastest, ΔE varies slowest.
        self._u_phi = np.tile(u, self.Np_side)
        self._u_E = np.repeat(u, self.Np_side)
        #=============================================================

        self.MAG = int(self.RESOLUTION/2)

        # use compact phase window
        self.ϕ_right = self.machineModel.right
        self.ϕ_left = self.machineModel.left
        #self.ϕ_right = π
        #self.ϕ_left = -π
        
        self.ΔX_0 = abs(self.ϕ_right - self.ϕ_left)/self.RESOLUTION
        # quickly compute the bucket height for the initial distribution
        self.ΔY_0 = 2*self.machineModel.getBucketHeightFast(self.Es)/self.RESOLUTION # for frame 0

        #self.W_all = [] # holds subarrays for each frame, where eacch subarray has the transport information for each cell
        self.W_all = np.empty(0, dtype=object)
        self.p = np.zeros((self.FRAMES+1, self.RESOLUTION, self.RESOLUTION)) # holds the back-projection matrix for each frame (so len(self.p) == self.FRAMES+1)

        # use these to specify direction in which the maps simulate the machine
        self.DIRECTION = direction

        assert self.DIRECTION in [MapGenerator.FORWARD, MapGenerator.BACKWARD]

        self.SAVE_PATH = '../../../../Desktop/MAPS/'
        self.BASE_FILENAME = 'compact'
        self.ATTRIBUTE = 'running'

        self.filename = "%s_%s_res%d_Es%0.8fMeV_nppp%d_frames%d_turns%d_phis%0.5fpi_V%0.5fkV_%s.npy" % \
        ('forward' if self.DIRECTION == MapGenerator.FORWARD else 'backward', \
                                                                       self.BASE_FILENAME, \
                                                                       self.RESOLUTION,\
                                                                       self.Es/1e6, \
                                                                       self.Nppp, \
                                                                       self.FRAMES, \
                                                                       self.N, \
                                                                       self.machineModel.ϕs/π, \
                                                                       self.machineModel.Vmax/1e3, \
                                                                       self.ATTRIBUTE)
        self.filepath = os.path.join(self.SAVE_PATH, self.filename)

        #===========================================
        # now for the back-projection (BP) matrices
        #===========================================
        
        self.BP_SAVE_PATH = '../../../../Desktop/BP_MATRICES/'
        self.BP_BASE_FILENAME = 'bpmatrices'

        self.bp_filename = "%s_res%d_nppp%d_frames%d_turns%d_phis%0.5f_%s.npy" % (\
                                                                       self.BASE_FILENAME, \
                                                                       self.RESOLUTION,\
                                                                       self.Nppp, \
                                                                       self.FRAMES, \
                                                                       self.N, \
                                                                       self.machineModel.ϕs, \
                                                                       self.ATTRIBUTE)
        self.bp_filepath = os.path.join(self.BP_SAVE_PATH, self.bp_filename)

        #===========================================
        # Parallel processing
        #===========================================
        self.CPU_COUNT = os.cpu_count()
        #self.pool = multiprocessing.Pool(self.CPU_COUNT)

        #===========================================
        # Precompute input pool template
        #===========================================
        x_arr = np.arange(0, self.RESOLUTION) # invariant
        y_arr = np.arange(0, self.RESOLUTION) # invariant
        X, Y = np.meshgrid(x_arr, y_arr) # invariant
        Es_frame_arr = self.Es * np.ones((self.RESOLUTION, self.RESOLUTION)) # different for each frame
        self.input_pool = np.column_stack((Es_frame_arr.ravel(), X.ravel(), Y.ravel()))

    #***********************************************************
    # Useful methods for converting between abs and rel pixel
    # coordinates
    #***********************************************************
    def _absCoordsToRelCoords(self, y, x):
        rel_coords = (
            self.MAG - y if y < self.MAG else -(y - (self.MAG-1)),
            x - (self.MAG-1) if x >= self.MAG else x - self.MAG
        )
        return rel_coords
    
    def _relCoordsToAbsCoords(self, y, x):
        abs_coords = (
            self.MAG - y if y > 0 else (self.MAG-1) - y,
            x + (self.MAG-1) if x > 0 else x + self.MAG
        )
        return abs_coords

    #***********************************************************
    # Weight matrix creation
    #***********************************************************
    def placeTestParticles(self, state, y_rel, x_rel):
        """
        Place Nppp test particles uniformly inside the pixel addressed by
        signed relative coordinates (y_rel, x_rel), but using the actual
        running-bucket frame:
    
            phi_left <= phi <= phi_right
            -ΔE_max <= ΔE <= +ΔE_max
    
        y_rel, x_rel are only labels. We first convert them to absolute
        matrix indices, then use the true bucket-frame boundaries.
        """
    
        Np_side = int(np.sqrt(self.Nppp))
        assert Np_side**2 == self.Nppp
    
        # ------------------------------------------------------------
        # Frame geometry for this state/frame
        # ------------------------------------------------------------
    
        # For now, your running bucket uses fixed phase bounds.
        # If later phi_left/right vary with energy, set these on the state
        # before calling this function.
        ϕ_left = self.ϕ_left
        ϕ_right = self.ϕ_right
    
        # Bucket height for this frame
        ΔE_max = state.ΔE_max
        assert ΔE_max is not None
    
        # Optional center; usually ΔE is measured relative to synchronous energy,
        # so the bucket is centered at ΔE = 0.
        #ΔE_center = getattr(state, "ΔE_center", 0.0)
        ΔE_center = 0 # should always be centered at zero
    
        # Cell sizes for this frame
        ΔX = (ϕ_right - ϕ_left) / self.RESOLUTION
        ΔY = (2 * ΔE_max) / self.RESOLUTION
    
        # ------------------------------------------------------------
        # Convert signed relative pixel coordinates to absolute indices
        # ------------------------------------------------------------
    
        y_abs, x_abs = self._relCoordsToAbsCoords(y_rel, x_rel)
    
        assert 0 <= y_abs < self.RESOLUTION
        assert 0 <= x_abs < self.RESOLUTION
    
        # ------------------------------------------------------------
        # Get actual physical boundaries of this pixel
        # ------------------------------------------------------------
    
        # Column x_abs runs left to right:
        #
        # x_abs = 0              -> [ϕ_left, ϕ_left + ΔX]
        # x_abs = RESOLUTION - 1 -> [ϕ_right - ΔX, ϕ_right]
        ϕ_min = ϕ_left + x_abs * ΔX
        ϕ_max = ϕ_left + (x_abs + 1) * ΔX
    
        # Row y_abs runs top to bottom:
        #
        # y_abs = 0              -> [ΔE_max - ΔY, ΔE_max]
        # y_abs = RESOLUTION - 1 -> [-ΔE_max, -ΔE_max + ΔY]
        ΔE_top = ΔE_center + ΔE_max - y_abs * ΔY
        ΔE_bottom = ΔE_center + ΔE_max - (y_abs + 1) * ΔY
    
        # ------------------------------------------------------------
        # Place particles at interior grid points, not on pixel edges
        # ------------------------------------------------------------
    
        ϕ_vals = np.linspace(
            ϕ_min + ΔX / (2 * Np_side),
            ϕ_max - ΔX / (2 * Np_side),
            Np_side
        )
    
        ΔE_vals = np.linspace(
            ΔE_bottom + ΔY / (2 * Np_side),
            ΔE_top - ΔY / (2 * Np_side),
            Np_side
        )
    
        ϕ_grid, ΔE_grid = np.meshgrid(ϕ_vals, ΔE_vals)
    
        state.ϕ = ϕ_grid.flatten()
        state.ΔE = ΔE_grid.flatten()
    
        return state
    
    def placeTestParticlesFast(self, state, y_abs, x_abs):
        """
        Place Nppp test particles uniformly inside one pixel of the running-bucket frame.
    
        Optimized version:
        - no np.linspace per call
        - no np.meshgrid per call
        - no flatten per call
        - reuses precomputed unit-pixel coordinates
        """
    
        ΔE_max = state.ΔE_max
        assert ΔE_max is not None
    
        # Frame bounds
        ϕ_left = self.ϕ_left
        ϕ_right = self.ϕ_right
    
        # Cell sizes
        ΔX = self.ΔX_0
        ΔY = (2.0 * ΔE_max) / self.RESOLUTION
    
        # Convert signed relative pixel coords to absolute matrix indices
        #y_abs, x_abs = self._relCoordsToAbsCoords(y_rel, x_rel)
    
        # Physical lower-left corner of this cell
        ϕ_min = ϕ_left + x_abs * ΔX
    
        # Row y_abs runs top-to-bottom, so bottom edge is:
        ΔE_bottom = ΔE_max - (y_abs + 1) * ΔY
    
        # Make sure arrays exist and have the correct size
        if state.ϕ is None or state.ϕ.shape[0] != self.Nppp:
            state.ϕ = np.empty(self.Nppp)
    
        if state.ΔE is None or state.ΔE.shape[0] != self.Nppp:
            state.ΔE = np.empty(self.Nppp)
    
        # In-place affine transform from unit-cell coordinates to actual coordinates
        np.multiply(self._u_phi, ΔX, out=state.ϕ)
        state.ϕ += ϕ_min
    
        np.multiply(self._u_E, ΔY, out=state.ΔE)
        state.ΔE += ΔE_bottom
    
        # Store current frame geometry
        state.ΔX = ΔX
        state.ΔY = ΔY
    
        return state

    #***********************************************************
    # Binning particle distribution map
    #***********************************************************

    #========================================
    # OPTIMIZATION TARGETS:
    # 
    #========================================
    def getBinnedDistributionMap(self, state):
        assert state.ϕ.shape == state.ΔE.shape

        ΔX = state.ΔX
        ΔY = state.ΔY
        ΔE_max = state.ΔE_max
        assert ΔX is not None
        assert ΔY is not None
        assert ΔE_max is not None
    
        D = np.zeros((self.RESOLUTION, self.RESOLUTION)) # binned particle distribution map

        lost_count = 0 # number of particles that were lost from the stable region

        H0 = self.machineModel.H(state.Es, 0, self.machineModel.right) # Hamiltonian along the separatrix for this Es
        β = self.machineModel.β(state.Es)
        η = self.machineModel.η(β)*self.machineModel.Npasses # η on this turn

        #print('getBinnedDistributionMap(): H0 = %f' % H0)

        # use Hamiltonian of separatrix to determine if particle is inside stable region,
        # use coordinates to determine if particle is in the "tail" of the "fish", 
        # in which case it would be counted as outside the bucket
        #
        # Is the Hamiltonian inside stable region supposed to invert(turn inside out) at transition?
        # Ex. below transition, particle is stable if H > H0
        #     above transition, particle is stable if H < H0
        # Is this the correct behavior?
        def _isParticleInBucket(ϕ, ΔE):
            H = self.machineModel.H(state.Es, ΔE, ϕ)
            if η < 0: # below transition
                insideBucket = (H > H0)
            elif η > 0: # above transition
                insideBucket = (H < H0)
                #insideBucket = (H > H0)
                #print('Above transition, H0=%f, H=%f, insideBucket=%s' % (H0, H, str(insideBucket)))
            else: # precisely  at transition
                raise ValueError('η is zero, halting.')
                
            return insideBucket == True \
                and (ϕ >= self.machineModel.left) \
                and (ϕ <= self.machineModel.right) \
                and (ΔE <= ΔE_max) \
                and (ΔE >= -ΔE_max)
        
        # for each particle being tracked
        for i in range(0, len(state.ϕ)):
            # grab coordinates in (ϕ, ΔE) units
            ϕ_i = state.ϕ[i]
            ΔE_i = state.ΔE[i]

            '''# determine which pixel this particle is in
            rel_row = (int(ΔE_i / ΔY) + (-1 if ΔE_i < 0 else 1))
            rel_col = (int(ϕ_i / ΔX) + (-1 if ϕ_i < 0 else 1))'''

            # determine which cell this particle is in using absolute coordinates
            #abs_row = int(np.floor(ΔE_i / ΔY)) # make sure to cast to int to use value as index
            abs_row = int(np.floor((ΔE_max - ΔE_i) / ΔY)) # vertical distance from ΔE_max (top edge)
            abs_col = int(np.floor((ϕ_i - self.ϕ_left) / ΔX)) # horizontal distance from ϕ_left (left edge)

            if not _isParticleInBucket(ϕ_i, ΔE_i):
                lost_count += 1
            else:
                #print('(rel_row, rel_col)=(%d, %d)' % (rel_row, rel_col))
                #abs_row, abs_col = self._relCoordsToAbsCoords(rel_row, rel_col)
                '''print('type(abs_row)=%s' % (str(type(abs_row))))
                print('type(abs_col)=%s' % (str(type(abs_col))))
                print('(abs_row, abs_col)=(%d, %d)' % (abs_row, abs_col))'''
                D[abs_row, abs_col] += 1    

        #print('%d particles were lost' % lost_count)

        return D, lost_count

    # same as above but return the normalized version
    def getBinnedProbabilityDistributionMap(self, state):
        return self.normalizeBinnedDistribution(self.getBinnedDistributionMap(state)[0])

    '''# weight each cell based on how much of the total pixels it contains
    def createWeightMatrix(self, state):
        W, lost_count = self.getBinnedDistributionMap(state)
        
        # count how many test particles remained in the stable region after
        test_particles_left = self.Nppp - lost_count

        #print('%d particles were lost' % (lost_count))

        # divide by the number of test particles left to normalize the distribution
        assert sum(W.flatten()) == test_particles_left
        
        # avoid division by zero
        if test_particles_left > 0:
            return W/test_particles_left
        else:
            return np.zeros_like(W)'''

    # weight each cell based on how much of the total pixel it contains
    # this version cuts down storage space by ~90%
    def createWeightMatrixCompact(self, state):
        W, lost_count = self.getBinnedDistributionMap(state)
        
        # count how many test particles remained in the stable region after
        test_particles_left = self.Nppp - lost_count

        #print('%d particles were lost' % (lost_count))

        # divide by the number of test particles left to normalize the distribution
        assert sum(W.flatten()) == test_particles_left
        
        W_compact = []

        # avoid division by zero
        if test_particles_left == 0:
            return W_compact

        W /= test_particles_left # normalize
        
        positive_cell_rows, positive_cell_cols = np.where(W > 0) # find cells that aren't zero

        # add en entry for each positive (meaningful) cell
        for i in range(len(positive_cell_rows)):
            row_i, col_i = positive_cell_rows[i], positive_cell_cols[i]            
            val_i = W[row_i, col_i]
            
            W_entry = [row_i, col_i, val_i] # storage format: [row, col, value]
            W_compact.append(W_entry)
        return W_compact

    #***********************************************************
    # Plotting utilities
    #***********************************************************        
    def plotWeightMatrix(self, W):
        fig, ax = plt.subplots()
    
        M, N = W.shape
    
        ax.imshow(
            W,
            cmap='binary',
            vmin=0.0,
            vmax=1.0,
            extent=(0, N, M, 0),
            interpolation='none'
        )
    
        # pixel boundaries
        ax.set_xticks(np.arange(0, N + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, M + 1, 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.1)
        ax.tick_params(which='minor', bottom=False, left=False)
    
        # pixel centers
        ax.set_xticks(np.arange(N) + 0.5)
        ax.set_yticks(np.arange(M) + 0.5)
    
        ax.set_xticklabels([i - N//2 if i < N//2 else i - N//2 + 1 for i in range(N)])
        ax.set_yticklabels([M//2 - i if i < M//2 else M//2 - i - 1 for i in range(M)])
    
        ax.set_xlim(0, N)
        ax.set_ylim(M, 0)
    
        plt.show()

    def drawGrid(self, state):
        ΔX = state.ΔX
        ΔY = state.ΔY
        ΔE_max = state.ΔE_max
        assert ΔX is not None
        assert ΔY is not None
        assert ΔE_max is not None

        ax = plt.gca()
        ax.set_xticks(np.arange(self.ϕ_left, self.ϕ_right + ΔX, ΔX))
        ax.set_yticks(np.arange(-ΔE_max, ΔE_max + ΔY, ΔY) / 1e6)

        ax.set_xlim(self.ϕ_left, self.ϕ_right)
        ax.set_ylim(-ΔE_max/1e6, ΔE_max/1e6)
        ax.grid(True)
    
    def plotWithGrid(self, state):
        self.machineModel.plot(state, show=False, fixed=False)
        self.drawGrid(state)
        plt.show()

    def plotWeightMatrixProjection(self, W):
        M, N = W.shape
        self.MAG = N // 2
    
        x = np.arange(N)  # actual bar positions
        pixels = np.concatenate((np.arange(-self.MAG, 0), np.arange(1, self.MAG + 1)))
    
        fig, ax = plt.subplots()
    
        ax.bar(x, P(W), width=1.0, color='black', align='center')
    
        ax.set_xticks(x)
        ax.set_xticklabels(pixels)
    
        # pixel boundaries
        ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
        ax.grid(which='minor', axis='x', color='gray', linewidth=0.5)
    
        ax.set_xlim(-0.5, N - 0.5)
    
        plt.show()

    #***********************************************************
    # Map generation
    #***********************************************************

    #========================================
    # OPTIMIZATION TARGETS:
    # 2x machineModel.getBucketHeight() -> machineModel.getBucketHeightFast()
    # 1x placeTestParticles() -> placeTestParticlesFast()
    # 1x machineModel.turnFor() -> now uses machineModel.getBucketHeightFast()
    # 1x createWeightMatrix()
    #========================================
    def createSingleMap(self, Es, y_abs, x_abs):
        state = State(Es, np.zeros(self.Nppp), np.zeros(self.Nppp))
        #state.ΔE_max = self.machineModel.getBucketHeight(state) # supposed to be the model's job but can't be helped here
        state.ΔE_max = self.machineModel.getBucketHeightFast(state.Es) # OPTIMIZED
        state.ΔX = self.ΔX_0
        state.ΔY = (2*state.ΔE_max)/self.RESOLUTION # calculate ΔY for this frame
        #state = self.placeTestParticlesFast(state, y_rel, x_rel)
        state = self.placeTestParticlesFast(state, y_abs, x_abs)

        #print('state.ΔE_max = %f' % state.ΔE_max)

        #self.plotWithGrid(state)

        #W0 = self.createWeightMatrix(state)
        #print(W0)
        #self.plotWeightMatrix(W0)
        #self.plotWeightMatrixProjection(W0)

        state = self.machineModel.turnFor(state, self.N, self.DIRECTION)
        #print('%d turns complete' % self.N)
        #state.ΔE_max = self.machineModel.getBucketHeight(state)
        state.ΔE_max = self.machineModel.getBucketHeightFast(state.Es) # OPTIMIZED
        state.ΔX = self.ΔX_0
        state.ΔY = (2*state.ΔE_max)/self.RESOLUTION # calculate ΔY for this frame

        #print('state.ΔE_max = %f' % state.ΔE_max)

        #self.plotWithGrid(state)

        W = self.createWeightMatrixCompact(state)
        #self.plotWeightMatrix(W)
        #self.plotWeightMatrixProjection(W)

        return W

    @time_it
    def createSingleFrameMaps(self, frame, Es_frame, y_start=0, x_start=0):
        '''print('FRAME[%d]: Generating %d %s maps...' % (frame, self.RESOLUTION**2, \
                                            'FORWARD' if self.DIRECTION == MapGenerator.FORWARD else 'BACKWARD'))'''

        W_frame = [] # holds transport matrices from this frame to the next
        
        # for each pixel in this frame
        for y_abs in range(y_start, self.RESOLUTION):
            for x_abs in range(x_start if y_abs==y_start else 0, self.RESOLUTION):
                #y_rel, x_rel = self._absCoordsToRelCoords(y_abs, x_abs)
                #print('Generating map for abs=(%d, %d), rel=(%d, %d)...' % (y_abs, x_abs, y_rel, x_rel))
                #self.W_all.append(self.createSingleMap(y_rel, x_rel))
                
                #W_frame.append(self.createSingleMap(Es_frame, y_rel, x_rel))
                W_frame.append(self.createSingleMap(Es_frame, y_abs, x_abs))
                
        #return W_frame # return instead of appending
        #self.W_all.append(W_frame)
        
        # safely append object
        # hopefully this doesn't drag the perofrmance too much (fingers crossed)
        _W_frame = np.empty(1, dtype=object)
        _W_frame[0] = W_frame
        self.W_all = np.append(self.W_all, _W_frame)

    def createAllFrameMaps(self, y_start=0, x_start=0):
        #assert self.W_all == [] # check no maps have been generated yet\
        
        if self.loadTransportMaps(): # use pre-generated maps if available
            return 

        #local_W_all = []

        if self.DIRECTION == MapGenerator.FORWARD:
            Es_frame = self.Es # set to initial sync. energy
            
            for frame in tqdm(range(0, self.FRAMES), 
                              'Generating %d FORWARD maps' % (self.FRAMES*(self.RESOLUTION**2))):
                self.createSingleFrameMaps(frame, Es_frame)
                #local_W_all.append(self.createSingleFrameMaps_Parallel(frame, Es_frame))
                Es_frame += self.machineModel.δE_turn(self.machineModel.ϕs) # increment energy for next frame
        elif self.DIRECTION == MapGenerator.BACKWARD:
            Es_frame = self.Es + (self.FRAMES * self.machineModel.δE_turn(self.machineModel.ϕs)) # set to energy on last frame
            
            for frame in tqdm(reversed(range(0, self.FRAMES)), 
                              'Generating %d BACKWARD maps' % (self.FRAMES*(self.RESOLUTION**2))):
                self.createSingleFrameMaps(frame, Es_frame)
                #local_W_all.append(self.createSingleFrameMaps_Parallel(frame, Es_frame))
                Es_frame -= self.machineModel.δE_turn(self.machineModel.ϕs) # decrement energy for next frame

        #self.W_all = np.array(self.W_all) # can numpy allocate this big of a contiguous memory block?
        #self.W_all = np.array(local_W_all)
        
        print('Successfully generated %d frames (%d maps)' % (self.FRAMES, self.FRAMES*(self.RESOLUTION**2)))
        size_bytes = (8 * (self.RESOLUTION**2) * (self.RESOLUTION**2) * self.FRAMES)
        print('Saving to \'%s\' - size: ~%.2f GB (%d bytes + α)' % (self.filepath, size_bytes / 1e9, size_bytes))
        self.saveTransportMaps()

    # search for a file containing the maps that matches the specified parameters
    # returns True on success, False on failure
    def loadTransportMaps(self):
        if not os.path.exists(self.filepath): # don't try to load if path doesn't exist
            return False

        print('Loading %s...' % (self.filepath))
        self.W_all = np.load(self.filepath, allow_pickle=True)
        return True

    # save the generated maps into a file
    def saveTransportMaps(self):
        #return # don't save maps for param sweeps
        print('Saving %s...' % (self.filepath))
        np.save(self.filepath, self.W_all)

    #=============================================================================================================
    
    # same as the transport maps, but for the back-projection matrices (they take forever to load on each run)
    def loadBackProjectionMatrices(self):
        if not os.path.exists(self.bp_filepath): # don't try to load if path doesn't exist
            return False

        print('Loading %s...' % (self.bp_filepath))
        self.p = np.load(self.bp_filepath)
        return True

    # save the generated maps into a file
    def saveBackProjectionMatrices(self):
        #return # dont save BP matrices for param sweep
        print('Saving %s...' % (self.bp_filepath))
        np.save(self.bp_filepath, self.p)

    #***********************************************************
    # Do one frame worth of machine turns using generated maps
    #***********************************************************

    # CLARIFICATION: getNextFrame() will take D0 from frame=frame to frame=frame+1 (for FORWARD)
    #                                                 frame=frame to frame=frame-1 (for BACKWARD)
    def getNextFrame(self, frame, D0):
        assert len(self.W_all) == self.FRAMES # make sure the maps have been generated
        assert D0.shape == (self.RESOLUTION, self.RESOLUTION)

        D = np.zeros_like(D0) # final distribution map

        for y_abs in range(0, self.RESOLUTION):
            for x_abs in range(0, self.RESOLUTION):
                d_i = D0[y_abs, x_abs]
                
                if d_i == 0: # skip if cell is empty
                    continue
                
                i = (y_abs*self.RESOLUTION) + x_abs

                # NEW
                if self.DIRECTION == MapGenerator.FORWARD:
                    index = frame
                    assert index >= 0
                elif self.DIRECTION == MapGenerator.BACKWARD:
                    #index = frame - 1
                    index = self.FRAMES - frame
                    assert index >= 0, 'index=%d is invalid for frame=%d' % (index, frame)
                W_i = self.W_all[index][i] # weight matrix for this pixel that takes D0 from this frame to the next

                # OLD
                #W_i = self.W_all[frame, i] # weight matrix for this pixel that takes D0 from this frame to the next         

                #assert W_i.shape == D.shape

                #D += W_i*d_i # distribute the value of this pixel according to the weight matrix
                
                # use W_entry to correctly transport the cell
                for n in range(len(W_i)):
                    W_entry = W_i[n]
                    assert len(W_entry) == 3
                    row_n, col_n = W_entry[0], W_entry[1]                    
                    val_n = W_entry[2]
                    assert (val_n > 0) and (val_n <= 1) 
                    D[row_n, col_n] += d_i * val_n
                    
        return D

    def getBinnedProjection(self, D):
        _P = P(D) # project distribution
        #assert sum(_P) == sum(D.flatten()) # ensure the projection contains the same integrated charge as the distribution
        '''print('D.sum = %s' % str(D.sum()))
        print('sum(P(D)) = %s' % str(sum(_P)))
        print('diff. = %s' % str(D.sum() - sum(_P)))'''
        assert np.allclose(sum(_P), sum(D.flatten()), rtol=1e-12, atol=1e-12) # use floating point comparison
        return _P

    def plotBinnedProjection(self, D):
        self.plotWeightMatrixProjection(D)

    #***********************************************************
    # Back-projection
    #***********************************************************
    # create a BP matrix for 'frame', where Es_frame = Es_recon + frame*δE(ϕs)
    def createBackProjectionMatrix(self, frame):
        TOTAL_PARTICLES = self.Nppp * self.NUM_PIXELS
        #Es_frame = self.machineModel.E_inj + (frame * self.machineModel.δE(self.machineModel.ϕs))
        Es_frame = self.Es + (frame * self.machineModel.δE_turn(self.machineModel.ϕs))
        
        uni_state = State(Es_frame, np.zeros(self.Nppp), np.zeros(self.Nppp)) # state of the uniformly distributed particles
        uni_state.ΔE_max = self.machineModel.getBucketHeightFast(uni_state.Es)
        uni_state.ΔX = self.ΔX_0
        uni_state.ΔY = (2*uni_state.ΔE_max)/self.RESOLUTION

        # this is the fixed point that lies on the separatrix
        ϕ0 = self.machineModel.right
        ΔE0 = 0
        H0 = self.machineModel.H(Es_frame, ΔE0, ϕ0) # Hamiltonian along the separatrix
        β = self.machineModel.β(uni_state.Es)
        η = self.machineModel.η(β)*self.machineModel.Npasses # η on this turn

        def _isParticleInBucket(ϕ, ΔE):
            H = self.machineModel.H(uni_state.Es, ΔE, ϕ)
            if η < 0: # below transition
                insideBucket = (H > H0)
            elif η > 0: # above transition
                insideBucket = (H < H0)
                #insideBucket = (H > H0)
                #print('Above transition, H0=%f, H=%f, insideBucket=%s' % (H0, H, str(insideBucket)))
            else: # precisely  at transition
                raise ValueError('η is zero, halting.')
                
            return insideBucket == True \
                and (ϕ >= self.machineModel.left) \
                and (ϕ <= self.machineModel.right) \
                and (ΔE <= uni_state.ΔE_max) \
                and (ΔE >= -uni_state.ΔE_max)

        p_frame = np.zeros((self.RESOLUTION, self.RESOLUTION))

        for x_abs in range(0, self.RESOLUTION): # column by column        
            for y_abs in range(0, self.RESOLUTION): # row by row
                #y_rel, x_rel = self._absCoordsToRelCoords(y_abs, x_abs)
                uni_state = self.placeTestParticlesFast(uni_state, y_abs, x_abs)

                # count how many particles are in the bucket for this pixel
                num_in = 0
                for i in range(0, self.Nppp):
                    '''H_i = self.machineModel.H(Es_frame, uni_state.ΔE[i], uni_state.ϕ[i])
                    if (H_i < H0) \
                        or (uni_state.ϕ[i] < self.machineModel.left) \
                        or (uni_state.ϕ[i] > self.machineModel.right):
                        continue # particle is not inside bucket'''
                    ϕ_i = uni_state.ϕ[i]
                    ΔE_i = uni_state.ΔE[i]
                    if not _isParticleInBucket(ϕ_i, ΔE_i):
                        continue
                    
                    num_in += 1
                    
                p_frame[y_abs, x_abs] = num_in / self.Nppp # fraction of pixel inside bucket
        p_frame = np.array(p_frame)
        self.p[frame] = p_frame.copy()

    def createAllFramesBackProjectionMatrices(self):
        if self.loadBackProjectionMatrices(): # use pre-generated ones if available
            return
        
        for i in tqdm(range(0, self.FRAMES+1), 'Generating %d back-projection matrices' % (self.FRAMES+1)):
            self.createBackProjectionMatrix(i)
            #self.plotWeightMatrix(self.p[i])
        self.saveBackProjectionMatrices()

    # normalizes
    def getBackProjection(self, Q, frame):
        assert Q.shape == (self.RESOLUTION,)
        assert self.p[frame].shape == (self.RESOLUTION, self.RESOLUTION)
        
        N = self.RESOLUTION
        
        P_in = np.zeros(N)
        qprime_N = np.zeros(N)

        q_N = (Q / N) # charge per pixel along projection path if all cells were fully inside bucket

        P_in = P(self.p[frame]*self.Nppp) / (N*self.Nppp) # fraction of particles inside bucket along projection path

        # charge per pixel (adjusted from simply Q/N such that once multiplied by the fraction of pixel inside bucket, p, the original total charge rem(ains)) # charge per pixel along projection path if all cells were inside bucket
        qprime_N = np.zeros_like(q_N)
        np.divide(q_N, P_in, out=qprime_N, where=(P_in > 0)) # only perform the division if P_in > 0

        D = np.zeros((N, N)) # backprojected distribution

        # smear j-th qprime_N across j-th projection path
        for j in range(0, N):
            D[:, j] = self.p[frame, :, j] * qprime_N[j]
        
        # ensure the integrated charge remains the same
        # update: i edited this out because for running buckets, creating a uniform distribution means backprojecting
        # 1's into bins with P_in=0, so a bin-by-bin comparison afterward would fail
        #assert np.allclose(P(D), Q, rtol=1e-12, atol=1e-12) # use floating point comparison

        #return self.normalizeBinnedDistribution(D) # normalize then return
        return D # DON'T NOMRALIZE OR CLIP NEGATIVES!!!

    def normalizeBinnedDistribution(self, D):
        D = D.copy()

        D[D < 0] = 0 # clip negatives

        total_charge = D.sum()

        assert total_charge != 0

        return D/total_charge