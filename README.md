# Longitudinal Phase Space Tomography for the Fermilab Booster



## Description
Code to reconstruct the longitudinal phase space distribution of an arbitrary Booster bunch from a Wall Current Monitor (WCM) measurement.

## Usage
Download & place all files into a folder named 'booster_tomography', then import into your code as shown in the examples below. 


## Input Data Format
The code expects the input sinograms to be .CSV files centered on the synchronous phase of the bunch. Each row should represent a turn and each column a .2ns WCM measurement. The sinogram should at least span the full width of stable region.

## Examples

### Reconstruct a single bunch

```Python
import booster_tomography as bt

# set machine parameters here
machine_params = bt.MachineParameters(
    ϕs=deg2rad(5.2),
    V=250e3,
    Es=1446.3e6
)

# set reconstruction parameters here
params = bt.Parameters(
    machine=machine_params,
    L=64,
    N_iter=25
)

# create the input sinogram object
sinogram = bt.InputSinogram('wcm_t3ms_b03_v1pad.csv', params)

# create the tomography engine object
tomoEngine = bt.BoosterTomography(params, sinogram)

# perform the reconstruction
result = tomoEngine.reconstruct()

# plot the reult and its sinogram
result.plot()
result.plotSinogram()
```

### Reconstruct multiple bunches

```Python
import booster_tomography as bt

# set machine parameters here
machine_params = bt.MachineParameters(
    ϕs=deg2rad(5.2), # rad
    V=250e3, # V
    Es=1446.3e6 # eV
)

# set reconstruction parameters here
params = bt.Parameters(
    machine=machine_params,
    L=64, # number of bins (i.e. LxL image)
    N_iter=25 # number of reconstruction iterations
)

# create the input sinogram object
sinogram = bt.InputSinogram('wcm_t3ms_b03_v1pad.csv', params)

# create the tomography engine object
tomoEngine = bt.BoosterTomography(params, sinogram)

# perform the reconstruction
result = tomoEngine.reconstruct()

# plot and save the results
result.plot(title='Reconstruction, ITER=%d' % params.N_iter, save_figure=True)
result.plotSinogram(title='Reconstruction Sinogram', save_figure=True)

#----------------------------------------------------
# different bunch
#----------------------------------------------------

params.machine = bt.MachineParameters(
    ϕs=deg2rad(20.9),
    V=250e3,
    Es=4374.3e6
)

# new data
sinogram = bt.InputSinogram('wcm_t15ms_b39_v1pad.csv', params)

# reset to new parameters and input
tomoEngine.setParameters(params)
tomoEngine.setInputSinogram(sinogram)

result = tomoEngine.reconstruct()

# plot but don't save the results
result.plot()
result.plotSinogram()
```