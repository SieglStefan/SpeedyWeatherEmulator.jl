using SpeedyWeather

# Creating the grid
spectral_grid = SpectralGrid(trunc=5, nlayers=1,Grid=FullGaussianGrid)

# Creating the model
model = BarotropicModel(spectral_grid)

# Initialize the simulation
simulation = initialize!(model)

model.time_stepping