using SpeedyWeather, CairoMakie

spectral_grid = SpectralGrid(trunc=5, 
    nlayers=1,
    Grid=FullGaussianGrid
)
model = BarotropicModel(spectral_grid)
simulation = initialize!(model)

m = 4
ω = 7.848e-6
K = 7.848e-6

ζ(λ, θ, σ) = 2ω*sind(θ) - K*sind(θ)*cosd(θ)^m*(m^2 + 3m + 2)*cosd(m*λ)
set!(simulation, vor=ζ)

c = 1e-10       # cut-off amplitude

# 1 = first leapfrog timestep of spectral vorticity
vor = get_step(simulation.prognostic_variables.vor, 1)      # get the first leapfrog step
low_values = abs.(vor) .< c
vor[low_values] .= 0

# [:, 1, 1] for all values on first layer and first leapfrog step
vor = simulation.prognostic_variables.vor[:, 1, 1]
vor_grid = transform(vor)

CairoMakie.heatmap(vor_grid, title="Relative vorticity [1/s] of Rossby-Haurwitz wave")

run!(simulation, period=Hour(7*6.0))

# a running simulation always transforms spectral variables
# so we don't have to do the transform manually but just pull
# layer 1 (there's only 1) from the diagnostic variables
vor = simulation.diagnostic_variables.grid.vor_grid[:, 1]

CairoMakie.heatmap(vor, title="Relative vorticity [1/s], Rossby Haurwitz wave after 3 days")

#model.initial_conditions