using SpeedyWeather, LinearAlgebra

spectral_grid = SpectralGrid(
    trunc=5, 
    nlayers=1,
    Grid=FullGaussianGrid
)
model = BarotropicModel(spectral_grid)
simulation = initialize!(model)
#run!(simulation, period=Day(0))


fieldnames(typeof(simulation))
display(fieldnames(typeof(simulation.prognostic_variables)))

println("")
println("Vorticity: ")
simulation.prognostic_variables.vor

println("")
println("Grid: ")
display(simulation.prognostic_variables.grid)
println("")

vor = vec(simulation.prognostic_variables.vor[:,1,1])

simulation.prognostic_variables.vor[:,1,1]



