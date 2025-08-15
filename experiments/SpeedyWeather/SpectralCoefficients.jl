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

vor = vec(simulation.prognostic_variables.vor[:,1,1])

println(vor)
println(length(vor))

display(fieldnames(typeof(simulation.prognostic_variables.random_pattern)))
















# Saving
jldsave("experiments/SpeedyWeather/training_data"; data)
println("Data saved")

# Loading 1
data1 = JLD2.load("experiments/SpeedyWeather/training_data", "data")
println("Data loaded")

# Loading 2
data = load("experiments/SpeedyWeather/training_data")
data2 = data["data"]

println("Data is the same: ", data1 == data2)
