using SpeedyWeather, Plots, JLD2


# Create data
spectral_grid = SpectralGrid(trunc=5, nlayers=1,Grid=FullGaussianGrid)
model = BarotropicModel(spectral_grid)
simulation = initialize!(model)

n_steps = 5    # number of time steps
n_vars = 27     # complex spectral coefficients for T5
time_step = 6   # in hours, min. 3h (timestep of T5 model)

data = zeros(Float32, n_steps, 2*n_vars)

data[1, 1:n_vars] .= Float32.(real.(vec(simulation.prognostic_variables.vor[:,1,1])))
data[1, n_vars+1:2*n_vars] .= Float32.(imag.(vec(simulation.prognostic_variables.vor[:,1,1])))

for step in 2:n_steps
    run!(simulation, period=Hour(time_step))
    vort = vec(simulation.prognostic_variables.vor[:,1,1])
    data[step, 1:n_vars] .= Float32.(real.(vec(simulation.prognostic_variables.vor[:,1,1])))
    data[step, n_vars+1:2*n_vars] .= Float32.(imag.(vec(simulation.prognostic_variables.vor[:,1,1])))
end


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


# Plotting coefficients
fig = plot(1:n_steps, data1[:,1])

for i in 2:2*n_vars
    plot!(0:time_step:time_step*(n_steps-1), data1[:,i], label=false)
end

fig

println(length(vec(simulation.prognostic_variables.vor[:,1,1])))