using SpeedyWeather
using Statistics, JLD2


# Parameter definition
const t_max = 54                    # maximal forecast length in hours (t_max is not overreached!)
const t_step = 6                    # in hours, min. 3h (timestep of T5 model)
const t_spinup = 10                 # spinup time (settle-in time)
const trunc = 5                     # parameter of model: e.g. T5
const N_IC = 200                    # number of Initial Conditions


function saveVorticity!(data::Array{Float32, 3}, simulation, step::Int, ic::Int)
    vorticity_vec = vec(simulation.prognostic_variables.vor[:,1,1])
    
    data[1:n_vars, step, ic] .= Float32.(real.(vorticity_vec))
    data[n_vars+1:2*n_vars, step, ic] .= Float32.(imag.(vorticity_vec))
end


# Basic simulation for getting the dimension
spectral_grid = SpectralGrid(trunc=trunc, nlayers=1,Grid=FullGaussianGrid)
model = BarotropicModel(spectral_grid)
sim0 = initialize!(model) 

n_vars = length(vec(sim0.prognostic_variables.vor[:,1,1]))  # number of complex spectral coeff.


# Defining steps and data array
n_steps = Integer(floor((t_max - t_spinup) / t_step)) + 1   # number of time steps from t_spinup to t_max
data = zeros(Float32, 2*n_vars, n_steps, N_IC)              # data vector for storing the spectral coeff.


# Forecast loop
for ic in 1:N_IC
    simulation = initialize!(model)             # initialize the model with new IC

    run!(simulation, period=Hour(t_spinup))     # spinup simulation
    saveVorticity!(data, simulation, 1, ic)

    for step in 2:n_steps
        run!(simulation, period=Hour(t_step))
        saveVorticity!(data, simulation, step, ic)
    end
end


# Z-score transformation
µ = vec(mean(data, dims=(2,3)))
σ = vec(std(data, dims=(2,3)))

data_norm = (data .- μ) ./ (σ .+ eps(Float32))


# Creating training pairs x_i and x_{i+1}
@views X = data_norm[:, 1:end-1, :]
@views Y = data_norm[:, 2:end,   :]


# Saving
jldsave("experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(n_steps)_IC$(N_IC).jld2"; data_norm, µ, σ)
println("Normed training data saved at: 'experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(n_steps)_IC$(N_IC).jld2'")