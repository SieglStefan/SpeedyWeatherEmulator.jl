using SpeedyWeather
using Plots


# Parameter definition
const t_max = 54                   # maximal forecast length in hours
const t_step = 6                    # in hours, min. 3h (timestep of T5 model)
const trunc = 5                     # parameter of model: e.g. T5


# Simulation for getting the dimension
spectral_grid = SpectralGrid(trunc=trunc, nlayers=1, Grid=FullGaussianGrid)
model = BarotropicModel(spectral_grid)
simulation = initialize!(model) 

n_vars = length(vec(sim0.prognostic_variables.vor[:,1,1]))  # number of complex spectral coeff.


n_steps = Integer(floor(t_max / t_step))+1
data = zeros(Float32, 2*n_vars, n_steps)              


for step in 1:n_steps
    run!(simulation, period=Hour(t_step))
    vorticity_vec = vec(simulation.prognostic_variables.vor[:,1,1])

    data[1:n_vars, step] .= Float32.(real.(vorticity_vec))
    data[n_vars+1:2*n_vars, step] .= Float32.(imag.(vorticity_vec))
end


fig = Plots.plot(0:t_step:t_max, data[1,:], labels=false)

for i in 2:2*n_vars
    Plots.plot!(0:t_step:t_max, data[i,:], labels=false)
end

fig