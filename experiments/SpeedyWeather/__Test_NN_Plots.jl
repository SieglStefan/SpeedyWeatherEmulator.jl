using Flux, Statistics, ProgressMeter
using CUDA
using Plots
using JLD2
using SpeedyWeather

### LOADING MODEL

struct MyModel
    net
end
Flux.@layer MyModel

(m::MyModel)(x) = m.net(x)

MyModel() = MyModel(Chain(
    Dense(54 => 128, relu),
    Dense(128 => 128, relu),
    Dense(128 => 54)
)) |> device


model = MyModel()

# Loading model
model_state = JLD2.load("experiments/SpeedyWeather/model_T5_nsteps8_IC1000.jld2", "model_state")
µ = JLD2.load("experiments/SpeedyWeather/model_T5_nsteps8_IC1000.jld2", "mu")
σ = JLD2.load("experiments/SpeedyWeather/model_T5_nsteps8_IC1000.jld2", "σ")
Flux.loadmodel!(model.net, model_state)


### 0) INITIAL Conditions

m = 4
ω = 7.848e-6
K = 7.848e-6

ζ(λ, θ, σ) = 2ω*sind(θ) - K*sind(θ)*cosd(θ)^m*(m^2 + 3m + 2)*cosd(m*λ)


### 1) SPEEDYWEATHER

# Create model
spectral_grid = SpectralGrid(trunc=5, nlayers=1, Grid=FullGaussianGrid)
model_sw = BarotropicModel(spectral_grid)
sim = initialize!(model_sw)

# Set IC
set!(sim, vor=ζ)

# Get spectral coeff.
vorticity0 = vec(sim.prognostic_variables.vor[:,1,1])
n_vars = length(vorticity0)
vor00 = zeros(Float32, 2*n_vars)
vor00[1:n_vars] .= Float32.(real.(vorticity0))
vor00[n_vars+1:2*n_vars] .= Float32.(imag.(vorticity0))

run!(sim, period=Hour(6))
vorticity0 = vec(sim.prognostic_variables.vor[:,1,1])
n_vars = length(vorticity0)
vor0 = zeros(Float32, 2*n_vars)
vor0[1:n_vars] .= Float32.(real.(vorticity0))
vor0[n_vars+1:2*n_vars] .= Float32.(imag.(vorticity0))
vor_SW = vor0



### 2) NEURAL NETWORK

# z-score trafo:
v0_norm = (vor00 .- μ) ./ (σ .+ eps(Float32))

# Calculating coeff of model
coeff = model(v0_norm)

# Back z-score trafo
vor_NN = coeff .* (σ .+ eps(Float32)) .+ μ


# Comparison
rel_err = abs.(vor_NN .- vor_SW) ./ (abs.(vor_SW) .+ eps(Float32)) .* 100
mean_err = mean(rel_err)
max_err  = maximum(rel_err)

println("Mittlerer Fehler: ", mean_err, " %")
println("Max Fehler:       ", max_err, " %")



