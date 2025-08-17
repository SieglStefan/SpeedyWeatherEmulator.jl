using JLD2, SpeedyWeather, Statistics

include(joinpath(@__DIR__, "..", "utils", "utils.jl"))
using .DataUtils: get_vorticity!, zscore_trafo, inv_zscore_trafo
include(joinpath(@__DIR__, "..", "utils", "structs.jl"))
using .DataStructs: MyModel

trunc = 5
n_steps = 8
n_ic = 1000



# Load the model
dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    
filename = "model_T$(trunc)_nsteps$(n_steps)_IC$(n_ic).jld2"
filepath = normpath(joinpath(dir, filename))
file = load(filepath)

model = file["model"]
norm_stats = file["norm_stats"]



### 0) Initial conditions

m = 4
ω = 7.848e-6
K = 7.848e-6

ζ(λ, θ, σ) = 2ω*sind(θ) - K*sind(θ)*cosd(θ)^m*(m^2 + 3m + 2)*cosd(m*λ)



### 1) SpeedyWeather

 #Create simulation
spectral_grid = SpectralGrid(trunc=5, nlayers=1, Grid=FullGaussianGrid)
model_sw = BarotropicModel(spectral_grid)
sim = initialize!(model_sw)

# Set IC
#set!(sim, vor=ζ)

# Run simulation and get spectral coeff. for the NN
run!(sim, period=Hour(20))
vorA_SW = get_vorticity!(sim)

run!(sim, period=Hour(6))
vorB_SW = get_vorticity!(sim)



### 2) Neural Network

vorA_pre = zscore_trafo(vorA_SW, norm_stats)

vorA_post = model(vorA_pre)

vorB_NN = inv_zscore_trafo(vorA_post, norm_stats)



### 3) Comparison
rel_err = abs.(vorB_NN .- vorB_SW) ./ (abs.(vorB_SW) .+ eps(Float32)) .* 100
mean_err = mean(rel_err)
max_err  = maximum(rel_err)


println("-----------------------------------")

println("Mittlerer Fehler: ", mean_err, " %")
println("Max Fehler:       ", max_err, " %")

println("-----------------------------------")

for i in 1:54
    println("coeff ", i, ": ", rel_err[i])
end

println("-----------------------------------")

