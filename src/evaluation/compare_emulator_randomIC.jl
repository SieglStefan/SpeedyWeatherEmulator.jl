### WRITE TO FUNCTION

using JLD2, SpeedyWeather, Statistics

#include(joinpath(@__DIR__, "..", "utils", "BasicStructs.jl"))
using .BasicStructs
#include(joinpath(@__DIR__, "..", "utils", "ZScoreTrafoUtils.jl"))
using .ZScoreTrafoUtils
#include(joinpath(@__DIR__, "..", "utils", "SpeedyWeatherUtils.jl"))
using .SpeedyWeatherUtils
#include(joinpath(@__DIR__, "..", "utils", "EmulatorUtils.jl"))
using .EmulatorUtils


function create_sim_data(sim_para::SimPara;  
                            t_spinup::Real = 10.,    # spinup time (settle-in time)
                            t_step::Real = 6.)       # in hours, min. 3h (timestep of T5 model)

    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic

    # Basic simulation for getting the dimension
    spectral_grid = SpectralGrid(trunc=trunc, nlayers=1, Grid=FullGaussianGrid)
    model = BarotropicModel(spectral_grid)
    sim0 = initialize!(model) 

    n_vars = length(vec(sim0.prognostic_variables.vor[:,1,1]))  # number of complex spectral coeff.
    
    @info "Max time generation length: $(t_spinup + n_steps*t_step) h !"

    vor = zeros(Float32, 2*n_vars, n_steps, n_ic)              # data vector for storing the spectral coeff.


    # Forecast loop
    for ic in 1:n_ic
        sim = initialize!(model)             # initialize the model with new IC

        run!(sim, period=Hour(t_spinup))     # spinup simulation
        get_vorticity!(vor, sim, 1, ic)

        for step in 2:n_steps
            run!(sim, period=Hour(t_step))
            get_vorticity!(vor, sim, step, ic)
        end
    end


    sim_para = SimPara(trunc, n_steps, n_ic)

    return SimulationData(sim_para, vor)
end


function save_sim_data(sim_data::SimulationData)
    dir = joinpath(@__DIR__, "..", "..", "data", "sim_data")

    sim_para = sim_data.sim_para

    filename = "sim_data_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; sim_data)
    @info "Simulation data saved at: $filepath"
end



sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000)
model, norm_stats = load_neuralnetwork(sim_para)

sim_para_comp = SimPara(trunc=5, n_steps=8, n_ic=1001)
sim_data_comp = create_sim_data(sim_para_comp)
save_sim_data(sim_data_comp)


function compare_emulator(model, norm_stats, n_ic=100)
    


td = TrainingData(sim_para_comp, split=1.0, norm_stats=norm_stats)


### 2) Neural Network
vorA_NN = model(td.data_pairs.x_train)
vorA_SW = td.data_pairs.y_train

vorB_NN = inv_zscore_trafo(vorA_NN, norm_stats)
vorB_SW = inv_zscore_trafo(vorA_SW, norm_stats)


### 3) Comparison
rel_err = abs.(vorB_NN .- vorB_SW) ./ (abs.(vorB_SW) .+ eps(Float32)) .* 100
mean_err = vec(mean(rel_err, dims=2))

mean_mean_err = mean(mean_err)
max_mean_err = maximum(mean_err)


println("-----------------------------------")

println("Mittlerer Fehler: ", mean_mean_err, " %")
println("Max Fehler:       ", max_mean_err, " %")

println("-----------------------------------")

for i in 1:54
    println("coeff ", i, ": ", mean_err[i])
end

println("-----------------------------------")