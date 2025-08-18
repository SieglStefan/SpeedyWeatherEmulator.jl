module SimDataHandling

using SpeedyWeather
using JLD2
using Logging


using ..GeneralUtils
using ..BasicStructs


export SimData, save_sim_data, load_sim_data, prepare_sim_data, is_coeff_zero


struct SimData
    sim_para::SimPara
    data::Array{Float32, 3}
end

function SimData(sim_para::SimPara;  
                            t_spinup::Real = 9.,    # spinup time (settle-in time)
                            t_step::Real = 6.,
                            initial_cond::Union{Nothing,Function}=nothing)       # in hours, min. 3h (timestep of T5 model)

    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    if initial_cond != nothing
        n_ic = 1
    else
        n_ic = sim_para.n_ic
    end

    # Basic simulation for getting the dimension
    spectral_grid = SpectralGrid(trunc=trunc, nlayers=1, Grid=FullGaussianGrid)
    model = BarotropicModel(spectral_grid)
    sim0 = initialize!(model) 

    n_vars = length(vec(sim0.prognostic_variables.vor[:,1,1]))  # number of complex spectral coeff.

    vor = zeros(Float32, 2*n_vars, n_steps, n_ic)              # data vector for storing the spectral coeff.
    

    # Forecast loop
    for ic in 1:n_ic
        sim = initialize!(model)             # initialize the model with new IC

        if initial_cond == nothing
            run!(sim, period=Hour(t_spinup))
            println(t_spinup)
        else
            set!(sim, vor=initial_cond)
        end
            
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


function load_sim_data(sim_para::SimPara)
    dir = joinpath(@__DIR__, "..", "..", "data", "sim_data")

    filename = "sim_data_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    sim_data = JLD2.load(filepath, "sim_data")
    @info "Simulation data $filepath loaded"

    return sim_data
end


function prepare_sim_data(sim_para::SimPara;  
                    t_spinup::Real = 9.,
                    t_step::Real = 6.)

    data = create_sim_data(sim_para, t_spinup=t_spinup, t_step=t_step)
    save_sim_data(data)
    return load_sim_data(sim_para)
end


function is_coeff_zero(i::Int, sim_data::SimulationData)
    data = sim_data.data
    return all(data[i, :, :] .== 0)
end



end