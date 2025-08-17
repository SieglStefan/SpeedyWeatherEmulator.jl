using SpeedyWeather
using JLD2

include(joinpath(@__DIR__, "..", "utils", "utils.jl"))
using .DataUtils
include(joinpath(@__DIR__, "..", "utils", "structs.jl"))
using .DataStructs: SimPara, NormStats

# Parameter definition
const TRUNC = 5                     # parameter of model: e.g. T5


function create_sw_data(;       t_max::Real = 54.,       # maximal forecast length in hours (t_max is not overreached!)
                                t_step::Real = 6.,       # in hours, min. 3h (timestep of T5 model)
                                t_spinup::Real = 10.,    # spinup time (settle-in time)
                                n_ic::Integer = 1000,
                                trunc::Integer = TRUNC)         # number of Initial Conditions


    # Basic simulation for getting the dimension
    spectral_grid = SpectralGrid(trunc=trunc, nlayers=1, Grid=FullGaussianGrid)
    model = BarotropicModel(spectral_grid)
    sim0 = initialize!(model) 

    n_vars = length(vec(sim0.prognostic_variables.vor[:,1,1]))  # number of complex spectral coeff.
    n_steps = Integer(floor((t_max - t_spinup) / t_step)) + 1   # number of time steps from t_spinup to t_max

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


    # Z-score transformation
    vor_norm, μ, σ = zscore_trafo(vor)

    sim_para = SimPara(trunc, n_steps, n_ic)
    norm_stats = NormStats(μ, σ)

    return sim_para, vor_norm, norm_stats
end


function save_sw_data(sim_para::SimPara, vor_norm::Array{Float32,3}, norm_stats::NormStats)
    dir = joinpath(@__DIR__, "..", "..", "data", "training_data")

    filename = "training_data_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; sim_para, vor_norm, norm_stats)
    @info "Normed training data saved at: $filepath"
end


sim_para, vor_norm, norm_stats = create_sw_data()
save_sw_data(sim_para, vor_norm, norm_stats)
