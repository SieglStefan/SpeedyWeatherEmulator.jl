using SpeedyWeather
using JLD2

include(joinpath(@__DIR__, "..", "utils", "utils.jl"))
using .DataUtils

# Parameter definition
const TRUNC = 5                     # parameter of model: e.g. T5


function create_training_data(; t_max::Real = 54.,       # maximal forecast length in hours (t_max is not overreached!)
                                t_step::Real = 6.,       # in hours, min. 3h (timestep of T5 model)
                                t_spinup::Real = 10.,    # spinup time (settle-in time)
                                n_ic::Integer = 10)         # number of Initial Conditions


    # Basic simulation for getting the dimension
    spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=1, Grid=FullGaussianGrid)
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

    return (n_steps, n_ic, vor_norm, μ, σ)
end


function save_training_data(n_steps::Integer, n_ic::Integer, data::Array{Float32, 3}, μ::Vector{Float32}, σ::Vector{Float32})
    dir = joinpath(@__DIR__, "..", "..", "data", "training_data")

    filename = "training_data_T$(TRUNC)_nsteps$(n_steps)_IC$(n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; data, μ, σ)
    @info "Normed training data saved at: $filepath"
end


training_data = create_training_data()
save_training_data(training_data...)
