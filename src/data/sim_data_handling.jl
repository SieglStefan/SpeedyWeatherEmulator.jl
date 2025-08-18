module SimDataHandling

using SpeedyWeather
using JLD2
using ..BasicStructs


export SimData, save_sim_data, load_sim_data, is_coeff_zero, calc_n_coeff


"""
    SimData(sim_para::SimPara, 
            data::Array{Float32,3})

Container for SpeedyWeather.jl simulation data and corresponding simulation parameter, see constructor for details.

# Fields
- `sim_para::SimPara`: Simulation parameters for data generation.
- `data::Array{Float32,3}`: Data storage, where `data` = (n_coeff) x `n_steps` x `n_ic`, where
    (n_coeff) is the number of spectral coefficients (real and imaginary part), e.g. for `trunc`=5: n_coeff=54
"""
struct SimData
    sim_para::SimPara
    data::Array{Float32, 3}
end


"""
    SimData(sim_para::SimPara)  

Convenience Constructor for SimData and vorticity data generation.

Corresponding to the simulation parameters in SimPara, this constructor:
    - defines a SpeedyWeather.jl simulation `sim`
    - runs it for different random initial conditions (if there is no other initial condition defined in SimPara)
    - stores the vorticity of the actual leap frog step in `data`, for array details see struct definition

# Arguments
- `sim_para::SimPara`: Simulation parameters for data generation.

# Examples
```julia
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000)
sim_data = SimData(sim_para)
```
"""
function SimData(sim_para::SimPara)

    # Get the parameters out of sim_para
    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic
    t_spinup = sim_para.t_spinup
    t_step = sim_para.t_step
    initial_cond = sim_para.initial_cond

    # Calculate number of complex spectral coefficients
    n_coeff = calc_n_coeff(trunc=trunc)

    # Define the data vector for storing the spectral coeff.
    data = zeros(Float32, 2*n_coeff, n_steps, n_ic)  
    
    # Generate the SpeedyWeather.jl model
    spectral_grid = SpectralGrid(trunc=5, nlayers=1,Grid=FullGaussianGrid)
    model = BarotropicModel(spectral_grid)


    # Forecast loop
    for ic in 1:n_ic                                # looping over the number of initial conditions
        sim = initialize!(model)                    # initialize the model with new (random) initial conditions

        if initial_cond === nothing                  
            run!(sim, period=Hour(t_spinup))       
        else
            set!(sim, vor=initial_cond)             # if specific IC are defined, set the model to them and dont spinup the simulation
        end
            
        get_vorticity!(data, sim, 1, ic)             # get the vorticity of the first step at the initial condition

        for step in 2:n_steps                       # loop over the remaining steps
            run!(sim, period=Hour(t_step))
            get_vorticity!(data, sim, step, ic)
        end
    end

    return SimData(sim_para, data)                  # return a SimData object
end


"""
    save_sim_data(sim_data::SimData)  

Saves simulation data in data/sim_data/ according to sim_data.sim_para.

# Arguments
- `sim_data::SimData`: Storaged data and simulation parameters

# Returns
- `nothing`

# Examples
```julia
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000)
sim_data = SimData(sim_para)
save_sim_data(sim_data)
```
"""
function save_sim_data(sim_data::SimData)
    filepath = create_sim_DIR(sim_para = sim_data.sim_para)      # Create the storage DIR

    jldsave(filepath; sim_data)                                 # Save the simulation data
    @info "Simulation data saved at: $filepath"                 # Information, if and where the data is saved

    return nothing
end


"""
    load_sim_data(sim_para::SimPara)  

Loads existing simulation data from data/sim_data/ according to sim_data.sim_para.

# Arguments
- `sim_para::SimPara`: Simulation parameters (key) of the stored simulation data

# Returns
- `SimData`: Returns the loaded simulation data

# Examples
```julia
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000)
sim_data = SimData(sim_para)
save_sim_data(sim_data)
...
sim_data_loaded = load_sim_data(sim_para)
```
"""
function load_sim_data(sim_para::SimPara)                        
    filepath = create_sim_DIR(sim_para = sim_para)              # Create the storage DIR

    sim_data = JLD2.load(filepath, "sim_data")                  # Load the simulation data
    @info "Simulation data $filepath loaded"                    # Information, if and from where the data is loaded

    return sim_data
end


"""
    create_sim_DIR(;sim_para::SimPara)  

Creates a DIR for saving and loading SimulationData according to `sim_para`.

# Arguments
- `sim_para::SimPara`: Simulation parameters (key) for the DIR generation.

# Returns
- `filepath::String`: Returns the filepath
"""
function create_sim_DIR(;sim_para::SimPara)
    dir = joinpath(@__DIR__, "..", "..", "data", "sim_data")
    filename = "sim_data_" *
        "T$(sim_para.trunc)_" *                                         
        "nsteps$(sim_para.n_steps)_" *
        "IC$(sim_para.n_ic)_" *
        "key$(sim_para.storage_key).jld2"
    filepath = normpath(joinpath(dir, filename))

    return filepath
end


"""
    is_coeff_zero(i::Int, sim_data::SimData)  

Checks if a specific spectral coefficient - numbered `i` - is always zero in the simulation data `sim_data`.

# Arguments
- `i::Int64`: Number of spectral coefficient.
- `sim_data::SimData`: Simulation data which is checked.

# Returns
- `is_zero::Bool`: Returns if they are all zero.
"""
function is_coeff_zero(i::Int64, sim_data::SimData)
    data = sim_data.data
    is_zero = all(data[i, :, :] .== 0f0)

    return is_zero
end


"""
    calc_n_coeff(;trunc::Int64)

Calculates the number of spectral coefficents of a specific truncation `trunc`.

# Arguments
- `trunc::Int64`: Number of spectral coefficient.

# Returns
- `n_coeff::Int64`: Number of spectral coefficients.
"""
function calc_n_coeff(;trunc::Int64)
    n_coeff = 0

    for i in 1:trunc+2
        n_coeff = n_coeff + i
    end

    return n_coeff-1
end


"""
    get_vorticity!(vor::Array{Float32,3}, sim, step::Int, ic::Int)

Accesses the spectral coefficents of SpeedyWeather.jl simulations.

Accesses the spectral coefficents of a SpeedyWeather.jl simulation `sim` and at step `step` and for initial condition `ic`
and stores it at vorticity array `vor`.
It stores specifically the current leapfrog step `sim.prognostic_variables.vor[:,1,1]`, where the third dimension denotes the 
current leapfrog step (`[:,1,2]` would denote the last leapfrog step)

# Arguments
- `vor::Array{Float32,3}`: Storaging array for the vorticity.
- `sim`: Current Simulation at step `step` and initial contion `ic`.
- `step::Int`: Current step in the data gen. loop.
- `ic::Int`: Current initial condition in the data gen. loop.

# Returns
- `nothing`
"""
function get_vorticity!(vor::Array{Float32,3}, sim, step::Int, ic::Int)
    vorticity = sim.prognostic_variables.vor[:,1,1]
    n_coeff = size(vorticity, 1)
    
    vor[1:n_coeff, step, ic] .= Float32.(real.(vorticity))              # real part of complex spectral coeff.
    vor[n_coeff+1:2*n_coeff, step, ic] .= Float32.(imag.(vorticity))    # imaginary part

    return nothing
end



end