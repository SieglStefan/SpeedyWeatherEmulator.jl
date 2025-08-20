module SimDataHandling

using SpeedyWeather
using JLD2
using Printf
using ..BasicStructs


export SimData, save_sim_data, load_sim_data, is_coeff_zero, calc_n_coeff, create_sim_data


function create_sim_data(sim_para::SimPara; overwrite::Bool=false)
    # Unpack simulation parameters
    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic
    n_spinup = sim_para.n_spinup
    t_step = sim_para.t_step
    initial_cond = sim_para.initial_cond
    
    
    # Create simulation data folder
    output_path, cancel = create_sim_folder(sim_para, overwrite=overwrite)
    
    if cancel
        # Do nothing, because overwriting data is not allowed
        @info "Calculation was canceled, because data already exists!"
        return nothing
    else
        # Create simulation data
        spectral_grid = SpectralGrid(trunc=trunc, nlayers=1)
        output = JLD2Output(output_dt = Hour(t_step), path=output_path, output_diagnostic=false)
        model = BarotropicModel(spectral_grid; output=output)

        # Forecast loop for different initial conditions
        for ic in 1:n_ic                                # looping over the number of initial conditions
            sim = initialize!(model)                    # initialize the model with new (random) initial conditions

            if initial_cond !== nothing
                set!(sim, vor=initial_cond)             # if specific IC are defined, set the simulation to them
            end

            t_max = (n_spinup + n_steps - 1) * t_step     # calculate the whole forecast time
            run!(sim, period=Hour(t_max), output=true)               # run the simulation
        end
    end

    return nothing
end


function create_sim_folder(sim_para::SimPara; overwrite::Bool=false)
    folderpath = create_sim_DIR(sim_para)

    cancel_sim = false
    if isdir(folderpath)
        @warn "Simulation data with trunc=$(sim_para.trunc), n_steps=$(sim_para.n_steps), n_ic=$(sim_para.n_ic) and key=$(sim_para.storage_key) already exists!"
        
        if overwrite
            rm(folderpath; recursive=true, force=true)
            mkpath(folderpath)
            @info "Folder $folderpath was overwritten."
        else
            @info "Folder $folderpath stays untouched. Set parameter overwrite::String = 'true' to overwrite the existing folder or use a key!"
            cancel_sim = true
        end
    else
        mkpath(folderpath)
        @info "Folder $folderpath was created."
    end

    return folderpath, cancel_sim
end




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
    # Unpack simulation parameters
    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic
    n_spinup = sim_para.n_spinup
    
    # Creating data array
    n_coeff = calc_n_coeff(trunc=trunc)
    data = zeros(Float32, 2*n_coeff, n_steps, n_ic)  

    file_folder = create_sim_DIR(sim_para)

    for ic in 1:n_ic
        # Create filepath
        file_subfolder = @sprintf("run_%04d", ic)
        filepath = normpath(joinpath(file_folder, file_subfolder, "output.jld2"))

        # Load file and storage data
        file = jldopen(filepath, "r")
        out = file["output_vector"]
        close(file)


        # 
        for step in n_spinup:1:n_spinup+n_steps-1
            prog, _ = out[step]
            vor = vec(prog.vor[:,:,1])

            data[1:n_coeff, step+1-n_spinup, ic] .= Float32.(real.(vor))
            data[n_coeff+1:2*n_coeff, step+1-n_spinup, ic] .= Float32.(imag.(vor))
        end
    end
        
    return SimData(sim_para, data)                  
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
    folder_path = create_sim_DIR(sim_data.sim_para)      # Create the storage DIR
    file = normpath(joinpath(folder_path, ".jld2"))

    jldsave(file; sim_data)                                 # Save the simulation data
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
    folder_path = create_sim_DIR(sim_para)      # Create the storage DIR
    file = normpath(joinpath(folder_path, ".jld2"))             # Create the storage DIR

    sim_data = JLD2.load(file, "sim_data")                  # Load the simulation data
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
function create_sim_DIR(sim_para::SimPara)
    dir = joinpath(@__DIR__, "..", "..", "data", "sim_data")
    filename = "sim_data_" *
        "T$(sim_para.trunc)_" *                                         
        "nsteps$(sim_para.n_steps)_" *
        "IC$(sim_para.n_ic)_" *
        "key$(sim_para.storage_key)"
    filepath = normpath(joinpath(dir, filename))

    return filepath
end


end