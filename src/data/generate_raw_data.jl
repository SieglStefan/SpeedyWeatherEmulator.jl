using SpeedyWeather


"""
    generate_raw_data(sim_para::SimPara; overwrite::Bool=false)

Generate raw vorticity data with SpeedyWeather.jl based on the given simulation parameters.

# Description
- Prepares a raw-data folder using `prepare_folder`.
- Runs the barotropic model with spectral truncation `sim_para.trunc` and time step `sim_para.t_step`.
- For each initial condition (IC), creates a new run subfolder and stores the simulated vorticity.
- If `overwrite=false` and data already exist, generation is canceled.

# Arguments
- `sim_para::SimPara`: Container for parameters that define the simulation and data storage.
- `overwrite::Bool = false`: If true, delete existing data and regenerate.  
    If false, aborts safely when data already exist.

# Returns
- `nothing`: Data is written in the folder `data_path(sim_para; type="raw_data")`.

# Notes
- Spin-up steps (`n_spinup`) are run but **not stored** in later `SimData`.
- If `sim_para.initial_cond` is not `nothing`, this function applies the given IC via `set!(sim, vor=…)`.
- Sometimes overwriting `raw_data` files is not possible because the folders are open/busy.
    Data Generation is then canceled.
- Only prognostic variables are stored.

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200, id_key="123")
generate_raw_data(sim_para; overwrite=true)
# → creates data/raw_data/raw_data_T5_ndata50_IC200_ID123/run_0001/output.jld2, ..., run_0200/output.jld2
```
"""
function generate_raw_data(sim_para::SimPara; overwrite::Bool=false)
    # Unpack simulation parameters
    trunc = sim_para.trunc
    n_data = sim_para.n_data
    n_ic = sim_para.n_ic
    n_spinup = sim_para.n_spinup
    t_step = sim_para.t_step
    initial_cond = sim_para.initial_cond
    
    
    # Delete old raw_data folder
    output_path, cancel = delete_data(sim_para, overwrite=overwrite, type="raw_data")

    
    if cancel
        # Do nothing, because overwriting data is not allowed
        @error "Raw data generation was canceled, because data already exists and overwriting is not allowed!"
        return nothing
    elseif cancel == false && (isdir(output_path) || isfile(output_path))
        # Deleting has NOT worked: abort
        @error "Raw data generation was canceled, because already existing data could not be deleted!"
        return nothing        
    else
        # Create new folder
        mkpath(output_path) 
        @info "New folder $output_path was created."

        # Create simulation data
        spectral_grid = SpectralGrid(trunc=trunc, nlayers=1)
        output = JLD2Output(output_dt = Hour(t_step), path=output_path, output_diagnostic=false)
        model = BarotropicModel(spectral_grid; output=output)

        # Forecast loop for different initial conditions
        for _ in 1:n_ic                                # looping over the number of initial conditions
            sim = initialize!(model)                    # initialize the model with new (random) initial conditions

            if initial_cond !== nothing
                set!(sim, vor=initial_cond)             # if specific IC are defined, set the simulation to them
            end

            t_max = (n_spinup + n_data - 1) * t_step     # calculate the whole forecast time
            run!(sim, period=Hour(t_max), output=true)               # run the simulation
        end
    end

    return nothing
end