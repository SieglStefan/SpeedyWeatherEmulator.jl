using JLD2



"""
    save_data(data::Union{SimData, Emulator, Losses}; overwrite::Bool=false)

Save simulation- or training-related data (`SimData`, `Emulator`, `Losses`) using JLD2.

# Description
- Builds a consistent file/folder path from the associated `SimPara` via `data_path` for saving `data`.
- Prevents overwriting unless `overwrite=true`.
- Uses JLD2 to serialize the given `data`.

# Arguments
- `data::Union{SimData, Emulator, Losses}`: The container to save.  
  Must have a field `sim_para::SimPara`.
- `overwrite::Bool = false`: If true, existing file/folder is deleted before writing.
- `path::String = ""`: Optional absolute path for data saving.
    If left empty, the function defaults to the package's internal `data/<type>` folder.  

# Returns
- `nothing`: Data is written to the file system.

# Notes
- For `raw_data`: Creates a directory tree with subfolders `run_0001`, `run_0002`, ….
- For all other types: Saves to a single `.jld2` file.

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200, id_key="demo")
sim_data = SimData(sim_para)
save_data(sim_data; type="sim_data", overwrite=true)
```
"""
function save_data(data::Union{SimData, Emulator, Losses}; overwrite::Bool=false, path::String="")
    
    data_type = typeof(data)

    if data_type == SimData
        type = "sim_data"
    elseif data_type == Emulator
        type = "emulator"
    elseif data_type == Losses
        type = "losses"
    else
        @warn "Data type '$data_type' is not supported!"
    end

    sim_para = data.sim_para
    data_path, cancel_sim = delete_data(sim_para, overwrite=overwrite, type=type, path=path)

    if cancel_sim
        @error "Data saving was canceled, because data already exists!"
        return nothing
    else

        file_path = normpath(joinpath(data_path))

        jldsave(file_path; data)                                 # Save the simulation data
        @info "Data of type '" * type * "' and parameters trunc=$(sim_para.trunc), n_data=$(sim_para.n_data), n_ic=$(sim_para.n_ic) and ID=$(sim_para.id_key) was saved!"

        return nothing
    end
end


"""
    load_data(sim_para::SimPara; type::String)

Load previously saved data of a given type using the defining simulation parameters.

# Arguments
- `sim_para::SimPara`: Simulation parameters; determines the folder/file name.
- `type::String`: Dataset type, e.g. `"sim_data"`, `"emulator"`, `"losses"`.
- `path::String = ""`: Optional absolute path for data loading.
    If left empty, the function defaults to the package's internal `data/<type>` folder.  

# Returns
- `::Union{SimData, Emulator, Losses}`: The saved object stored in the JLD2 file under the key `"data"`.  
  (For example a `SimData`, `Emulator`, or `Losses` object.)

# Notes
- `load_data` is intended for JLD2-based single-file storage types.

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200)
sim_data_loaded = load_data(sim_para; type="sim_data")
```
"""
function load_data(sim_para::SimPara; type::String, path::String="")     
    path = data_path(sim_para, type=type, path=path)
    
    file_path = normpath(joinpath(path))             # Create the storage DIR

    data = JLD2.load(file_path, "data")                  # Load the simulation data
    @info "Data '$file_path' was loaded!"                    # Information, if and from where the data is loaded

    return data
end



