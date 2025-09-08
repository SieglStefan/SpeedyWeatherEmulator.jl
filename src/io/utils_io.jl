using JLD2


"""
    data_path(sim_para::SimPara; type::String, path::String="")

Creates the folder or file path for storing data associated with `sim_para`.

# Arguments
- `sim_para::SimPara`: Simulation parameters (used to build unique name).
- `type::String`: Data type; `"raw_data"`, `"sim_data"`, `"emulator"` or `"losses"`.
- `path::String = ""`: Optional absolute path for data storage.  
    If left empty, the function defaults to the package's internal `data/<type>` folder.  

# Returns
- `::String`: Absolute normalized path to the storage location.

# Naming Convention
- For raw data: `/data/raw_data/T<trunc>_ndata<n_data>_IC<n_ic>_ID<id_key>/`
- For all other types: `/data/<type>/<type>_T<trunc>_ndata<n_data>_IC<n_ic>_ID<id_key>.jld2`

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200, id_key="demo")
data_path(sim_para; type="sim_data")
# → ".../data/sim_data/sim_data_T5_ndata50_IC200_IDdemo.jld2"
```
"""
function data_path(sim_para::SimPara; type::String, path::String="")
    
    # Default data path
    if path === ""
        folder = joinpath(@__DIR__, "..", "..", "data", type)
    # User-custom data path
    else
        folder = path
    end
    
    # If data type is in ["raw_data"] create a folder and no .jld2
    if type in ["raw_data"]
        file_ex = ""
    else
        file_ex = ".jld2"
    end
    
    # Create storing/loading name from simulation parameters
    name = type * "_" *
        "T$(sim_para.trunc)_" *                                         
        "ndata$(sim_para.n_data)_" *
        "IC$(sim_para.n_ic)_" *
        "ID$(sim_para.id_key)" * file_ex

    # Return data path
    return normpath(joinpath(folder, name))
end



"""
    delete_data(sim_para::SimPara; type::String, overwrite::Bool=false, path::String="")

Delete existing data of `type` `"raw_data"`, `"sim_data"`, `"emulator"` or `"losses"`

# Description
- Checks whether the path already exists.
- If `overwrite=true`: deletes existing content.
- If `overwrite=false`: keeps existing data untouched and sets `cancel_sim=true` to cancel current process.

# Arguments
- `sim_para::SimPara`: Simulation parameters (used for identifying data).
- `type::String`: Data type; `"raw_data"`, `"sim_data"`, `"emulator"`, `"losses"`.
- `overwrite::Bool = false`: Control overwrite behavior.
- `path::String = ""`: Optional absolute path for data storage.  
    If left empty, the function defaults to the package's internal `data/<type>` folder.  

# Returns
- `path::String`: Target folder/file path.
- `cancel_sim::Bool`: True if the current process must be stopped because data is not allowed to be overwritten.

# Notes
- For `"raw_data"`, creates a directory tree.
- For other types, returns the target `.jld2` path (no folder created).

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200, id_key="test")
delete_data(sim_para; type="sim_data")
```
"""
function delete_data(sim_para::SimPara; type::String, overwrite::Bool=false, path::String="")
    
    # Get data path
    data = data_path(sim_para, type=type, path=path)

    # Handle data deleting
    cancel_sim = false

    # File exists
    if isdir(data) || isfile(data)
        @warn "Data of type '" * type * "' and parameters trunc=$(sim_para.trunc), n_data=$(sim_para.n_data), n_ic=$(sim_para.n_ic) and ID=$(sim_para.id_key) already exists!"
        
        # If overwriting is allowed, delete data
        if overwrite
            rm(data; recursive=true, force=true)
            @info "Data '$data' was deleted."

        # If overwriting is not allowed, cancel process
        else
            @info "Data '$data' stays untouched. Set parameter 'overwrite::Bool = true' to overwrite the existing data or use a unique ID!"
            cancel_sim = true
        end

    # File does not exist
    else
        @info "There was no '$data' !"
    end

    # Return data path and cancel status
    return data, cancel_sim
end