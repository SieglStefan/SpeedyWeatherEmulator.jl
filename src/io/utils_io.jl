using JLD2


"""
    data_path(sim_para::SimPara; type::String)

Creates the folder or file path for storing data associated with `sim_para`.

# Arguments
- `sim_para::SimPara`: Simulation parameters (used to build unique name).
- `type::String`: Data type; `"raw_data"`, `"sim_data"`, `"emulator"` or `"losses"`.

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
function data_path(sim_para::SimPara; type::String)
    folder = joinpath(@__DIR__, "..", "..", "data", type)
    
    if type in ["raw_data"]
        file_ex = ""
    else
        file_ex = ".jld2"
    end
    
    name = type * "_" *
        "T$(sim_para.trunc)_" *                                         
        "ndata$(sim_para.n_data)_" *
        "IC$(sim_para.n_ic)_" *
        "ID$(sim_para.id_key)" * file_ex

    return normpath(joinpath(folder, name))
end



"""
    delete_data(sim_para::SimPara; type::String, overwrite::Bool=false)

Delete existing data of `type` `"raw_data"`, `"sim_data"`, `"emulator"` or `"losses"`

# Description
- Checks whether the path already exists.
- If `overwrite=true`: deletes existing content and creates a new folder if `type = "raw_data"`.
- If `overwrite=false`: keeps existing data untouched and sets `cancel_sim=true` to cancel current process.

# Arguments
- `sim_para::SimPara`: Simulation parameters (used for identifying data).
- `type::String`: Data type; `"raw_data"`, `"sim_data"`, `"emulator"`, `"losses"`.
- `overwrite::Bool = false`: Control overwrite behavior.

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
function delete_data(sim_para::SimPara; type::String, overwrite::Bool=false)
    data = data_path(sim_para, type=type)

    cancel_sim = false
    if isdir(data) || isfile(data)
        @warn "Data of type '" * type * "' and parameters trunc=$(sim_para.trunc), n_data=$(sim_para.n_data), n_ic=$(sim_para.n_ic) and ID=$(sim_para.id_key) already exists!"
        
        if overwrite
            rm(data; recursive=true, force=true)
            @info "Data '$data' was deleted."
        else
            @info "Data '$data' stays untouched. Set parameter 'overwrite::Bool = true' to overwrite the existing data or use a unique ID!"
            cancel_sim = true
        end
        
    else
        @info "There was no '$data' !"
    end

    return data, cancel_sim
end