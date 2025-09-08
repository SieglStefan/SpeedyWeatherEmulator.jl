using Printf


"""
    SimData{F, A<:AbstractArray{Float32, 3}}

Container for SpeedyWeather.jl vorticity data together with the simulation parameters.

# Fields
- `sim_para::SimPara{F}`: Container for parameters that define the simulation and data storage.
- `data::A`: Vorticity tensor with shape (2 * `n_coeff`, `n_data`, `n_ic`)
        where `n_coeff = calc_n_coeff(sim_para.trunc)`. 
        The first `n_coeff` rows store the **real** parts, 
        the next `n_coeff` rows the **imaginary** parts of the complex spectral coefficients.

# Notes
- The layout is column-major and optimized for contiguous slicing over time and ICs.
- Dimensions are inferred from `sim_para` and remain consistent across the pipeline.
"""
struct SimData{F, A<:AbstractArray{Float32, 3}}
    sim_para::SimPara{F}
    data::A
end


"""
    SimData(sim_para::SimPara, path::String="")

Construct a `SimData` container by **loading previously generated raw data** and
extracting the spectral vorticity coefficients time series into a consistent tensor layout.

This constructor:
1. infers `(n_coeff, n_data, n_ic)` from `sim_para`,
2. allocates the target array `data::Array{Float32,3}` with shape (2 * `n_coeff`, `n_data`, `n_ic`),
3. iterates over runs `ic = 1:n_ic`, loads `output.jld2`, and reads `output_vector`,
4. for each stored step `step ∈ {n_spinup+1, …, n_spinup + n_data}`:
   - extracts the spectral vorticity `vor`,
   - writes `real(vor)` to rows `1:n_coeff` and `imag(vor)` to rows `n_coeff+1:2n_coeff`,
   - stores at time index `step + 1 - n_spinup`.

# Arguments
- `sim_para::SimPara{F}`: Container for parameters that define the simulation and data storage; 
    **must match** the generated raw data on disk.
- `path::String = ""`: Optional absolute path of storaged `raw_data`.  
    If left empty, the function defaults to the package's internal `data/<type>` folder.

# Returns
- `::SimData{F, Array{Float32,3}}`: Container holding simulation data and corresponding sim. parameters.

# Preconditions
- Expects raw data in `data_path(sim_para; type="raw_data")` with per-run subfolders
    `run_0001`, `run_0002`, … each containing `output.jld2` with an `output_vector`.
- Raw data should be created beforehand via `generate_raw_data(sim_para; overwrite=false)`.

# Notes
- The leading factor `2` in the first dimension stacks real and imaginary parts.
- The time indexing uses `step - n_spinup` to map stored steps to `1:n_data`.

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200)
# after generate_raw_data(sim_para) has been called:
sim_data = SimData(sim_para)
# inspect shapes
size(sim_data.data)  # (2*n_coeff, n_data, n_ic)
```
"""
function SimData(sim_para::SimPara, path::String="")

    # Unpack simulation parameters
    (; trunc, n_data, n_ic, n_spinup) = sim_para 
    
    # Creating data array
    n_coeff = calc_n_coeff(trunc=trunc)
    data = zeros(Float32, 2*n_coeff, n_data, n_ic)  

    # Extracting vorticity data from generated raw data
    raw_data_folder = data_path(sim_para, type="raw_data", path=path)

    for ic in 1:n_ic
        # Create filepath
        file_subfolder = @sprintf("run_%04d", ic)
        filepath = normpath(joinpath(raw_data_folder, file_subfolder, "output.jld2"))

        # Load file and storage data
        file = jldopen(filepath, "r")
        out = file["output_vector"]
        close(file)

        # Scaling factor
        sc = Float32(out[1].scale[])

        # Extracting n_data datapoints per initial condition
        for step in n_spinup+1:n_spinup+n_data
            prog = out[step]
            vor = vec(prog.vor[:,:,1] ./sc)

            data[1:n_coeff, step-n_spinup, ic] .= Float32.(real.(vor))
            data[n_coeff+1:2*n_coeff, step-n_spinup, ic] .= Float32.(imag.(vor))
        end
    end
        
    return SimData(sim_para, data)                  
end