module BasicStructs



export SimPara


"""
    SimPara(trunc::Int64, 
            n_steps::Int64, 
            n_ic::Int64, 
            t_spinup::Float32, 
            t_step::Float32, 
            initial_cond::Union{Nothing, Function}, 
            storage_key::String)

Container for basic simulation parameters for SpeedyWeather.jl simulation data generation, see constructor for details.
"""
struct SimPara
    trunc::Int64
    n_steps::Int64
    n_ic::Int64
    t_spinup::Float32
    t_step::Float32
    initial_cond::Union{Nothing, Function}
    storage_key::String
end

"""
    SimPara(;   trunc::Int64, 
                n_steps::Int64, 
                n_ic::Int64, 
                t_spinup::Float32 = 12f0, 
                t_step::Float32 = 6f0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                storage_key::String = "")

Convenience Constructor for SimPara

# Fields
- `trunc::Int64`: Spectral truncation of the barotropic model (e.g. 5 for T5).
- `n_steps::Int64`: Number of data points after the spin-up time, with the first at t = t_spinup.
- `n_ic::Int64`: Number of simulated initial conditions.

- `t_step::Float32`: Time step length.
- `t_spinup::Float32`: Time after which data generation begins.
- `initial_cond::Union{Nothing, Function}`: Specific initial condition, if empty: random initial conditions are applied.
- `storage_key::String`: Key for accessing saved simulation data.
    Data is saved with the parameters trunc, n_steps and n_ic. Key is for further differentiation of simulation data.

# Examples
```julia
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000, storage_key="test")
"""
function SimPara(;trunc::Int64, 
                n_steps::Int64, 
                n_ic::Int64, 
                t_spinup::Float32 = 12f0, 
                t_step::Float32 = 6f0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                storage_key::String = "")

    return SimPara(trunc, n_steps, n_ic, t_spinup, t_step, initial_cond, storage_key)
end



end