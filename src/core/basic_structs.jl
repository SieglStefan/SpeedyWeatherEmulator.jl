module BasicStructs


export SimPara


"""
    SimPara(trunc::Int, 
            n_steps::Int, 
            n_ic::Int, 
            t_spinup::Real, 
            t_step::Real, 
            initial_cond::Union{Nothing, Function}, 
            storage_key::String)

Container for basic simulation parameters for SpeedyWeather.jl simulation data generation and data shape, see constructor for details.
"""
struct SimPara
    trunc::Int
    n_steps::Int
    n_ic::Int
    t_spinup::Real
    t_step::Real
    initial_cond::Union{Nothing, Function}
    storage_key::String
end


"""
    SimPara(;   trunc::Int, 
                n_steps::Int, 
                n_ic::Int, 
                t_spinup::Real = 12.0, 
                t_step::Real = 6.0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                storage_key::String = "")

Convenience Constructor for SimPara

# Fields
- `trunc::Int`: Spectral truncation of the barotropic model (e.g. 5 for T5).
- `n_steps::Int`: Number of data points after the spin-up time, with the first at t = t_spinup.
- `n_ic::Int`: Number of simulated initial conditions.

- `t_step::Real`: Time step length.
- `t_spinup::Real`: Time after which data generation begins.
- `initial_cond::Union{Nothing, Function}`: Specific initial condition, if empty: random initial conditions are applied.
- `storage_key::String`: Key for accessing saved simulation data.
    Data is saved with the parameters trunc, n_steps and n_ic. Key is for further differentiation of simulation data.

# Examples
```julia
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000, storage_key="test")
"""
function SimPara(;trunc::Int, 
                n_steps::Int, 
                n_ic::Int, 
                t_spinup::Real = 12.0, 
                t_step::Real = 6.0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                storage_key::String = "")

    return SimPara(trunc, n_steps, n_ic, t_spinup, t_step, initial_cond, storage_key)
end



end