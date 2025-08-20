module BasicStructs


export SimPara


"""
    SimPara(trunc::Int, 
            n_steps::Int, 
            n_ic::Int, 
            n_spinup::Int, 
            t_step::Real, 
            initial_cond::Union{Nothing, Function}, 
            id_key::String)

Container for basic simulation parameters for SpeedyWeather.jl simulation data generation and data shape, see constructor for details.
"""
struct SimPara
    trunc::Int
    n_steps::Int
    n_ic::Int
    n_spinup::Int
    t_step::Real
    initial_cond::Union{Nothing, Function}
    id_key::String
end


"""
    SimPara(;   trunc::Int, 
                n_steps::Int, 
                n_ic::Int, 
                n_spinup::Int = 10, 
                t_step::Real = 1.0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                id_key::String = "")

Convenience Constructor for SimPara

# Fields
- `trunc::Int`: Spectral truncation of the barotropic model (e.g. 5 for T5).
- `n_steps::Int`: Number of data points after the spin-up time, with the first at t = t_spinup.
- `n_ic::Int`: Number of simulated initial conditions.
- `n_spinup::Int = 10`: Time after data is stored in `SimData`
- `t_step::Real = 1.0`: Time step length.
- `initial_cond::Union{Nothing, Function} = nothing`: Specific initial condition, if empty: random initial conditions are applied.
- `id_key::String = ""`: Key for accessing saved simulation data.
    Data is saved with the parameters trunc, n_steps and n_ic. Key is for further differentiation of simulation data.

# Examples
```julia
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000, storage_key="test")
"""
function SimPara(;trunc::Int, 
                n_steps::Int, 
                n_ic::Int, 
                n_spinup::Int = 10, 
                t_step::Real = 1.0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                id_key::String = "")

    return SimPara(trunc, n_steps, n_ic, n_spinup, t_step, initial_cond, id_key)
end



end

