
"""
    SimPara

Container for core simulation parameters that define SpeedyWeather.jl data generation, 
    resulting data shape and storage name.

# Fields
- `trunc::Int`: Spectral truncation of the barotropic model (e.g. 5 for T5).
- `n_data::Int`: Number of stored data time steps after spin-up.
- `n_ic::Int`: Number of simulated initial conditions (independent runs).
- `n_spinup::Int`: Number of spin-up steps discarded before sampling.
- `t_step::Real`: Physical time step length.
- `initial_cond::Union{Nothing, Function}`: Optional generator for initial conditions; `if nothing`, random ICs are used.
- `id_key::String`: Additional identifier to disambiguate saved datasets with identical `trunc`, `n_data` and `n_ic`.
"""
struct SimPara
    trunc::Int
    n_data::Int
    n_ic::Int
    n_spinup::Int
    t_step::Real
    initial_cond::Union{Nothing, Function}
    id_key::String
end


"""
    SimPara(; trunc::Int,
             n_data::Int,
             n_ic::Int,
             n_spinup::Int = 9,
             t_step::Real = 1.0,
             initial_cond::Union{Nothing, Function} = nothing,
             id_key::String = "")

Convenience constructor for `SimPara`.

# Arguments
- `trunc::Int`: Spectral truncation of the barotropic model (e.g. 5 for T5).
- `n_data::Int`: Number of stored data time steps after spin-up.
- `n_ic::Int`: Number of simulated initial conditions (independent runs).
- `n_spinup::Int = 9`: Number of spin-up steps discarded before sampling (Default data sampling begins at t=10h).
- `t_step::Real = 1.0`: Physical time step length.
- `initial_cond::Union{Nothing, Function}`: Optional generator for initial conditions; `if nothing`, random ICs are used.
- `id_key::String = ""`: Additional identifier to disambiguate saved datasets with identical (`trunc`, `n_data`, `n_ic`).

# Returns
- `::SimPara`: Container for simulation parameters that define the simulation and data storage.

# Examples
```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200, id_key="test-suite")
```
"""
function SimPara(;trunc::Int, 
                n_data::Int, 
                n_ic::Int, 
                n_spinup::Int = 9, 
                t_step::Real = 1.0, 
                initial_cond::Union{Nothing, Function} = nothing, 
                id_key::String = "")

    return SimPara(trunc, n_data, n_ic, n_spinup, t_step, initial_cond, id_key)
end