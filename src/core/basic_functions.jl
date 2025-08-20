module BasicFunctions

"""
    is_coeff_zero(i::Int, sim_data::SimData)  

Checks if a specific spectral coefficient - numbered `i` - is always zero in the simulation data `sim_data`.

# Arguments
- `i::Int64`: Number of spectral coefficient.
- `sim_data::SimData`: Simulation data which is checked.

# Returns
- `is_zero::Bool`: Returns if they are all zero.
"""
function is_coeff_zero(i::Int64, sim_data::SimData)
    data = sim_data.data
    is_zero = all(data[i, :, :] .== 0f0)

    return is_zero
end


"""
    calc_n_coeff(;trunc::Int64)

Calculates the number of spectral coefficents of a specific truncation `trunc`.

# Arguments
- `trunc::Int64`: Number of spectral coefficient.

# Returns
- `n_coeff::Int64`: Number of spectral coefficients.
"""
function calc_n_coeff(;trunc::Int64)
    n_coeff = 0

    for i in 1:trunc+2
        n_coeff = n_coeff + i
    end

    return n_coeff-1
end