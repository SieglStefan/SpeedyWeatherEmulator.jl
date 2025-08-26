
"""
    is_coeff_zero(i::Int64, data::Array{Float32, 3})  

Checks if a specific spectral coefficient - indexed `i` - is always zero in the simulation data `sim_data`.

# Arguments
- `i::Int64`: Index of spectral coefficient.
- `data::Array{Float32, 3}`: Spectral coefficients which are checked.

# Returns
- `is_zero::Bool`: Returns if they are all zero.
"""
function is_coeff_zero(i::Int64, data::Array{Float32, 3})
    is_zero = all(data[i, :, :] .== 0f0)

    return is_zero
end


"""
    calc_n_coeff(; trunc::Int64)

Calculate the number of complex spectral coefficients for a given spectral truncation.

# Arguments
- `trunc::Int64`: Spectral truncation of the barotropic model (e.g. 5 for T5).

# Returns
- `n_coeff::Int64`: Number of complex spectral coefficients (without splitting into real/imag parts).
"""
function calc_n_coeff(;trunc::Int64)
    n_coeff = 0

    for i in 1:trunc+2
        n_coeff = n_coeff + i
    end

    n_coeff = n_coeff - 1

    return n_coeff
end