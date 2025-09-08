
"""
    calc_n_coeff(; trunc::Int)

Calculate the number of complex spectral coefficients for a given spectral truncation.

# Arguments
- `trunc::Int`: Spectral truncation of the barotropic model (e.g. 5 for T5).

# Returns
- `n_coeff::Int`: Number of complex spectral coefficients (without splitting into real/imag parts).
"""
function calc_n_coeff(;trunc::Int)
    n_coeff = 0

    for i in 1:trunc+2
        n_coeff = n_coeff + i
    end

    n_coeff = n_coeff - 1

    return n_coeff
end