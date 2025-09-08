
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

    # Summing up the lines of a 7x7 LowerTriangularMatrix
    for i in 1:trunc+2
        n_coeff = n_coeff + i
    end

    # Correcting to a 7x6 LTM (e.g. trunc=5 corresponds to  7x6 LTM)
    n_coeff = n_coeff - 1

    return n_coeff
end