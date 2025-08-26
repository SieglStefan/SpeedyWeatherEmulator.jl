using CairoMakie
using SpeedyWeather


"""
    plot_heatmap(vec::Vector{Float32}; trunc::Int64, title::String="")

Plot a heatmap of a vorticity field reconstructed from a spectral coefficient vector.

# Description
- Converts spectral coefficients in vector form into a `LowerTriangularMatrix`.
- Transforms this into a grid suitable for plotting.
- Displays the grid as a heatmap using CairoMakie.

# Arguments
- `vec::Vector{Float32}`: Spectral coefficient vector (real/imag stacked).
- `trunc::Int64`: Spectral truncation of the model (e.g. 5 for T5).
- `title::String=""`: Optional title for the plot.

# Returns
- `::::CairoMakie.Plot`: Heatmap figure object.

# Examples
```julia
vec = rand(Float32, 54)   # random spectral coeffs for trunc=5
fig = plot_heatmap(vec; trunc=5, title="Vorticity field")
display(fig)
```
"""
function plot_heatmap(vec::Vector{Float32}; trunc::Int64, title::String="")
    
    vor_ltm = vec_to_ltm(vec, trunc)        # calculating the LTM
    vor_grid = transform(vor_ltm)           # transforming the LTM into a plotable grid

    return CairoMakie.heatmap(vor_grid, title=title)
end


"""
    vec_to_ltm(vec::Vector{Float32}, trunc::Int64)

Convert a spectral coefficient vector into a `LowerTriangularMatrix` representation.

# Description
- Interprets the coefficient vector as complex spectral coefficients.
- Places them into a triangular matrix layout consistent with SpeedyWeather.jl.

# Arguments
- `vec::Vector{Float32}`: Vector of spectral coefficients.
- `trunc::Int64`: Spectral truncation of the model (e.g. 5 for T5).

# Returns
- `L::LowerTriangularMatrix{ComplexF32}`: Complex spectral coefficient matrix.

# Notes
- For `trunc=5`, produces an `N=7 x M=6` `LowerTriangularMatrix` with 27 entries.
- `vec` is expected to be structured as [Re(c_1), …, Re(c_n_coeff), Im(c_1), …, Im(c_n_coeff)].

# Examples
```julia
n = calc_n_coeff(trunc=5)
vec = rand(Float32, 2*n)
L = vec_to_ltm(vec, 5)
```
"""
function vec_to_ltm(vec::Vector{Float32}, trunc::Int64)
    N = trunc+2
    M = trunc+1
    n_coeff = calc_n_coeff(trunc=trunc)

    # Defining the order of the LTM: Mat(C, NxM)
    L = rand(LowerTriangularMatrix{ComplexF32}, N, M)
    @info "Creating a $N x $M LowerTriangularMatrix"

    # Filling the LTM up
    counter = 1                       

    for j in 1:M
        for i in j:N
            if j == N && i == N
                continue
            else
                real_part = vec[counter]
                imag_part = vec[n_coeff+counter]
                L[i,j] = ComplexF32(real_part, imag_part)
                counter = counter + 1
            end
        end
    end

    return L
end