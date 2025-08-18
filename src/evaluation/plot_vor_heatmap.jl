module PlotVorHeatmap

using SpeedyWeather
using CairoMakie
using ..BasicStructs


export plot_vor_heatmap


"""
    plot_vor_heatmap(vec::Vector{Float32}, trunc::Int; title::String="") 

Plots the heatmap of the spectral coeff. stored in `vec` with title `title`.

# Arguments
- `vec::Vector{Float32}`:   Vector containing the spectral coeff.
- `trunc::Int`:             Truncation used to calculate the LTM needed.
- `title::String = ""`:     Title of the heatmap plot.

# Returns
- Plot of the heatmap.
"""
function plot_vor_heatmap(vec::Vector{Float32}, trunc::Int; title::String="")
    
    vor_ltm = vec_to_ltm(vec, trunc)        # calculating the LTM
    vor_grid = transform(vor_ltm)           # transforming the LTM into a plotable grid

    return CairoMakie.heatmap(vor_grid, title=title)
end


"""
    vec_to_ltm(vec::Vector{Float32}, trunc::Int)

Calculates a LTM corresponding to the spectral coeff. vector `vec`.

e.g.: trunc=5 -> n_coeff = 54 -> L = Mat(C, 7x6).

# Arguments
- `vec::Vector{Float32}`:                   Spectral coeff. vector which is used to calculate the LTM.
- `trunc`:                                  Truncation of the spectral coeff.

# Returns
- `L::LowerTriangularMatrix{ComplexF32}`:   Contains the complex spectral coeff.
"""
function vec_to_ltm(vec::Vector{Float32}, trunc::Int)
    N = trunc+2
    M = trunc+1

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
                L[i,j] = vec[counter]
                counter = counter + 1
            end
        end
    end

    return L
end



end