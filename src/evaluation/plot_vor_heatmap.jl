module PlotVorHeatmap

using SpeedyWeather
using CairoMakie
using ..BasicStructs

export plot_vor_heatmap

function plot_vor_heatmap(vec::Vector{Float32}, trunc::Int; title::String="")
    
    vor_ltm = vec_to_ltm(vec, trunc)  
    vor_grid = transform(vor_ltm)

    return CairoMakie.heatmap(vor_grid, title=title)
end


function vec_to_ltm(vec::Vector{Float32}, trunc::Int)
    N = trunc+2
    M = trunc+1

    L = rand(LowerTriangularMatrix{ComplexF32}, N, M)
    @info "Creating a $N x $M LowerTriangularMatrix"

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