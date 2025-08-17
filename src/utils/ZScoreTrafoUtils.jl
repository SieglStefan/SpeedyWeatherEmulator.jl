module ZScoreTrafoUtils

using Statistics
using ..BasicStructs

export zscore_trafo, inv_zscore_trafo


function zscore_trafo(data::Array{Float32, 3})
    μ = Float32.(vec(mean(data, dims=(2,3))))
    σ = Float32.(vec(std(data, dims=(2,3))))
    norm_stats = NormStats(μ, σ)

    data_norm = (data .- μ) ./ (σ .+ eps(Float32))

    return data_norm, norm_stats
end

function zscore_trafo(data::Array{Float32}, norm_stats)
    return (data .- norm_stats.μ) ./ (norm_stats.σ .+ eps(Float32))
end


function inv_zscore_trafo(data::Array{Float32}, norm_stats)
    return data .* (norm_stats.σ .+ eps(Float32)) .+ norm_stats.μ
end


end