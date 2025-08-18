module ZScoreUtils

using Statistics
using ..BasicStructs

export NormStats, zscore, inv_zscore


struct NormStats
    µ::Vector{Float32}
    σ::Vector{Float32}
end

"Z-Score-Transformation"
function zscore(x, stats::NormStats)
    return (x .- stats.µ) ./ (stats.σ .+ eps(Float32))
end

"Inverse Z-Score"
function inv_zscore(x, stats::NormStats)
    return x .* stats.σ .+ stats.µ
end

end