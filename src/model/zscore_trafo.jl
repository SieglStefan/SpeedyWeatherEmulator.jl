module ZscoreTrafo

using Statistics


export ZscorePara, zscore, inv_zscore


"""
   ZscorePara(µ::Vector{Float32}, σ::Vector{Float32})

Container of the parameters of the Z-score transformation.

# Fields
- `µ::Vector{Float32}`: Mean of the data set.
- `σ::Vector{Float32}`: Standard deviation of the data set. 
"""
struct ZscorePara
    µ::Vector{Float32}
    σ::Vector{Float32}
end


"""
    zscore(x, stats::ZscorePara)  

Z-score transformation of the data `x` with the parameters of `stats`.

# Arguments
- `x`: Data set which is Z-score transformed
- `stats::ZscorePara`: Parameters of the Z-score transformation

# Returns
- Z-score transformed data
"""
function zscore(x, stats::ZscorePara)
    return (x .- stats.µ) ./ (stats.σ .+ eps(Float32))
end


"""
    inv-zscore(x, stats::ZscorePara)  

Inverse Z-score transformation of the data `x` with the parameters of `stats`.

# Arguments
- `x`: Data set which is inverse Z-score transformed
- `stats::ZscorePara`: Parameters of the Z-score transformation

# Returns
- inverse Z-score transformed data
"""
function inv_zscore(x, stats::ZscorePara)
    return x .* stats.σ .+ stats.µ
end

end