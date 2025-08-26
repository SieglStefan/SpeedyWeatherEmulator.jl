using Statistics


"""
    ZscorePara(μ::Vector{Float32}, σ::Vector{Float32})

Container for the parameters of a Z-score transformation.

# Fields
- `μ::Vector{Float32}`: Mean for each coefficient across samples.
- `σ::Vector{Float32}`: Std for each coefficient across samples

# Notes
- Typically computed from the **training set only** to avoid data leakage.
- For each coefficient indexed i: z_i = (x_i - μ_i) / σ_i
"""
struct ZscorePara
    µ::Vector{Float32}
    σ::Vector{Float32}
end


"""
    zscore(x, stats::ZscorePara)

Apply a Z-score transformation to data `x` using the parameters in `stats`.

# Arguments
- `x::AbstractArray`: Input data (rows = coefficients, columns = samples).
- `stats::ZscorePara`: Parameters with mean μ and std σ.

# Returns
- `::Array{Float32}`: Z-score normalized data.

# Notes
- Each coefficient is transformed independently.
- A small `eps(Float32)` is added to σ to avoid division by zero.

# Examples
```julia
stats = ZscorePara([0.0f0, 1.0f0], [1.0f0, 2.0f0])
x = Float32[0 2; 1 3]
z = zscore(x, stats)
```
"""
function zscore(x::Array{Float32}, stats::ZscorePara)
    return (x .- stats.µ) ./ (stats.σ .+ eps(Float32))
end


"""
    inv_zscore(x::Array{Float32}, stats::ZscorePara)

Inverse Z-score transformation (restore original scale).

# Arguments
- `x::Array{Float32}`: Z-score normalized data.
- `stats::ZscorePara`: Parameters with mean μ and std σ.

# Returns
- `::Array{Float32}`: Data rescaled back to the original distribution.

# Examples
```julia
stats = ZscorePara([0.0f0, 1.0f0], [1.0f0, 2.0f0])
z = Float32[0 -0.5; 1 0.5]
x = inv_zscore(z, stats)
```
"""
function inv_zscore(x::Array{Float32}, stats::ZscorePara)
    return x .* stats.σ .+ stats.µ
end

