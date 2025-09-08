using Statistics


"""
    ZscorePara{F<:AbstractVector{Float32}}

Container for the parameters of a Z-score transformation.

# Fields
- `μ::F`: Mean for each coefficient across samples.
- `σ::F`: Std for each coefficient across samples.

# Notes
- Typically computed from the **training set only** to avoid data leakage.
- For each coefficient indexed i: z_i = (x_i - μ_i) / (σ_i + eps)
"""
struct ZscorePara{F<:AbstractVector{Float32}}
    µ::F
    σ::F
end


"""
    zscore(x::AbstractArray{Float32}, stats::ZscorePara{<:AbstractVector{Float32}})

Apply a Z-score transformation to data `x` using the parameters in `stats`.

# Arguments
- `x::AbstractArray{Float32}`: Input data (rows = coefficients, columns = samples).
- `stats::ZscorePara{<:AbstractVector{Float32}}`: Parameters with mean μ and std σ.

# Returns
- `::typeof(x)`: Z-score normalized data.

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
function zscore(x::AbstractArray{Float32}, stats::ZscorePara{<:AbstractVector{Float32}})
    return (x .- stats.µ) ./ (stats.σ .+ eps(Float32))
end


"""
    inv_zscore(x::AbstractArray{Float32}, stats::ZscorePara{<:AbstractVector{Float32}})

Inverse Z-score transformation (restore original scale).

# Arguments
- `x::AbstractArray{Float32}`: Z-score normalized data.
- `stats::ZscorePara{<:AbstractVector{Float32}}`: Parameters with mean μ and std σ.

# Returns
- `::typeof(x)`: Data rescaled back to the original distribution.

# Examples
```julia
stats = ZscorePara([0.0f0, 1.0f0], [1.0f0, 2.0f0])
z = Float32[0 -0.5; 1 0.5]
x = inv_zscore(z, stats)
```
"""
function inv_zscore(x::AbstractArray{Float32}, stats::ZscorePara{<:AbstractVector{Float32}})
    return x .* stats.σ .+ stats.µ
end

