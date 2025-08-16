module DataUtils   

using Statistics

export get_vorticity!, zscore_trafo, inv_zscore_trafo



"""
    get_vorticity!(A, sim, step, ic)

Speichert die reellen und imaginären Teile der Spektral-Koeffizienten der Vorticity
von `sim` in den Slice `A[:, step, ic]`.

- `A` ist ein Array der Größe (2N, n_steps, n_ic)
- `sim` ist eine SpeedyWeather-Simulation
- `step` ist der Zeitschritt (Int)
- `ic` ist die Initial Condition (Int)
"""
function get_vorticity!(vor::Array{Float32, 3}, sim, step::Int, ic::Int)
    vorticity_vec = vec(sim.prognostic_variables.vor[:,1,1])
    n_vars = length(vorticity_vec)
    
    vor[1:n_vars, step, ic] .= Float32.(real.(vorticity_vec))
    vor[n_vars+1:2*n_vars, step, ic] .= Float32.(imag.(vorticity_vec))
end

function get_vorticity!(vor::Array{Float32, 1}, sim)
    vorticity_vec = vec(sim.prognostic_variables.vor[:,1,1])
    n_vars = length(vorticity_vec)
    
    vor[1:n_vars] .= Float32.(real.(vorticity_vec))
    vor[n_vars+1:2*n_vars] .= Float32.(imag.(vorticity_vec))
end


function zscore_trafo(data::Array{Float32, 3})
    μ = vec(mean(data, dims=(2,3)))
    σ = vec(std(data, dims=(2,3)))

    data_norm = (data .- μ) ./ (σ .+ eps(Float32))

    return data_norm, μ, σ
end

function inv_zscore_trafo(data, μ, σ)
    return data .* (σ .+ eps(32)) .+ μ
end




end # module