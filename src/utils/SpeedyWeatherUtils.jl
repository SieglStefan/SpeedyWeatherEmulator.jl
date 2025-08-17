module SpeedyWeatherUtils   

using Statistics
using ..BasicStructs

export get_vorticity!


function get_vorticity!(vor::Array{Float32, 3}, sim, step::Int, ic::Int)
    vorticity_vec = vec(sim.prognostic_variables.vor[:,1,1])
    n_vars = length(vorticity_vec)
    
    vor[1:n_vars, step, ic] .= Float32.(real.(vorticity_vec))
    vor[n_vars+1:2*n_vars, step, ic] .= Float32.(imag.(vorticity_vec))
end

function get_vorticity!(sim)
    vorticity_vec = vec(sim.prognostic_variables.vor[:,1,1])
    n_vars = length(vorticity_vec)
    
    vor = zeros(Float32, 2*n_vars)

    vor[1:n_vars] .= Float32.(real.(vorticity_vec))
    vor[n_vars+1:2*n_vars] .= Float32.(imag.(vorticity_vec))

    return vor
end


end