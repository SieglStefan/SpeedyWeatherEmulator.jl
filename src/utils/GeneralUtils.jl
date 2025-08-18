module GeneralUtils

using CUDA, Pkg

export gpu_usage, get_vorticity!


function gpu_usage()
    if isnothing(Base.find_package("CUDA"))
        println("CUDA.jl not found — installing...")
        Pkg.add("CUDA")
    end

    if isnothing(Base.find_package("cuDNN"))
        println("cuDNN.jl not found — installing...")
        Pkg.add("cuDNN")
    end

    if CUDA.has_cuda()
        device = gpu
        @info "GPU found!"
    else
        device = cpu
        @info "GPU NOT found!"
    end
    
    return device

end

function get_vorticity!(vor::Array{Float32,3}, sim, step::Int, ic::Int)
    vorticity_vec = vec(sim.prognostic_variables.vor[:,1,1])
    n_vars = length(vorticity_vec)
    
    vor[1:n_vars, step, ic] .= Float32.(real.(vorticity_vec))
    vor[n_vars+1:2*n_vars, step, ic] .= Float32.(imag.(vorticity_vec))
end

function get_vorticity!(vor::Array{Float32,2}, sim, step::Int)
    vorticity_vec = vec(sim.prognostic_variables.vor[:,1,1])
    n_vars = length(vorticity_vec)
    
    vor[1:n_vars, step] .= Float32.(real.(vorticity_vec))
    vor[n_vars+1:2*n_vars, step] .= Float32.(imag.(vorticity_vec))
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