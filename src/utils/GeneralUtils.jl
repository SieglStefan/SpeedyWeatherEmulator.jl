module GeneralUtils

using CUDA, Pkg

export gpu_usage


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

end