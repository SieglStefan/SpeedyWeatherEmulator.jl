module DataStructs

using JLD2, Flux, Pkg, CUDA, cuDNN

export SimPara, DataPairs, NormStats, TrainingData, Losses, MyModel, gpu_usage


struct SimPara
    trunc::Int64
    n_steps::Int64
    n_ic::Int64
end

struct DataPairs
    x_train::Matrix{Float32}
    y_train::Matrix{Float32}
    x_valid::Matrix{Float32}
    y_valid::Matrix{Float32}
end

struct NormStats
    µ::Vector{Float32}
    σ::Vector{Float32}
end

struct TrainingData
    sim_para::SimPara
    data_pairs::DataPairs
    norm_stats::NormStats
end




function save_training_data(n_steps::Integer, n_ic::Integer, data::Array{Float32, 3}, μ::Vector{Float32}, σ::Vector{Float32})
    dir = joinpath(@__DIR__, "..", "..", "data", "training_data")

    filename = "training_data_T$(TRUNC)_nsteps$(n_steps)_IC$(n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; data, μ, σ)
    @info "Normed training data saved at: $filepath"
end



function TrainingData(; trunc::Integer, n_steps::Integer, n_ic::Integer, split::Float64=0.8)
    dir = joinpath(@__DIR__, "..", "..", "data", "training_data")
    
    filename = "training_data_T$(trunc)_nsteps$(n_steps)_IC$(n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))
    file = load(filepath)

    data = Float32.(file["vor_norm"])
    norm_stats = (file["norm_stats"])

    x = data[:, 1:end-1, :]
    y = data[:, 2:end,   :]

    n_data = (n_steps - 1) * n_ic
    n_train = Integer(split * n_data)

    # Creating training pairs x_i and x_{i+1}
    x_train = reshape(x, size(data,1), :)[:,1:n_train]
    y_train = reshape(y, size(data,1), :)[:,1:n_train]

    x_valid = reshape(x, size(data,1), :)[:,n_train+1:n_data]
    y_valid = reshape(y, size(data,1), :)[:,n_train+1:n_data]

    @info "Loaded data with a total of $n_data data points"

    return TrainingData(SimPara(trunc, n_steps, n_ic),
                        DataPairs(x_train, y_train, x_valid, y_valid),
                        norm_stats)

end


struct Losses
    train::Vector{Float32}      # loss per batch
    valid::Vector{Float32}      # loss per batch
    test::Union{Nothing, Vector{Float32}}
    bpe_train::Int64            # batch per epoch in the training set
    bpe_valid::Int64
end

Losses(bpe_train::Int, bpe_valid::Int) = Losses(Float32[], Float32[], nothing, bpe_train, bpe_valid)



struct MyModel
    chain::Flux.Chain
end

function MyModel(;input_dim::Int, hidden_dim::Int, output_dim::Int)
    chain = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim)
    ) |> gpu

    return MyModel(chain)
end

function (m::MyModel)(x)
    return m.chain(x)
end

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