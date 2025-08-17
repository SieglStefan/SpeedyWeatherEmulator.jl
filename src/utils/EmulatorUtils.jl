module EmulatorUtils

using Flux, JLD2
using ..BasicStructs
include(joinpath(@__DIR__, "..", "utils", "ZScoreTrafoUtils.jl"))
using .ZScoreTrafoUtils


export TrainingData, Losses, NeuralNetwork, load_neuralnetwork

struct DataPairs
    x_train::Matrix{Float32}
    y_train::Matrix{Float32}
    x_valid::Matrix{Float32}
    y_valid::Matrix{Float32}
end

struct TrainingData
    sim_para::SimPara
    data_pairs::DataPairs
    norm_stats::NormStats
end

function TrainingData(sim_para::SimPara; split::Float64=0.8, norm_stats::Union{Nothing,NormStats}=nothing)
    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic

    dir = joinpath(@__DIR__, "..", "..", "data", "sim_data")
    filename = "sim_data_T$(trunc)_nsteps$(n_steps)_IC$(n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    sim_data = load(filepath, "sim_data")
    data = Float32.(sim_data.data)

    if sim_para == sim_data.sim_para
        @info "Loaded Data is good"
    end

    if isnothing(norm_stats)
        data_norm, norm_stats = zscore_trafo(data)
    else
        data_norm = zscore_trafo(data, norm_stats)
    end

    x = data_norm[:, 1:end-1, :]
    y = data_norm[:, 2:end,   :]

    n_data = (n_steps - 1) * n_ic
    n_train = Integer(split * n_data)

    # Creating training pairs x_i and x_{i+1}
    x_train = reshape(x, size(data_norm,1), :)[:,1:n_train]
    y_train = reshape(y, size(data_norm,1), :)[:,1:n_train]

    x_valid = reshape(x, size(data_norm,1), :)[:,n_train+1:n_data]
    y_valid = reshape(y, size(data_norm,1), :)[:,n_train+1:n_data]

    @info "Loaded data with a total of $n_data data points"

    return TrainingData(sim_para,
                        DataPairs(x_train, y_train, x_valid, y_valid),
                        norm_stats)

end

#function save_training_data(;sim_para::SimPara, norm_stats::NormStats)
    #dir = joinpath(@__DIR__, "..", "..", "data", "training_data")

    #filename = "training_data_T$(TRUNC)_nsteps$(n_steps)_IC$(n_ic).jld2"
    #filepath = normpath(joinpath(dir, filename))

    #jldsave(filepath; data, Norm)
    #@info "Normed training data saved at: $filepath"
#end



struct Losses
    train::Vector{Float32}      # loss per batch
    valid::Vector{Float32}      # loss per batch
    test::Union{Nothing, Vector{Float32}}
    bpe_train::Int64            # batch per epoch in the training set
    bpe_valid::Int64
end

Losses(bpe_train::Int, bpe_valid::Int) = Losses(Float32[], Float32[], nothing, bpe_train, bpe_valid)



struct NeuralNetwork
    chain::Flux.Chain
end

function NeuralNetwork(;input_dim::Int, hidden_dim::Int, output_dim::Int)
    chain = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim)
    ) |> gpu

    return NeuralNetwork(chain)
end

function (m::NeuralNetwork)(x)
    return m.chain(x)
end


function load_neuralnetwork(sim_para::SimPara)

    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "model_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))
    file = load(filepath)

    model = file["model"]
    norm_stats = file["norm_stats"]

    return model, norm_stats
end



end