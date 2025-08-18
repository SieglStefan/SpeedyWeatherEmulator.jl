module ModelStructs

using JLD2
using Flux
using ..BasicStructs
using ..ZScoreUtils


export NeuralNetwork, TrainedModel, Losses



struct NeuralNetwork
    io_dim::Int
    hidden_dim::Int
end

struct TrainedModel
    chain::Flux.Chain
    norm_stats::NormStats
    sim_para::SimPara
end

function NeuralNetwork(;io_dim::Int=54, hidden_dim::Int=128)
    return NeuralNetwork(io_dim, hidden_dim)
end

function TrainedModel(nn::NeuralNetwork, norm_stats::NormStats, sim_para::SimPara)
    chain = Chain(
        Dense(nn.io_dim => nn.hidden_dim, relu),
        Dense(nn.hidden_dim => nn.hidden_dim, relu),
        Dense(nn.hidden_dim => nn.io_dim)
    ) |> gpu
    return TrainedModel(chain, norm_stats, sim_para)
end

function (m::TrainedModel)(x)
    x_norm = zscore(x, m.norm_stats)
    y_norm = m.chain(x_norm |> gpu)
    return inv_zscore(y_norm, m.norm_stats)
end


struct Losses
    train::Vector{Float32}      # loss per batch
    valid::Vector{Float32}      # loss per batch
    test::Union{Nothing, Vector{Float32}}
    bpe_train::Int64            # batch per epoch in the training set
    bpe_valid::Int64
    sim_para::SimPara
end

Losses(bpe_train::Int, bpe_valid::Int, sim_para::SimPara) = Losses(Float32[], Float32[], nothing, bpe_train, bpe_valid, sim_para)


end


