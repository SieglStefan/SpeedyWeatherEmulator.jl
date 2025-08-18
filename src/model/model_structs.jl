module ModelStructs

using JLD2
using Flux
using ..BasicStructs
using ..ZscoreTrafo


export NeuralNetwork, TrainedModel, Losses


"""
   NeuralNetwork(io_dim::Int, hidden_dim::Int)

Container of the layer parameters of a neural network, see constructor for details.
"""
struct NeuralNetwork
    io_dim::Int
    hidden_dim::Int
end


"""
   NeuralNetwork(;io_dim::Int=54, hidden_dim::Int=128)

Constructor for the parameters of a neural network.

# Arguments
- `io_dim::Int=54`: Dimension of the in-/ouput layer.
- `hidden_dim::Int=128`: Dimension of the hidden layer.
"""
function NeuralNetwork(;io_dim::Int=54, hidden_dim::Int=128)
    return NeuralNetwork(io_dim, hidden_dim)
end


"""
   TrainedModel(chain::Flux.Chain, zscore_para::ZscorePara, sim_para::SimPara)

Trained (or in the progressing of training) neural network.

# Fields
- `chain::Flux.Chain`: Chain (neural network architecture) of the trained model.
- `zscore_para::ZscorePara`: See constructor for details.
- `sim_para::SimPara`: See constructor for details.
"""
struct TrainedModel
    chain::Flux.Chain
    zscore_para::ZscorePara
    sim_para::SimPara
end


"""
   TrainedModel(nn::NeuralNetwork, norm_stats::NormStats, sim_para::SimPara)

Constructor for a trained model, which initializes the chain (neural network)

# Fields
- `nn::NeuralNetwork`: Parameters of the neural network
- `zscore_para::ZscorePara`: Zscore parameters of the used training set.
    (The spectral coefficents have different orders of magnitude, therefore a Z-score transformation is used to bring them on 
    a common level)
- `sim_para::SimPara`: Simulation parameters of the training-, validation- and testset.
"""
function TrainedModel(nn::NeuralNetwork, zscore_para::ZscorePara, sim_para::SimPara)
    chain = Chain(                                          # creates the neural network structure
        Dense(nn.io_dim => nn.hidden_dim, relu),
        Dense(nn.hidden_dim => nn.hidden_dim, relu),
        Dense(nn.hidden_dim => nn.io_dim)
    ) |> gpu                                                # "sends" the model to the gpu (if available)
    return TrainedModel(chain, zscore_para, sim_para)
end


"""
    (m::TrainedModel)(x)

Convenience function to directly use a trained model the the spectral coefficients of the vorticity.

# Arguments
- `x`: Spectral coefficients of the vorticity at t

# Returns
- `x`: Calculated (from the emulator) spectral coefficients of the vorticity at t + t_step
"""
function (m::TrainedModel)(x)
    x_norm = zscore(x, m.zscore_para)               # using Z-score trafo to normalize the spectral coeff. at t
    y_norm = m.chain(x_norm |> gpu)                 # emulator calculates the normalized spectral coeff. at t + t_step
    return inv_zscore(y_norm, m.zscore_para)        # using inverse Z-score trafo to get (normal) spectral coeff. at t + t_step
end


"""
   Losses(  train::Vector{Float32},
            valid::Vector{Float32},
            test::Vector{Float32},
            bpe_train::Int64,
            bpe_valid::Int64,
            bpe_test::Int64,           
            sim_para::SimPara)

Container of the different losses during the training process used for plotting.

# Fields
- `train::Vector{Float32}`: Stores Training Loss per batch.
- `valid::Vector{Float32}`: Stores Validation Loss per batch.
- `test::Vector{Float32}`: Stores Test Loss per batch.
- `bpe_train::Int64`: Batches Per Epoch for the training set.
- `bpe_valid::Vector{Float32}`: Batches Per Epoch for the validation set.
- `bpe_test::Vector{Float32}`: Batches Per Epoch for the test set.
- `sim_para::SimPara`: Simulation parameters of the used simulation data.
"""
struct Losses
    train::Vector{Float32}  
    valid::Vector{Float32}    
    test::Vector{Float32}
    bpe_train::Int64     
    bpe_valid::Int64
    bpe_test::Int64
    sim_para::SimPara
end


"""
   Losses(  bpe_train::Int,
            bpe_valid::Int,
            sim_para::SimPara)

Constructor for the Losses container, see struct definition for details.
"""
Losses(bpe_train::Int, bpe_valid::Int, sim_para::SimPara) = Losses(Float32[], Float32[], Float32[], bpe_train, bpe_valid, 0, sim_para)



end


