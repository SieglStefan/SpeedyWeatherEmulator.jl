using Flux



"""
    NeuralNetwork

Container for the layer dimensions of a neural network.

# Fields
- `io_dim::Int64`: Dimension of the input and output layer (e.g. number of spectral coefficients).
- `hidden_dim::Int64`: Dimension of each hidden layer.
- `n_hidden::Int64`: Number of hidden layers.
"""
struct NeuralNetwork
    io_dim::Int64
    hidden_dim::Int64
    n_hidden::Int64
end


"""
    NeuralNetwork(; io_dim::Int64=54, hidden_dim::Int64=128, n_hidden::Int64=1)

Convenience constructor for `NeuralNetwork`.

# Arguments
- `io_dim::Int64`: Dimension of the input and output layer (e.g. number of spectral coefficients).
- `hidden_dim::Int64`: Dimension of each hidden layer.
- `n_hidden::Int64`: Number of hidden layers.

# Returns
- `::NeuralNetwork`: Parameter container.
"""
function NeuralNetwork(;io_dim::Int64=54, hidden_dim::Int64=128, n_hidden::Int64=1)
    return NeuralNetwork(io_dim, hidden_dim, n_hidden)
end


"""
    Emulator

Container for a trained (or in-progress) neural network emulator.

# Fields
- `sim_para::SimPara`: Simulation parameters of the dataset used for training, validation and testing.
- `chain::Flux.Chain`: Neural network architecture and weights.
- `zscore_para::ZscorePara`: Normalization parameters (mean/std of training set).
"""
struct Emulator
    sim_para::SimPara
    chain::Flux.Chain
    zscore_para::ZscorePara  
end


"""
    Emulator(nn::NeuralNetwork, zscore_para::ZscorePara, sim_para::SimPara)

Constructor for an `Emulator`. Builds a feed-forward network with ReLU activations
according to the given `NeuralNetwork` specs.

# Arguments
- `nn::NeuralNetwork`: Parameters of the architecture (layer sizes).
- `zscore_para::ZscorePara`: Normalization parameters of the training data.
- `sim_para::SimPara`: Simulation parameters used for generating the training data.

# Returns
- `::Emulator`: A wrapped Flux model with normalization metadata.
"""
function Emulator(nn::NeuralNetwork, zscore_para::ZscorePara, sim_para::SimPara)
    
    chain = Chain(                                      # creates the neural network structure
        Dense(nn.io_dim => nn.hidden_dim, relu),
        [Dense(nn.hidden_dim => nn.hidden_dim, relu) for _ in 1:(nn.n_hidden-1)]...,
        Dense(nn.hidden_dim => nn.io_dim)
    ) |> gpu                                            # "sends" the model to the gpu (if available)
    return Emulator(sim_para, chain, zscore_para)
end


"""
    (m::Emulator)(x::Union{Vector{Float32}, Matrix{Float32}})

Convenience call overload. Apply the trained emulator to spectral coefficients
at time t to predict coefficients at t + Δt.

# Arguments
- `x::Vector{Float32}`: Spectral coefficients of vorticity at time t (size = 2 * n_coeff).
- `x::Matrix{Float32}`: Multiple states, each column a spectral coefficients vector at time t (size = (2 * n_coeff, N)).

# Returns
- `::Vector{Float32}`: Emulator prediction for a single state at t + Δt (same size as input vector).
- `::Matrix{Float32}`: Emulator predictions for multiple states, one prediction per column (same shape as input matrix).
"""
function (m::Emulator)(x::Union{Vector{Float32}, Matrix{Float32}})
    x_norm = zscore(x, m.zscore_para)                   # using Z-score trafo to normalize the spectral coeff. at t
    y_norm = m.chain(x_norm |> gpu)                     # emulator calculates the normalized spectral coeff. at t + t_step
    return inv_zscore(y_norm, m.zscore_para) |> cpu     # using inverse Z-score trafo to get (normal) spectral coeff. at t + t_step
end


"""
    Losses

Container for logging training, validation, and test losses.

# Fields
- `sim_para::SimPara`: Simulation parameters of the dataset used.
- `train::Vector{Float32}`: Training loss per batch.
- `valid::Vector{Float32}`: Validation loss per batch.
- `test::Vector{Float32}`: Test loss per batch.
- `bpe_train::Int64`: Batches per epoch (training set).
- `bpe_valid::Int64`: Batches per epoch (validation set).
- `bpe_test::Int64`: Batches per epoch (test set).
"""
struct Losses
    sim_para::SimPara
    train::Vector{Float32}  
    valid::Vector{Float32}    
    test::Vector{Float32}
    bpe_train::Int64     
    bpe_valid::Int64
    bpe_test::Int64
end


"""
    Losses(sim_para::SimPara, bpe_train::Int64, bpe_valid::Int64)

Constructor for an empty `Losses` container.

# Arguments
- `sim_para::SimPara`: Simulation parameters of the dataset used.
- `bpe_train::Int64`: Batches per epoch (training set).
- `bpe_valid::Int64`: Batches per epoch (validation set).

# Returns
- `::Losses`: Initialized container with empty loss vectors and `bpe_test = 0`.
"""
Losses(sim_para::SimPara, bpe_train::Int64, bpe_valid::Int64) = Losses(sim_para, Float32[], Float32[], Float32[], bpe_train, bpe_valid, 0)

