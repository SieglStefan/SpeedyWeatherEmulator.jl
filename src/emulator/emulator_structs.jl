using Flux



"""
    NeuralNetwork

Container for the layer dimensions of a neural network.

# Fields
- `io_dim::Int`: Dimension of the input and output layer (e.g. number of spectral coefficients).
- `hidden_dim::Int`: Dimension of each hidden layer.
- `n_hidden::Int`: Number of hidden layers.
"""
struct NeuralNetwork
    io_dim::Int
    hidden_dim::Int
    n_hidden::Int
end


"""
    NeuralNetwork(; io_dim::Int=54, hidden_dim::Int=1024, n_hidden::Int=1)

Convenience constructor for `NeuralNetwork`.

# Arguments
- `io_dim::Int`: Dimension of the input and output layer (e.g. number of spectral coefficients).
- `hidden_dim::Int`: Dimension of each hidden layer.
- `n_hidden::Int`: Number of hidden layers.

# Returns
- `::NeuralNetwork`: Parameter container.
"""
function NeuralNetwork(;io_dim::Int=54, hidden_dim::Int=1024, n_hidden::Int=1)
    return NeuralNetwork(io_dim, hidden_dim, n_hidden)
end


"""
    Emulator{A<:AbstractVector{Float32}}

Container for a trained (or in-progress) neural network emulator.

# Fields
- `sim_para::SimPara`: Simulation parameters of the dataset used for training, validation and testing.
- `chain::Flux.Chain`: Neural network architecture and weights.
- `zscore_para::ZscorePara{A} `: Normalization parameters (mean/std of training set).
"""
struct Emulator{F, A<:AbstractVector{Float32}}
    sim_para::SimPara{F}
    chain::Flux.Chain
    zscore_para::ZscorePara{A} 
end


"""
    Emulator(sim_para::SimPara, nn::NeuralNetwork, zscore_para::ZscorePara)

Constructor for an `Emulator`. Builds a feed-forward network with ReLU activations
according to the given `NeuralNetwork` specs.

# Arguments
- `sim_para::SimPara`: Simulation parameters used for generating the training data.
- `nn::NeuralNetwork`: Parameters of the architecture (layer sizes).
- `zscore_para::ZscorePara`: Normalization parameters of the training data.

# Returns
- `::Emulator`: A wrapped Flux model with normalization metadata.
"""
function Emulator(sim_para::SimPara, nn::NeuralNetwork, zscore_para::ZscorePara)
    
    chain = Chain(                                      # creates the neural network structure
        Dense(nn.io_dim => nn.hidden_dim, relu),
        [Dense(nn.hidden_dim => nn.hidden_dim, relu) for _ in 1:(nn.n_hidden-1)]...,
        Dense(nn.hidden_dim => nn.io_dim)
    ) |> gpu                                            # "sends" the model to the gpu (if available)
    return Emulator(sim_para, chain, zscore_para)
end


"""
    (m::Emulator)(x::AbstractArray{Float32})

Convenience call overload. Apply the trained emulator to spectral coefficients at time t to predict coefficients at t + Δt.

# Arguments
- `x::AbstractArray{Float32}`: Spectral coefficients of vorticity at time t. First dimension must be `2 * n_coeff`.

# Returns
- `::Array{Float32}`: Emulator prediction for a single state at t + Δt (same size as input array).
"""
function (m::Emulator)(x::AbstractArray{Float32})
    x_norm = zscore(x, m.zscore_para)                   # using Z-score trafo to normalize the spectral coeff. at t
    y_norm = m.chain(x_norm |> gpu)                     # emulator calculates the normalized spectral coeff. at t + t_step
    return inv_zscore(y_norm, m.zscore_para) |> cpu     # using inverse Z-score trafo to get (normal) spectral coeff. at t + t_step
end


"""
    Losses{F}

Container for logging training and validation losses and training time.

# Fields
- `sim_para::SimPara{F}`: Simulation parameters of the dataset used.
- `train::Vector{Float32}`: Training loss per batch.
- `valid::Vector{Float32}`: Validation loss per batch.
- `bpe_train::Int`: Batches per epoch (training set).
- `bpe_valid::Int`: Batches per epoch (validation set).
- `training_time::Float32`: Time needed for training the model in seconds.
"""
struct Losses{F}
    sim_para::SimPara{F}
    train::Vector{Float32}
    valid::Vector{Float32}   
    bpe_train::Int     
    bpe_valid::Int
    training_time::Float32
end


"""
    Losses(sim_para::SimPara, bpe_train, bpe_valid)

Constructor for an empty `Losses` container.

# Arguments
- `sim_para::SimPara{F}`: Simulation parameters of the dataset used.
- `bpe_train::Int`: Batches per epoch (training set).
- `bpe_valid::Int`: Batches per epoch (validation set).

# Returns
- `::Losses{F}`: Initialized container with empty loss vectors and `training_time = 0.0`.
"""
Losses(sim_para::SimPara, bpe_train, bpe_valid) = Losses(sim_para, Float32[], Float32[], Int(bpe_train), Int(bpe_valid), Float32(0.0))

