module DataStructs

using JLD2, Flux

export SimPara, DataPairs, NormStats, TrainingData, Losses, MyModel


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


function TrainingData(; trunc::Integer, n_steps::Integer, n_ic::Integer, split::Float64=0.8)
    data = load("experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(n_steps)_IC$(n_ic).jld2")

    data_norm = Float32.(data["data_norm"])
    µ = Float32.(data["μ"])
    σ = Float32.(data["σ"])

    x = data_norm[:, 1:end-1, :]
    y = data_norm[:, 2:end,   :]

    n_data = (n_steps - 1) * n_ic
    n_train = Integer(split * n_data)

    # Creating training pairs x_i and x_{i+1}
    x_train = reshape(x, size(data_norm,1), :)[:,1:n_train]
    y_train = reshape(y, size(data_norm,1), :)[:,1:n_train]

    x_valid = reshape(x, size(data_norm,1), :)[:,n_train+1:n_data]
    y_valid = reshape(y, size(data_norm,1), :)[:,n_train+1:n_data]

    @info "Loaded data with ", n_data ," data points"

    return TrainingData(SimPara(trunc, n_steps, n_ic),
                        DataPairs(x_train, y_train, x_valid, y_valid),
                        NormStats(µ, σ))

end


struct Losses
    train::Vector{Float32}
    valid::Vector{Float32}
    test::Union{Nothing, Vector{Float32}}
end


Losses() = Losses(Float32[], Float32[], nothing)


struct MyModel
    chain::Flux.Chain
end

function MyModel(input_dim::Int, hidden_dim::Int, output_dim::Int)
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


end