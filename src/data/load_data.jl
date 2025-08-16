using JLD2

struct TrainingData
    trunc::Int64
    nsteps::Int64
    IC::Int64
    X_train::Matrix{Float32}
    Y_train::Matrix{Float32}
    X_valid::Matrix{Float32}
    Y_valid::Matrix{Float32}
    μ::Vector{Float32}
    σ::Vector{Float32}
end


function TrainingData(trunc::Int, nsteps::Int, IC::Int)
    data = load("experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(nsteps)_IC$(IC).jld2")

    data_norm = Float32.(data["data_norm"])
    μ = Float32.(data["μ"])
    σ = Float32.(data["σ"])

    X = data_norm[:, 1:end-1, :]
    Y = data_norm[:, 2:end,   :]

    N = (nsteps - 1) * IC
    n_train = Integer(0.8 * N)

    # Creating training pairs x_i and x_{i+1}
    X_train = reshape(X, size(data_norm,1), :)[:,1:n_train]
    Y_train = reshape(Y, size(data_norm,1), :)[:,1:n_train]

    X_valid = reshape(X, size(data_norm,1), :)[:,n_train+1:N]
    Y_valid = reshape(Y, size(data_norm,1), :)[:,n_train+1:N]

    println("Loaded data with ", N ," data points")

    return TrainingData(trunc, nsteps, IC, X_train, Y_train, X_valid, Y_valid, μ, σ)
end

test = TrainingData(5,8,1000)




