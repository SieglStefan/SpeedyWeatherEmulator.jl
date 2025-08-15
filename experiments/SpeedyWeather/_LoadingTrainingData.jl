using JLD2

struct TrainingData
    trunc::Int64
    nsteps::Int64
    IC::Int64
    X::Matrix{Float32}
    Y::Matrix{Float32}
    μ::Vector{Float32}
    σ::Vector{Float32}
end


function TrainingData(trunc::Int, nsteps::Int, IC::Int)
    data = load("experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(nsteps)_IC$(IC).jld2")

    data_norm = Float32.(data["data_norm"])
    μ = Float32.(data["μ"])
    σ = Float32.(data["σ"])

    X_temp = data_norm[:, 1:end-1, :]
    Y_temp = data_norm[:, 2:end,   :]

    # Creating training pairs x_i and x_{i+1}
    X = reshape(X_temp, size(data_norm,1), :)
    Y = reshape(Y_temp, size(data_norm,1), :)

    return TrainingData(trunc, nsteps, IC, X, Y, μ, σ)
end

test = TrainingData(5,8,10)

println("Loading was successful")


