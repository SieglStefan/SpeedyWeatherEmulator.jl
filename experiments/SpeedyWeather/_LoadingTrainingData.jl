using JLD2

struct TrainingData
    trunc::Int64
    nsteps::Int64
    IC::Int64
    X::Array{Float32, 3}
    Y::Array{Float32, 3}
    μ::Vector{Float32}
    σ::Vector{Float32}
end

function TrainingData(trunc::Int, nsteps::Int, IC::Int)
    data = load("experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(nsteps)_IC$(IC).jld2")

    data_norm = Float32.(data["data_norm"])
    μ = Float32.(data["μ"])
    σ = Float32.(data["σ"])

    # Creating training pairs x_i and x_{i+1}
    X = data_norm[:, 1:end-1, :]
    Y = data_norm[:, 2:end,   :]

    return TrainingData(trunc, nsteps, IC, X, Y, μ, σ)
end

test = TrainingData(5,8,10)

println("Loading was successful")


