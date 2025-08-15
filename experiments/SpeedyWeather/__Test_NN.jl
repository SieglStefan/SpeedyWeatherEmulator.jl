using JLD2, Plots
using Flux, Optimisers, Statistics, ProgressMeter
using CUDA


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





### TRAINING THE MODEL ###

# Choosing gpu or cpu
use_gpu = CUDA.functional() && false
device = use_gpu ? gpu : cpu

println("GPU use: ", use_gpu)


# Generate data
td = TrainingData(5,8,1000)
loader = Flux.DataLoader((td.X, td.Y), batchsize=32, shuffle=true)


# Define model
model = Chain(
    Dense(54 => 128, relu),
    Dense(128 => 128, relu),
    Dense(128 => 54)
) |> device


# Implementing optimiser
η = 0.01
opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, model)


# Training loop (!!! in the lecture "train!" is used!, here: more control!)
losses = []
@showprogress for epoch in 1:100
    for xy_cpu in loader
        x,y = xy_cpu |> device
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.mse(y_hat,y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)
    end
end



### PLOTTING THE RESULTS ###

# Plotting the loss function
display(plot(losses; xaxis=(:log10, "iteration"),
    yaxis=(:log10, "loss"), label="per batch"))
n = length(loader)  # batches per epoch
println(n)
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200)




#

