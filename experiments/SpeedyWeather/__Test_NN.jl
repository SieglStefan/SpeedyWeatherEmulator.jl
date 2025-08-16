using JLD2, Plots
using Flux, Optimisers, Statistics, ProgressMeter
using CUDA


struct TrainingData
    trunc::Int64
    nsteps::Int64
    IC::Int64
    X_train::Matrix{Float32}
    Y_train::Matrix{Float32}
    X_valid::Matrix{Float32}
    Y_valid::Matrix{Float32}
    µ::Vector{Float32}
    σ::Vector{Float32}
end


function TrainingData(trunc::Int, nsteps::Int, IC::Int)
    data = load("experiments/SpeedyWeather/training_data_T$(trunc)_nsteps$(nsteps)_IC$(IC).jld2")

    data_norm = Float32.(data["data_norm"])
    µ = Float32.(data["μ"])
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

    return TrainingData(trunc, nsteps, IC, X_train, Y_train, X_valid, Y_valid, µ, σ)
end






### TRAINING THE MODEL ###

# Choosing gpu or cpu
use_gpu = CUDA.functional() && true
device = use_gpu ? gpu : cpu

println("GPU use: ", use_gpu)


# Generate data
td = TrainingData(5,8,1000)
loader = Flux.DataLoader((td.X_train, td.Y_train), batchsize=32, shuffle=true)  # groups data in input: (x_val x N_batch) and target: (y_val x N_batch)                       
loaderVD = Flux.DataLoader((td.X_valid, td.Y_valid), batchsize=32, shuffle=true)


# Define model
model = Chain(
    Dense(54 => 128, relu),
    Dense(128 => 128, relu),
    Dense(128 => 54)
) |> device


# Implementing optimiser
η = 0.001
opt = Optimisers.Adam(0.001)
opt_state = Optimisers.setup(opt, model)


# Training loop (!!! in the lecture "train!" is used!, here: more control!)
losses = Float32[]          # loss per batch            
lossesVD = Float32[]        # loss per batch
              
@showprogress for epoch in 1:300
    for xy_cpu in loader            # loop over every batches of loader
        x,y = xy_cpu |> device
        loss, grads = Flux.withgradient(model) do m
            Flux.mse(m(x),y)
        end

        Flux.update!(opt_state, model, grads[1])
        push!(losses, Float32(loss))
    end

    if (epoch %30) == 0
        global η /= 2
        Optimisers.adjust!(opt_state, η)
    end

    for (x,y) in loaderVD
        loss = Flux.mse(model(x), y)
        push!(lossesVD, Float32(loss))
    end
end



### PLOTTING THE RESULTS ###

# Plotting the loss function
display(plot(losses; xaxis=(:log10, "batches"),
    yaxis=(:log10, "loss"), label="per batch"))

n = length(loader)  # batches per epoch (SAME AS n_batches_per_epoch)
nVD = length(loaderVD)

plot!(n:n:length(losses), 
    mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200, lw=3
)

plot!(n:n:length(losses),
    mean.(Iterators.partition(lossesVD, nVD)),
    label="val epoch mean", dpi=800, lw=3, color=:black
)



### SAVE THE MODEL PARAMETERS

# Saving model
model_state = Flux.state(model)
mu = td.µ

jldsave("experiments/SpeedyWeather/model_T$(td.trunc)_nsteps$(td.nsteps)_IC$(td.IC).jld2"; model_state, mu, td.σ)



model