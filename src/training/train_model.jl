using JLD2, Plots
using Flux, Optimisers, Statistics, ProgressMeter
using CUDA

include(joinpath(@__DIR__, "..", "utils", "structs.jl"))
using .DataStructs: TrainingData, MyModel, Losses




### TRAINING THE MODEL ###

# Choosing gpu or cpu
use_gpu = CUDA.has_cuda()
device = use_gpu ? gpu : cpu

println("GPU use: ", use_gpu)


# Generate data
td = TrainingData(trunc=5, n_steps=8, n_ic=1000)
loader_train = Flux.DataLoader((td.data_pairs.x_train, td.data_pairs.y_train), batchsize=32, shuffle=true)  # groups data in input: (x_val x N_batch) and target: (y_val x N_batch)                       
loader_valid = Flux.DataLoader((td.data_pairs.x_valid, td.data_pairs.y_valid), batchsize=32, shuffle=true)


# Define model
model = MyModel(54, 128, 54)


# Implementing optimiser
η = 0.001
opt = Optimisers.Adam(0.001)
opt_state = Optimisers.setup(opt, model)

losses = Losses()

# Training loop (!!! in the lecture "train!" is used!, here: more control!)
  # loss per batch
              
@showprogress for epoch in 1:3
    for xy_cpu in loader_train            # loop over every batches of loader
        x,y = xy_cpu |> device
        loss, grads = Flux.withgradient(model) do m
            Flux.mse(m(x),y)
        end

        Flux.update!(opt_state, model, grads[1])
        push!(losses.train, Float32(loss))
    end

    if (epoch %30) == 0
        global η /= 2
        Optimisers.adjust!(opt_state, η)
    end

    for (x,y) in loader_valid
        loss = Flux.mse(model(x), y)
        push!(losses.valid, Float32(loss))
    end
end



### PLOTTING THE RESULTS ###

# Plotting the loss function
display(plot(losses.train; xaxis=(:log10, "batches"),
    yaxis=(:log10, "loss"), label="per batch"))

n = length(loader)  # batches per epoch (SAME AS n_batches_per_epoch)
nVD = length(loaderVD)

plot!(n:n:length(losses.train), 
    mean.(Iterators.partition(losses.train, n)),
    label="epoch mean", dpi=200, lw=3
)

plot!(n:n:length(losses.train),
    mean.(Iterators.partition(losses.valid, nVD)),
    label="val epoch mean", dpi=800, lw=3, color=:black
)



### SAVE THE MODEL PARAMETERS

# Saving model
#model_state = Flux.state(model)
#mu = td.µ

#jldsave("experiments/SpeedyWeather/model_T$(td.trunc)_nsteps$(td.nsteps)_IC$(td.IC).jld2"; model_state, mu, td.σ)



#model