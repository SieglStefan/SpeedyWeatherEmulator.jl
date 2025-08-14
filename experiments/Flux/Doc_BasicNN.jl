using Flux, Statistics, ProgressMeter
using CUDA
using Plots



### TRAINING THE MODEL ###

# Choosing gpu or cpu
use_gpu = CUDA.functional() && true
device = use_gpu ? gpu : cpu

println("GPU found: ", use_gpu)


# Generate data
noisy = rand(Float32, 2, 1000)
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]


# Define model
model = Chain(
    Dense(2 => 3, tanh),
    BatchNorm(3),
    Dense(3 => 2)) |> device


# Using untrained model
out1 = model(noisy |> device)
probs1 = softmax(out1) |> cpu


# Generating test data and optimizer
target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((noisy,target), batchsize=64, shuffle=true);


# Implementing optimiser (!!! different to doc !!!)
η = 0.01
opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, model)


# Training loop (!!! in the lecture "train!" is used!, here: more control!)
losses = []
@showprogress for epoch in 1:1_000
    for xy_cpu in loader
        x,y = xy_cpu |> device
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat,y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)
    end
end


# Using trained model
out2 = model(noisy |> device)
probs2 = softmax(out2) |> cpu
println(mean((probs2[1,:] .> 0.5) .== truth))



### PLOTTING THE RESULTS ###

# Plotting XOR results
p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network", legend=false)

display(plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330)))


# Plotting the loss function
display(plot(losses; xaxis=(:log10, "iteration"),
    yaxis="loss", label="per batch"))
n = length(loader)
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200)



