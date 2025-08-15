using Flux, Statistics, ProgressMeter
using CUDA
using Plots
using JLD2

dir = "experiments/Flux/Flux_SavedModels"; mkpath(dir)

### BASIC LOADING / SAVING OF A MODEL

# Define model
struct MyModel
    net
end
Flux.@layer MyModel

MyModel() = MyModel(Chain(
    Dense(10 => 5, relu),
    Dense(5 => 2)
))

model1 = MyModel()


# Saving model
model_state1 = Flux.state(model1)

jldsave(dir * "SavedModel.jld2"; model_state1)


# Loading model
model_state2 = JLD2.load(dir * "SavedModel.jld2", "model_state1")
model2 = MyModel();

Flux.loadmodel!(model2, model_state2)


# Testing
println("Loaded and Saved model are equal: ", 
    Flux.params(model1) == Flux.params(model2))







### SAVING A SERIES OF MODELS

# Saving many models + opt_state to continue if failure
opt_state1 = Optimisers.setup(Optimisers.Adam(0.01), model1)

for epoch1 in 1:10
    if epoch1 == 5
        break
    end
    # train model ...
    jldsave(dir * "model-epoch$(epoch1).jld2";
    model_state1 = Flux.state(model1),
    opt_state1,
    loss1 = Float32(epoch1),
    epoch1
    )
end


# Loading latest models + opt_state 
files = filter(f -> endswith(f, ".jld2"), readdir(dir; join=true))

if !isempty(files)
    # Find newest file
    latest_file = files[argmax(mtime.(files))]
    println("Load latest model: ", latest_file)

    # Load mode
    model_state2 = JLD2.load(latest_file, "model_state1")
    opt_state2 = JLD2.load(latest_file, "opt_state1")
    loss2 = JLD2.load(latest_file, "loss1")
    epoch1 = JLD2.load(latest_file, "epoch1")

    Flux.loadmodel!(model2, model_state2)

    # Testing
    println("Loaded and Saved model are equal: ", 
        Flux.params(model1) == Flux.params(model2))
    display(opt_state2 == opt_state1)
    println("loss: ", loss2)
    println("epoch: ", epoch1)
else
    println("No saved model found!")
end






