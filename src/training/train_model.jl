using JLD2, Plots
using Flux, Optimisers, Statistics, ProgressMeter
using CUDA

include(joinpath(@__DIR__, "..", "utils", "BasicStructs.jl"))
using .BasicStructs
include(joinpath(@__DIR__, "..", "utils", "EmulatorUtils.jl"))
using .EmulatorUtils
include(joinpath(@__DIR__, "..", "utils", "GeneralUtils.jl"))
using .GeneralUtils



### TRAINING THE MODEL ###

function train_model!(model, td; batchsize::Int = 32, n_epochs::Int=300, η0::Real=0.001)
    device = gpu

    # Generate data
    loader_train = Flux.DataLoader((td.data_pairs.x_train, td.data_pairs.y_train), batchsize=batchsize, shuffle=true)  # groups data in input: (x_val x N_batch) and target: (y_val x N_batch)                       
    loader_valid = Flux.DataLoader((td.data_pairs.x_valid, td.data_pairs.y_valid), batchsize=batchsize, shuffle=true)


    # Define model
    
    # Implementing optimiser
    η = η0
    opt = Optimisers.Adam(η)
    opt_state = Optimisers.setup(opt, model)

    losses = Losses(length(loader_train), length(loader_valid))

    # Training loop (!!! in the lecture "train!" is used!, here: more control!)
    # loss per batch
                
    @showprogress for epoch in 1:n_epochs
        for xy_cpu in loader_train            # loop over every batches of loader
            x,y = xy_cpu |> device
            loss, grads = Flux.withgradient(model) do m
                Flux.mse(m(x),y)
            end

            Optimisers.update!(opt_state, model, grads[1])
            push!(losses.train, Float32(loss))
        end

        if (epoch %30) == 0
            η /= 2
            Optimisers.adjust!(opt_state, η)
        end

        for (x,y) in loader_valid
            loss = Flux.mse(model(x), y)
            push!(losses.valid, Float32(loss))
        end
    end

    return losses
end


### SAVE THE MODEL

# Saving model
function save_model(model::NeuralNetwork, td::TrainingData)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "model_T$(td.sim_para.trunc)_nsteps$(td.sim_para.n_steps)_IC$(td.sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; model, td.norm_stats)
    @info "Model saved at: $filepath"
end


# Saving losses
function save_losses(losses::Losses, td::TrainingData)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "losses_T$(td.sim_para.trunc)_nsteps$(td.sim_para.n_steps)_IC$(td.sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; losses)
    @info "Losses saved at: $filepath"
end


sim_para = SimPara(trunc=5, n_steps=8, n_ic=100)
model = NeuralNetwork(input_dim=54, hidden_dim=128, output_dim=54)

td = TrainingData(sim_para)

losses = train_model!(model, td)

save_model(model, td)
save_losses(losses, td)