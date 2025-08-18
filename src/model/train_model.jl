module TrainModel

using JLD2, Plots
using Flux, Optimisers, Statistics, ProgressMeter
using CUDA

using ..BasicStructs
using ..ZScoreUtils
using ..ModelStructs
using ..DataFormatting


export train_model


### TRAINING THE MODEL ###

function train_model(nn::NeuralNetwork, fd::FormattedData; 
        batchsize::Int = 32, n_epochs::Int=300, η0::Real=0.001)

    μ = Float32.(vec(mean(fd.data_pairs.x_train; dims=2)))
    σ = Float32.(vec(std(fd.data_pairs.x_train; dims=2)))
    norm_stats = NormStats(μ, σ)

    tm = TrainedModel(nn, norm_stats, fd.sim_para)


    x_train_norm = zscore(fd.data_pairs.x_train, norm_stats)
    y_train_norm = zscore(fd.data_pairs.y_train, norm_stats)

    x_valid_norm = zscore(fd.data_pairs.x_valid, norm_stats)
    y_valid_norm = zscore(fd.data_pairs.y_valid, norm_stats)


    loader_train = Flux.DataLoader((x_train_norm, y_train_norm),
                                   batchsize=batchsize, shuffle=true)
    loader_valid = Flux.DataLoader((x_valid_norm, y_valid_norm),
                                   batchsize=batchsize, shuffle=true)


    # Define model
    
    # Implementing optimiser
    η = η0
    opt = Optimisers.Adam(η)
    opt_state = Optimisers.setup(opt, tm.chain)

    losses = Losses(length(loader_train), length(loader_valid), fd.sim_para)

    # Training loop (!!! in the lecture "train!" is used!, here: more control!)
    # loss per batch
                
    @showprogress for epoch in 1:n_epochs
        for xy_cpu in loader_train            # loop over every batches of loader
            x,y = xy_cpu |> gpu
            loss, grads = Flux.withgradient(tm.chain) do chain
                Flux.mse(chain(x), y)
            end

            Optimisers.update!(opt_state, tm.chain, grads[1])
            push!(losses.train, Float32(loss))
        end

        if (epoch %30) == 0
            η /= 2
            Optimisers.adjust!(opt_state, η)
        end

        for (x,y) in loader_valid
            loss = Flux.mse(tm.chain(x), y)
            push!(losses.valid, Float32(loss))
        end
    end

    return tm, losses
end

end


