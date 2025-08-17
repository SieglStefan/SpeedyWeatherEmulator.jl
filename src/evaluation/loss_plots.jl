using JLD2, Plots, Statistics

include(joinpath(@__DIR__, "..", "utils", "BasicStructs.jl"))
using .BasicStructs


function plot_losses(sim_para::SimPara)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    
    filename = "losses_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))
    losses = JLD2.load(filepath, "losses")


    # Plotting the loss functions
    display(plot(losses.train; xaxis=(:log10, "batches"),
        yaxis=(:log10, "loss"), label="per batch"))
        
    bpe_t = losses.bpe_train
    bpe_v = losses.bpe_valid
    @info "Batches per epoch (in the training set): $bpe_t"
    @info "Number of epochs: $(Integer(length(losses.train) / bpe_t))"

    plot!(bpe_t:bpe_t:length(losses.train), 
        mean.(Iterators.partition(losses.train, bpe_t)),
        label="epoch mean", dpi=200, lw=3
    )

    plot!(bpe_t:bpe_t:length(losses.train),
        mean.(Iterators.partition(losses.valid, bpe_v)),
        label="val epoch mean", dpi=800, lw=3, color=:black
    )
end

sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000)
plot_losses(sim_para)