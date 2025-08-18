module PlotLosses

using JLD2, Plots, Statistics


using ..BasicStructs
using ..ModelStructs
using ..ModelDataHandling

export plot_losses


function plot_losses(losses::Losses)

    # Plotting the loss functions
    p = plot(losses.train; xaxis=(:log10, "batches"),
        yaxis=(:log10, "loss"), label="per batch")
        
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

    return p
    
end

end
