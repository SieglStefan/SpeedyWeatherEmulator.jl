module PlotLosses

using JLD2, Plots, Statistics
using ..BasicStructs
using ..ModelStructs
using ..ModelDataHandling


export plot_losses


"""
    plot_losses(losses::Losses)  

Plots the losses stored in `losses`.

# Arguments
- `losses::Losses`: Stores the losses, which are plotted.

# Returns
- `p`:              Plot of the losses.
"""
function plot_losses(losses::Losses)

    # Printing information
    bpe_t = losses.bpe_train
    bpe_v = losses.bpe_valid
    @info "Batches per epoch (in the training set): $bpe_t"
    @info "Number of epochs: $(Integer(length(losses.train) / bpe_t))"


    # Plotting the training loss per batch
    p = plot(losses.train; xaxis=(:log10, "batches"),
        yaxis=(:log10, "loss"), label="training loss per batch", title="Losses of the emulator")

    # Plotting the training loss per epoch
    plot!(bpe_t:bpe_t:length(losses.train), 
        mean.(Iterators.partition(losses.train, bpe_t)),
        label="training epoch mean", dpi=200, lw=3)

    # Plotting the validation loss per epoch
    plot!(bpe_t:bpe_t:length(losses.train),
        mean.(Iterators.partition(losses.valid, bpe_v)),
        label="validaiton epoch mean", dpi=800, lw=3, color=:black)


    return p
end



end
