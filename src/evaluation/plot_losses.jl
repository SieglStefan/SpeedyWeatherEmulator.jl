using Plots



"""
    plot_losses(losses::Losses)

Plot training and validation losses stored in a `Losses` container.

# Description
- Plots training loss per batch (log-log scale).
- Adds epoch-averaged training and validation loss curves.
- Returns a `Plots.Plot` object for further customization or saving.

# Arguments
- `losses::Losses`: Container with training/validation loss history and number of batches per epoch.

# Returns
- `p::Plots.Plot`: Combined plot of training and validation losses.

# Notes
- Training batches per epoch = `losses.bpe_train`.
- Validation batches per epoch = `losses.bpe_valid`.
- Number of epochs is inferred as `length(losses.train) / bpe_train`.

# Examples
```julia
emu, losses = train_emulator(nn, fd)
p = plot_losses(losses)
display(p)  
```
"""
function plot_losses(losses::Losses)

    # Printing information
    bpe_t = losses.bpe_train
    bpe_v = losses.bpe_valid
    @info "Batches per epoch (in the training set): $bpe_t"
    @info "Number of epochs: $(Integer(length(losses.train) / bpe_t))"


    # Plotting the training loss per batch
    p = Plots.plot(losses.train; xaxis=(:log10, "batches"),
        yaxis=(:log10, "loss"), label="training loss per batch", title="Losses of the emulator")

    # Plotting the training loss per epoch
    Plots.plot!(bpe_t:bpe_t:length(losses.train), 
        mean.(Iterators.partition(losses.train, bpe_t)),
        label="training loss epoch mean", dpi=200, lw=3)

    # Plotting the validation loss per epoch
    Plots.plot!(bpe_t:bpe_t:length(losses.train),
        mean.(Iterators.partition(losses.valid, bpe_v)),
        label="validation loss epoch mean", dpi=800, lw=3, color=:black)


    return p
end