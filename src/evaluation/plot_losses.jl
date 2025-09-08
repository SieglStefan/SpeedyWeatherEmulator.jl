using Plots
using Statistics

"""
    plot_losses(losses::Losses)

Plot training and validation losses stored in a `Losses` container.

# Description
- Plots training loss per batch (log-log scale).
- Adds epoch-averaged training and validation loss curves.
- Returns a `Plots.Plot` object for further customization or saving.

# Arguments
- `losses::Losses`: Container with training/validation loss history and number of batches per epoch.
- `title::String="Losses of the emulator"`: Optional argument for different plot titles (e.g. differen simulation parameters).

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
function plot_losses(losses::Losses; title::String="Losses of the emulator")

    # Printing information
    (; train, valid, bpe_train, bpe_valid) = losses

    @info "Batches per epoch (in the training set): $bpe_train"
    @info "Number of epochs: $(Integer(length(train) / bpe_train))"


    # Plotting the training loss per batch
    p = Plots.plot(train; xaxis=(:log10, "batches"),
        yaxis=(:log10, "loss"), label="training loss per batch", title=title)

    # Plotting the training loss per epoch
    Plots.plot!(bpe_train:bpe_train:length(train), 
        mean.(Iterators.partition(train, bpe_train)),
        label="training loss epoch mean", dpi=200, lw=3)

    # Plotting the validation loss per epoch
    Plots.plot!(bpe_train:bpe_train:length(train),
        mean.(Iterators.partition(valid, bpe_valid)),
        label="validation loss epoch mean", dpi=800, lw=3, color=:black)


    return p
end