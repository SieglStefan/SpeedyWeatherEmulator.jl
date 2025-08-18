module ModelDataHandling

using JLD2


using ..BasicStructs
using ..ModelStructs


export save_model, load_model, save_losses, load_losses


"""
    save_model(tm:TrainedModel)  

Saves trained model `tm` according to tm.sim_para for later use.

# Arguments
- `tm:TrainedModel`: Storaged model

# Returns
- `nothing`
"""
function save_model(tm::TrainedModel)
    filepath = create_DIR(sim_para = tm.sim_para, type="model")         # Create the storage DIR

    jldsave(filepath; tm)                                               # Save the model
    @info "Model saved at: $filepath"                                   # Information, if and where the model is saved

    return nothing
end


"""
    load_model(sim_para::SimPara)  

Loads existing trained model from data/model/ according to sim_para.

# Arguments
- `sim_para::SimPara`:      Simulation parameters (key) of the stored simulation data

# Returns
- `model::TrainedModel`:    Returns the loaded model
"""
function load_model(sim_para::SimPara)                                  # Create the storage DIR
    filepath = create_DIR(sim_para = sim_para, type="model")

    model = JLD2.load(filepath, "tm")                                   # Load the model
    @info "Model $filepath loaded"                                      # Information, if and from where the model is loaded

    return model
end


"""
    save_losses(losses::Losses)  

Saves losses `losses` according to tm.sim_para for later use.

# Arguments
- `losses::Losses`: Storaged losses

# Returns
- `nothing`
"""
function save_losses(losses::Losses)
    filepath = create_DIR(sim_para = losses.sim_para, type="losses")       # Create the storage DIR

    jldsave(filepath; losses)                                               # Save the losses
    @info "Losses saved at: $filepath"                                  # Information, if and where the model is saved

    return nothing
end


"""
    load_losses(sim_para::SimPara)  

Loads existing losses from data/losses/ according to sim_para.

# Arguments
- `sim_para::SimPara`: Simulation parameters (key) of the stored losses

# Returns
- `losses::Losses`: Returns the loaded losses
"""
function load_losses(sim_para::SimPara)                      
    filepath = create_DIR(sim_para = sim_para, type="losses")           # Create the storage DIR

    losses = JLD2.load(filepath, "losses")                                  # Load the losses
    @info "Losses $filepath loaded"                                     # Information, if and from where the losses are loaded

    return losses
end


"""
    create_DIR(;sim_para::SimPara, type::String) 

Creates a DIR for saving and loading models and losses.

# Arguments
- `sim_para::SimPara`: Simulation parameters (key) for the DIR generation.
- `type::String`: Switches the type, depending what is stored, possibilietes: `model` or `losses`

# Returns
- `filepath::String`: Returns the filepath
"""
function create_DIR(;sim_para::SimPara, type::String)
    dir = joinpath(@__DIR__, "..", "..", "data", type)
    filename = type *
        "T$(sim_para.trunc)_" *                                         
        "nsteps$(sim_para.n_steps)_" *
        "IC$(sim_para.n_ic)_" *
        "key$(sim_para.storage_key).jld2"
    filepath = normpath(joinpath(dir, filename))

    return filepath
end



end