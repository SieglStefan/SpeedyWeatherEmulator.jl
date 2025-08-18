module ModelDataHandling

using JLD2


using ..BasicStructs
using ..ModelStructs


export save_model, load_model, save_losses, load_losses


# Saving model
function save_model(tm::TrainedModel)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "model_T$(tm.sim_para.trunc)_nsteps$(tm.sim_para.n_steps)_IC$(tm.sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; tm)
    @info "Model saved at: $filepath"
end

function load_model(sim_para::SimPara)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "model_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    model = JLD2.load(filepath, "tm")
    @info "Model $filepath loaded"

    return model
end


# Saving losses
function save_losses(losses::Losses)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "losses_T$(losses.sim_para.trunc)_nsteps$(losses.sim_para.n_steps)_IC$(losses.sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    jldsave(filepath; losses)
    @info "Losses saved at: $filepath"
end

function load_losses(sim_para::SimPara)
    dir = joinpath(@__DIR__, "..", "..", "data", "model_data")
    filename = "losses_T$(sim_para.trunc)_nsteps$(sim_para.n_steps)_IC$(sim_para.n_ic).jld2"
    filepath = normpath(joinpath(dir, filename))

    losses = JLD2.load(filepath, "losses")
    @info "Losses $filepath loaded"

    return losses
end

end