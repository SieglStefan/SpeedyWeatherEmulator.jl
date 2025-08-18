module DataFormatting

using JLD2


using ..GeneralUtils
using ..SimDataHandling
using ..BasicStructs




export FormattedData

struct DataPairs
    x_train::Matrix{Float32}
    y_train::Matrix{Float32}
    x_valid::Matrix{Float32}
    y_valid::Matrix{Float32}
end

struct FormattedData
    sim_para::SimPara
    data_pairs::DataPairs
end

function FormattedData(sim_data::SimulationData; split::Float64=0.8)
    sim_para = sim_data.sim_para
    
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic

    data = Float32.(sim_data.data)

    x = data[:, 1:end-1, :]
    y = data[:, 2:end,   :]

    n_data = (n_steps - 1) * n_ic
    n_train = Integer(split * n_data)

    # Creating training pairs x_i and x_{i+1}
    x_train = reshape(x, size(data,1), :)[:,1:n_train]
    y_train = reshape(y, size(data,1), :)[:,1:n_train]

    x_valid = reshape(x, size(data,1), :)[:,n_train+1:n_data]
    y_valid = reshape(y, size(data,1), :)[:,n_train+1:n_data]


    return FormattedData(sim_para, DataPairs(x_train, y_train, x_valid, y_valid))
end


function FormattedData(sim_para::SimPara; split::Float64=0.8)
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic

    sim_data = load_sim_data(sim_para)
    data = Float32.(sim_data.data)

    if sim_para != sim_data.sim_para
        @warn "Error in loading!"
    end

    x = data[:, 1:end-1, :]
    y = data[:, 2:end,   :]

    n_data = (n_steps - 1) * n_ic
    n_train = Integer(split * n_data)

    # Creating training pairs x_i and x_{i+1}
    x_train = reshape(x, size(data,1), :)[:,1:n_train]
    y_train = reshape(y, size(data,1), :)[:,1:n_train]

    x_valid = reshape(x, size(data,1), :)[:,n_train+1:n_data]
    y_valid = reshape(y, size(data,1), :)[:,n_train+1:n_data]

    @info "Loaded formatted data with a total of $n_data data pairs"

    return FormattedData(sim_para, DataPairs(x_train, y_train, x_valid, y_valid))
end


end


