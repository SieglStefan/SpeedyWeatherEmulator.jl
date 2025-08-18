module DataFormatting

using JLD2
using ..SimDataHandling
using ..BasicStructs


export FormattedData


"""
    DataPairs(  x_train::Matrix{Float32}, y_train::Matrix{Float32},
                x_valid::Matrix{Float32}, y_valid::Matrix{Float32},
                x_test::Matrix{Float32}, y_test::Matrix{Float32})

Container for data pairs (x,y) = (vor(t), vor(t+t_step)) splitted in train- and validation set.

XXX

# Fields
- `x_train::Matrix{Float32}`: Container for training set vor(t).
- `x_train::Matrix{Float32}`: Container for training set vor(t+t_step).
- `x_valid::Matrix{Float32}`: Container for validation set vor(t).
- `x_valid::Matrix{Float32}`: Container for validation set vor(t+t_step).
- `x_test::Matrix{Float32}`: Container for test set vor(t).
- `x_test::Matrix{Float32}`: Container for test set vor(t+t_step).
"""
struct DataPairs
    x_train::Matrix{Float32}
    y_train::Matrix{Float32}
    x_valid::Matrix{Float32}
    y_valid::Matrix{Float32}
    x_test::Matrix{Float32}
    y_test::Matrix{Float32}
end


"""
    FormattedData(sim_para::SimPara, data_pairs::DataPairs)

Container for formatted (= data prepaired in (x,y) = (vor(t), vor(t+t_step))) data.

# Fields
- `sim_para::SimPara`: Simulation parameters for data formatting.
- `data_pairs::DataPairs`: Formatted data.
"""
struct FormattedData
    sim_para::SimPara
    data_pairs::DataPairs
end


"""
    FormattedData(sim_data::SimData; split::Float64=0.7, test_set::Bool=false, split_valid::Float64=0.15)

Constructor for the direct creation of formatted (x,y) = (vor(t), vor(t+t_step)) and splitted data from simulation data `sim_data`.

This Constructor is e.g also used for `compare_emulator` for the creation of data pairs, without the need of splitting the data in training-,
    validation- and testsets. (Setting `split_train` = 1.0)

# Fields
- `sim_data::SimData`: Simulation data, which gets formatted.
- `split_train::Float64 = 0.7`: Fraction of the formatted data which is used for training.
- `test_set::Bool = false`: Switch for enabling test sets.
- `split_valid::Float64 = 0.15`: Fraction of the formatted data which is used for validation. If `test_set` is `false`,
    the remaining data fraction of the training set is used (e.g. `split_train` = 0.7, results in 30% validation set).
"""
function FormattedData(sim_data::SimData; split_train::Float64=0.7, test_set::Bool=false, split_valid::Float64=0.15)
    sim_para = sim_data.sim_para
    data = sim_data.data

    trunc = sim_para.trunc
    n_steps = sim_para.n_steps
    n_ic = sim_para.n_ic

    n_data = (n_steps - 1) * n_ic                               # total number of data pairs coming from sim_data
    n_train = Integer(round(split_train * n_data))                     # number of data pairs in the training set

    # Calculate n_valid for array indexing below, depending on the exitence of a test set
    if test_set
        n_valid = Integer(split_valid * n_data)                 
    else
        n_valid = n_data - n_train                        
    end

    # Data pair components
    x = data[:, 1:end-1, :]                                     # list of vor(t)            (vor_1, vor_2,..., vor_{N-1})
    y = data[:, 2:end,   :]                                     # list of vor(t+t_step)     (vor_2, vor_3,..., vor_N)

    # Creating training pairs vor_i and vor_{i+1}
    x_train = reshape(x, size(data,1), :)[:, 1:n_train]
    y_train = reshape(y, size(data,1), :)[:, 1:n_train]

    x_valid = reshape(x, size(data,1), :)[:, n_train+1:n_train+n_valid]
    y_valid = reshape(y, size(data,1), :)[:, n_train+1:n_train+n_valid]

    x_test = reshape(x, size(data,1), :)[:, n_train+n_valid+1:n_data]
    y_test = reshape(y, size(data,1), :)[:, n_train+n_valid+1:n_data]


    return FormattedData(sim_para, 
                        DataPairs(x_train, y_train, x_valid, y_valid, x_test, y_test))
end



end


