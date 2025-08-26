

"""
    DataPairs(x_train, y_train, x_valid, y_valid, x_test, y_test)

Container for paired data samples (x,y) = (vor(t), vor(t+Δt)), already split into
training, validation and test sets.

# Fields
- `x_train::Matrix{Float32}`: Training inputs vor(t).
- `y_train::Matrix{Float32}`: Training targets vor(t+Δt).
- `x_valid::Matrix{Float32}`: Validation inputs vor(t).
- `y_valid::Matrix{Float32}`: Validation targets vor(t+Δt).
- `x_test::Matrix{Float32}`: Test inputs vor(t).
- `y_test::Matrix{Float32}`: Test targets vor(t+Δt).

# Notes
- All matrices have the same row dimension = 2 * n_coeff.
- Columns index over independent time-pairs and ICs.
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
    FormattedData

Container for formatted data, i.e. paired vorticity samples (x,y) = (vor(t), vor(t+Δt)).

# Fields
- `sim_para::SimPara`: Container for parameters that define the simulation and data storage.
- `data_pairs::DataPairs`: The split and paired data.
"""
struct FormattedData
    sim_para::SimPara
    data_pairs::DataPairs
end


"""
    FormattedData(sim_data::SimData; splits=(train=0.7, valid=0.15, test=0.15))

Construct `FormattedData` directly from `SimData` by pairing consecutive time steps
    and splitting them into train/validation/test sets.

# Description
- Builds (x,y) pairs as
  - x = vor(t) = spectral vorticity state at time t,
  - y = vor(t+Δt) at the next time step.
- Reshapes all ICs and times into column vectors.
- Splits the resulting pairs according to the fractions in `splits`.

# Arguments
- `sim_data::SimData`: Container holding simulation data and corresponding sim. parameters.
- `splits::NamedTuple`: Fractions for train-, valid- and test-set.  
    Default = (0.7, 0.15, 0.15).

# Returns
- `::FormattedData`: Container holding formatted (paired) simulation data and corresponding sim. parametrs.

# Notes
- The number of total pairs is (`n_data` - 1) * `n_ic.
- Splits are normalized so that train + valid + test = 1.

# Examples
```julia
fd = FormattedData(sim_data; splits=(train=0.7, valid=0.2, test=0.1))
size(fd.data_pairs.x_train)  # (2*n_coeff, n_train)
```
"""
function FormattedData(sim_data::SimData; 
    splits::NamedTuple{(:train, :valid, :test),<:Tuple{Vararg{Real,3}}} = 
        (train=0.7, valid=0.15, test=0.15))

    sim_para = sim_data.sim_para
    data = sim_data.data

    total = splits.train + splits.valid + splits.test
    train_frac = splits.train / total
    valid_frac = splits.valid / total


    n_pairs = (sim_para.n_data - 1) * sim_para.n_ic                                # total number of data pairs coming from sim_data
    n_train = round(Int, train_frac * n_pairs)                     # number of data pairs in the training set
    n_valid = round(Int, valid_frac * n_pairs)


    # Data pair components
    x = reshape(data[:, 1:end-1, :], size(data,1), :)                                  # list of vor(t)            (vor_1, vor_2,..., vor_{N-1})
    y = reshape(data[:, 2:end,   :], size(data,1), :)                                  # list of vor(t+t_step)     (vor_2, vor_3,..., vor_N)

    # Creating training pairs vor_i and vor_{i+1}
    x_train = x[:, 1:n_train]
    y_train = y[:, 1:n_train]

    x_valid = x[:, (n_train+1):(n_train+n_valid)]
    y_valid = y[:, (n_train+1):(n_train+n_valid)]

    x_test = x[:, (n_train+n_valid+1):n_pairs]
    y_test = y[:, (n_train+n_valid+1):n_pairs]


    return FormattedData(sim_para, 
                        DataPairs(x_train, y_train, x_valid, y_valid, x_test, y_test))
end


