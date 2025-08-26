module SpeedyWeatherEmulator

using SpeedyWeather

# core
include("core/basic_structs.jl")
include("core/utils.jl")
export SimPara, is_coeff_zero, calc_n_coeff

# io 1
include("io/utils_io.jl")
export data_path, delete_data

# data
include("data/generate_raw_data.jl")
include("data/build_sim_data.jl")
include("data/format_sim_data.jl")
export generate_raw_data, SimData, DataPairs, FormattedData

# emulator
include("emulator/zscore_trafo.jl")
include("emulator/emulator_structs.jl")
include("emulator/compare_emulator.jl")
include("emulator/train_emulator.jl")
export ZscorePara, zscore, inv_zscore
export NeuralNetwork, Emulator, Losses, compare_emulator, train_emulator

# io 2
include("io/io.jl")
export save_data, load_data

# evaluation
include("evaluation/_plot_forecast_test.jl")
include("evaluation/plot_losses.jl")
include("evaluation/plot_heatmap.jl")
export plot_losses, plot_forecast_test, vec_to_ltm, plot_heatmap

#export myfuncs

# optional pre-comp



end

