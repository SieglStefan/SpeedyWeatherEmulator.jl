module SpeedyWeatherEmulator

using SpeedyWeather

# core
include("core/basic_structs.jl")
include("core/utils.jl")

# io 1
include("io/utils_io.jl")

# data
include("data/generate_raw_data.jl")
include("data/build_sim_data.jl")
include("data/format_sim_data.jl")

# emulator
include("emulator/zscore_trafo.jl")
include("emulator/emulator_structs.jl")
include("emulator/compare_emulator.jl")
include("emulator/train_emulator.jl")

# io 2
include("io/io.jl")

# evaluation
include("evaluation/_plot_forecast_test.jl")
include("evaluation/plot_losses.jl")
include("evaluation/plot_heatmap.jl")

#export myfuncs

# optional pre-comp





end

