# Functions & Types

## Core
@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = ["src/core/basic_structs.jl", "src/core/utils.jl"]
Order = [:type, :function]
Private = false
@end

## Data I/O
@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = ["src/data/generate_raw_data.jl", "src/data/io.jl"]
Order = [:type, :function]
Private = false
@end

## Formatting / Utils
@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = ["src/formatting/zscore.jl", "src/formatting/coeff_tools.jl"]
Order = [:type, :function]
Private = false
@end

## Emulator
@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = ["src/emulator/neural_network.jl", "src/emulator/emulator.jl", "src/emulator/losses.jl"]
Order = [:type, :function]
Private = false
@end

## Training & Evaluation
@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = ["src/training/train_emulator.jl", "src/evaluation/compare_emulator.jl"]
Order = [:type, :function]
Private = false
@end

## Plotting
@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = ["src/plotting/plots.jl"]
Order = [:type, :function]
Private = false
@end