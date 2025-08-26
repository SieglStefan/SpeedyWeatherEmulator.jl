# Functions & Types

## Core
```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/core/basic_structs.jl", 
            "src/core/utils.jl"]
Order = [:type, :function]
Private = false
```

## IO
```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/io/utils_io.jl", 
            "src/io/io.jl"]
Order = [:type, :function]
Private = false
```

## Data
```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/data/generate_raw_data.jl", 
            "src/data/build_sim_data.jl",
            "src/data/format_sim_data.jl"]
Order = [:type, :function]
Private = false
```

## Emulator
```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/emulator/zscore_trafo.jl", 
            "src/emulator/emulator_structs.jl",
            "src/emulator/compare_emulator.jl",
            "src/emulator/train_emulator.jl"]
Order = [:type, :function]
Private = false
```

## Evaluation
```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/evaluation/_plot_forecast_test.jl", 
            "src/evaluation/plot_losses.jl",
            "src/evaluation/plot_heatmap.jl"]
Order = [:type, :function]
Private = false
```
