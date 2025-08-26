# Functions & Types index


## Core

Defines the fundamental data structure (`SimPara`) describing  how simulation data is parameterized. Further it contains helpful utility functions (`calc_n_coeff`, `is_coeff_zero`).

```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/core/basic_structs.jl", 
            "src/core/utils.jl"]
Order = [:type, :function]
Private = false
```


## IO

Provides functions for handling data. Including creating data paths (`data_path`), deleting data (`delete_data`), and saving/loading data (`save_data`, `load_data`) for specific types.

```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/io/utils_io.jl", 
            "src/io/io.jl"]
Order = [:type, :function]
Private = false
```


## Data

Handles data generation and preparation. This includes creating raw simulation data with SpeedyWeather.jl (`generate_raw_data`), wrapping it into structured containers (`SimData`), and formatting it into train/validation/test sets (`DataPairs`, `FormattedData`).

```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/data/generate_raw_data.jl", 
            "src/data/build_sim_data.jl",
            "src/data/format_sim_data.jl"]
Order = [:type, :function]
Private = false
```


## Emulator

Handles emulator definition, normalization, training, and evaluation.
This includes Z-score normalization utilities (`ZscorePara`, `zscore`, `inv_zscore`), core emulator types (`NeuralNetwork`, `Emulator`, `Losses`), training workflow (`train_emulator`), and evaluation against SpeedyWeather.jl reference data (`compare_emulator`).

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

Handles evaluation and visualization of emulator and simulation output.
This includes plotting forecast stability for SpeedyWeather.jl runs (`plot_forecast_test`), plotting loss plots (`plot_losses`), and reconstructing vorticity fields from spectral coefficients as heatmaps (`plot_heatmap`).

```@autodocs
Modules = [SpeedyWeatherEmulator]
Pages = [   "src/evaluation/plot_forecast_test.jl", 
            "src/evaluation/plot_losses.jl",
            "src/evaluation/plot_heatmap.jl"]
Order = [:type, :function]
Private = false
```
