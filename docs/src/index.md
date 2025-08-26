```@meta
CurrentModule = SpeedyWeatherEmulator
```

# SpeedyWeatherEmulator.jl

The SpeedyWeatherEmulator.jl project provides a framework for generating, processing, and emulating spectral vorticity data from the barotropic model in SpeedyWeather.jl.

It enables users to:
- configure and run controlled weather simulations,
- store and format spectral coefficients into training-ready datasets,
- train neural network emulators with normalization and logging (Emulator, train_emulator, Losses), and
- evaluate emulator performance against SpeedyWeather baselines with error metrics and visualizations.

The package is designed to streamline the workflow from simulation to machine learning surrogate modeling, making it easier to test neural network approaches for atmospheric dynamics.


## Contents

```@contents
Pages = ["man/quickstart.md", "man/theory.md", "man/api.md"]
Depth = 2
```


## Planned implementations / ToDos:

Planned implementations:

- Implementing code for further testing:
    - Emulator quality for Rossby-Haurwitz wave i.c. (as report results / example)
    - Emulator quality for multiple consecutive time steps (e.g. 12 * 1h) (as package function + report reults)
    - (if time left: Emulator quality for different simulation data parameters (e.g. for higher truncation) (as report results / example))
- (if time left: Implementing (simple) hyperparameter optimization (as package function))


ToDos:

- Writing test functions (not yet done, since the program's basic structure has only recently been set up)
- Checking the code quality (e.g. right types, type stability,...)
- Thorough testing of the emulator under various conditions to obtain results for the report
- Completion and revision of docstrings and code comments
- Completion and revision of GitHub documenation
- Writing the project report


## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/SieglStefan/SpeedyWeatherEmulator.jl")
```

```@index
```
