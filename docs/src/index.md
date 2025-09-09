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

For a quick overview of applications of SpeedyWeatherEmulator.jl, there is a project report available in the package repository: [Project Report](https://github.com/username/SpeedyWeatherEmulator.jl/blob/main/ProjectReport.pdf).
It briefly introduces the necessity and background of emulators and explains and discusses the [Examples](examples.md) as applications.


## Contents

```@contents
Pages = ["running_SWE.md", "examples.md", "functions_index.md"]
Depth = 2
```


## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/SieglStefan/SpeedyWeatherEmulator.jl")
```