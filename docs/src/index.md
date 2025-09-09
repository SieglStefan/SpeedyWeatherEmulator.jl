```@meta
CurrentModule = SpeedyWeatherEmulator
```

# SpeedyWeatherEmulator.jl

The SpeedyWeatherEmulator.jl package provides a framework for generating, processing, and emulating spectral vorticity data from the barotropic model in SpeedyWeather.jl.

It enables users to:
- configure and run controlled weather simulations,
- store and format spectral coefficients into training-ready datasets,
- train neural network emulators with normalization and logging, and
- evaluate emulator performance against SpeedyWeather baselines with error metrics and visualizations.

The package is designed to streamline the workflow from simulation to machine learning emulation, making it easier to test neural network approaches for atmospheric dynamics.

For a quick overview of applications of SpeedyWeatherEmulator.jl, there is a [Project Report](https://github.com/SieglStefan/SpeedyWeatherEmulator.jl/blob/main/ProjectReport.pdf) available in the package repository.
It briefly introduces the necessity and background of emulators and explains and discusses the [Examples](examples.md) featured in this documentation. It also contains the list of references.


If you are interested in specific parts of the code or other details, the package repository is linked here: [SpeedyWeatherEmulator.jl Repository](https://github.com/SieglStefan/SpeedyWeatherEmulator.jl).


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