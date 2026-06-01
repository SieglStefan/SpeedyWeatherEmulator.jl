
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SieglStefan.github.io/SpeedyWeatherEmulator.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SieglStefan.github.io/SpeedyWeatherEmulator.jl/dev/)
[![Build Status](https://github.com/SieglStefan/SpeedyWeatherEmulator.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SieglStefan/SpeedyWeatherEmulator.jl/actions/workflows/CI.yml?query=branch%3Amaster)


# SpeedyWeatherEmulator.jl

The **SpeedyWeatherEmulator.jl** package provides a compact framework for generating, processing, and emulating spectral vorticity data from the barotropic model in [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl).

It enables users to:

- configure and run controlled barotropic weather simulations,
- store and format spectral vorticity coefficients into training-ready datasets,
- train neural network emulators with normalization and logging, and
- evaluate emulator performance against SpeedyWeather baselines using error metrics and visualizations.

The package is designed to streamline the workflow from numerical simulation to machine learning emulation, making it easier to test neural-network-based approaches for simplified atmospheric dynamics.

For a quick overview of the motivation, workflow, and applications of **SpeedyWeatherEmulator.jl**, there is a [Project Report](https://github.com/SieglStefan/SpeedyWeatherEmulator.jl/blob/main/ProjectReport.pdf) available in the package repository.

The report briefly introduces the background of weather and climate emulators, explains the implemented workflow, discusses selected examples, and contains the list of references.

If you are interested in specific parts of the code or further implementation details, the package repository is linked here: [SpeedyWeatherEmulator.jl Repository](https://github.com/SieglStefan/SpeedyWeatherEmulator.jl).



## Basic Example

A minimal workflow consists of defining simulation parameters, generating SpeedyWeather data, formatting it for machine learning, and training a neural network emulator.

```julia
using SpeedyWeatherEmulator

# Define simulation setup
sim_para = SimPara(trunc=5, n_data=24, n_ic=800, id_key="_demo")

# Generate raw SpeedyWeather data
generate_raw_data(sim_para)

# Load and format data for machine learning
sim_data = SimData(sim_para)
fd = FormattedData(sim_data)

# Define and train neural network emulator
nn = NeuralNetwork()
em, losses = train_emulator(nn, fd)

# Plot training history
plot_losses(losses)
```

After training, the emulator can be applied iteratively to produce short vorticity forecasts and compared against the original SpeedyWeather.jl trajectory.