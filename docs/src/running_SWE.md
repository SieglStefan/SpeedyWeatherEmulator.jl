# Running SpeedyWeatherEmulator.jl

This section introduces the core functionality of the package.  
After a short review of the basic workflow of the package it provides a step-by-step overview of how to generate simulation data, format it for training, build and train an emulator, saving/loading data, and evaluate its performance.



## Basic workflow

This brief introduction to the workflow is meant to illustrate how the different steps, functions, and data types of the package interact. I have deliberately omitted details such as additional functions, parameters, or default values. These can be found in the sections below or seen in action in the examples.

Every workflow in SpeedyWeatherEmulator.jl begins by defining the simulation parameters in a `SimPara` object. These parameters control the spectral truncation, the number of timesteps to be stored, the number of independent initial conditions and more:

```julia
sim_para = SimPara(trunc=5, n_data=20, n_ic=500)
```

With these parameters defined, raw vorticity data is generated using SpeedyWeather.jl and stored on disk.

```julia
generate_raw_data(sim_para)
```

The raw files are then loaded into a `SimData` container, which provides a consistent array layout. From this, formatted datasets are built, splitting the vorticity time series into train/validation/test pairs.

```julia
sim_data = SimData(sim_para)
fd = FormattedData(sim_data)
```

Next, a neural network architecture is defined and trained on the formatted data. This produces both an `Emulator` (the trained model) and a `Losses` object that tracks training progress.

```julia
nn = NeuralNetwork()
em, losses = train_emulator(nn, fd)
```

After the training, the mean relative error and max relative error for one step are printed to STDOUT:

```text
--------------------------------------
Mean relative error: 13.041 %
Max relative error:  62.957 %
--------------------------------------
```

The recorded losses can then be visualized:

```julia
display(plot_losses(losses))
```

![Loss curves](assets/doc_basicworkflow_lossplot.png)

Finally, we can directly compare emulator predictions with SpeedyWeather.jl outputs. Here we select one vorticity state (`vor0`), its SpeedyWeather forecast (`vorSW`), and the emulator’s prediction (`vorEM`) after three steps. 

```julia
vor0 = sim_data.data[:,10,500]
vorSW = sim_data.data[:,13,500]
vorEM = em(em(em(vor0)))
```

Each of them can be visualized as a heatmap,

```
plot_heatmap(vor0, trunc=5, title="Initial Vorticity vor0")
plot_heatmap(vorSW, trunc=5, title="Real SpeedyWeather.jl Vorticity vorSW")
plot_heatmap(vorEM, trunc=5, title="Predicted Emulator Vorticity vorEM")
```

resulting in:

![Initial Vorticity vor0](assets/doc_basicworkflow_vor0.png)

![Real SpeedyWeather.jl Vorticity vorSW](assets/doc_basicworkflow_vorSW.png)

![Predicted Emulator Vorticity vorEM](assets/doc_basicworkflow_vorEM.png)

The difference between the initial vorticity and the final vorticity is small, but it can be seen that the emulator already approximates the real SpeedyWeather.jl data reasonably well.



## Generating and formatting data

Before training a neural network emulator, we need to generate and structure data.  
This process has three steps: defining the simulation parameters, running SpeedyWeather.jl to create raw data, and finally preparing formatted datasets for machine learning.

### Simulation Parameters
Every workflow begins by specifying the simulation setup in a `SimPara` object.  
This struct collects the essential parameters of a barotropic SpeedyWeather.jl simulation:

- the spectral truncation (`trunc`, e.g. T5),  
- the number of stored timesteps after spin-up (`n_data`),  
- the number of independent initial conditions (`n_ic`),  
- and additional metadata such as spin-up length, timestep size, or a unique `id_key`.

Together, these parameters determine both the structure of the generated data and the folder name under which it is stored.  
For example:

```julia
sim_para = SimPara(trunc=5, n_data=50, n_ic=200, id_key="_test")
```

### Generating Raw Data
With the simulation parameters defined, raw data can be generated directly from SpeedyWeather.jl:

```julia
generate_raw_data(sim_para; overwrite=true)
```

This command creates a folder with one subdirectory per initial condition and stores the corresponding simulation output.
Each run includes spin-up steps, which are used internally but not stored in the final dataset.

Generating raw data can be slow and memory-intensive, since not only vorticity but also auxiliary diagnostics are written to disk.
It should therefore be used sparingly, ideally only when new datasets are required.
If existing data should not be overwritten, a new `id_key` can be supplied to disambiguate different runs.

### Structured Simulation Data
The raw output is then loaded into a consistent array layout using the `SimData` constructor:

```julia
sim_data = SimData(sim_para)
```

The resulting tensor has dimensions XXX DIMENSIONS where real and imaginary parts of the spectral coefficients are stacked along the first axis.
This format is optimized for efficient slicing over time and initial conditions, and serves as the basis for all later steps.

### Preparing Formatted Data
For machine learning, we need to turn continuous time series into input–output pairs.
The `FormattedData` constructor automates this process by pairing consecutive timesteps: XXX TIMESTEP VECKTOR

```math
\frac{\partial \zeta}{\partial t} + \nabla \cdot (\mathbf{u}(\zeta + f)) =
F_\zeta + \nabla \times \mathbf{F}_\mathbf{u} + (-1)^{n+1}\nu\nabla^{2n}\zeta
```

It then reshapes all samples into column vectors and splits the dataset into training, validation, and test sets.
By default, the split is 70 % training, 15 % validation, and 15 % test.

```julia
fd = FormattedData(sim_data; splits=(train=0.7, valid=0.15, test=0.15))
```

Since the targets are shifted forward in time, the total number of pairs is `(n_data − 1) * n_ic`.
This ensures that, for instance, `y[1]` corresponds exactly to the next timestep after `x[1]`.



## Training
Once the data pipeline is established, we can train a neural network to emulate the barotropic SpeedyWeather.jl model.  
This involves defining an architecture, normalizing the data, building an emulator, training it on input–output pairs, and finally comparing predictions against reference data.

### Neural Network Architecture
The emulator uses a simple feed-forward neural network with ReLU activations.  
The architecture is described by a `NeuralNetwork` object, which specifies the input/output size, the hidden layer width, and the number of hidden layers:

```julia
nn = NeuralNetwork(io_dim=54, hidden_dim=128, n_hidden=2)
```

This compact container makes it easy to experiment with different model sizes without touching the actual Flux code.

### Normalizsation by Z-Score
Before training, spectral coefficients are normalized coefficient-wise to zero mean and unit variance.
This is achieved with a `ZscorePara` struct, which stores the mean and standard deviation of the training set.
Normalization is always based on the training data alone to avoid information leakage.

XXX EQUATION ZSCORE

The emulator applies this transformation automatically when called

### The Emulator Wrapper
The Emulator struct bundles three pieces of information:

- the simulation parameters of the dataset,
- the neural network chain built from Flux,
- the Z-score normalization parameters.

This design keeps the metadata and the trained model tightly coupled.
For convenience, an emulator can be used like a function:

```julia
y_pred = emu(x)
```

where `x` are spectral coefficients at time `t` and the output is the emulator prediction at `t + Δt`.

### Logging Training Progress
During training, losses are collected in a `Losses` object.
It stores the mean-squared errors for each batch of the training and validation sets, and later also for the test set.
This makes it straightforward to plot learning curves and diagnose overfitting.

### Training the Emulator
The central routine is `train_emulator`, which orchestrates the entire process:

1. Compute Z-score parameters from the training set.
2. Build an `Emulator` according to the chosen architecture.
3. Normalize training and validation data.
4. Train with the Adam optimizer, starting with a given learning rate that halves every 30 epochs.
5. Record training and validation losses batch-by-batch.
6. Evaluate the trained emulator on the test set.

A typical training run looks like this:

```julia
emu, losses = train_emulator(nn, fd; batchsize=64, n_epochs=100, η0=0.0005)
```

At the end, the function prints error statistics on the test set and returns both the trained emulator and its recorded loss history.

### Evaluating Accurarcy
To quantify performance, the function `compare_emulator` applies the trained model to unseen test inputs and compares the predictions against SpeedyWeather.jl reference outputs.
It reports mean and maximum relative errors in percent, and can optionally display the error of each spectral coefficient separately.
The function also returns the overall average error, which is useful for automated evaluation. 
`compare_emulator` is also used in `train_emulator` for the test set.

```julia
compare_emulator(emu; 
    x_test=fd.data_pairs.x_test,
    y_test=fd.data_pairs.y_test,
    all_coeff=true)
```


## Saving/Loading data

Training an emulator often requires repeating experiments with different network architectures or datasets.  
To make results reproducible and avoid re-running costly simulations, SpeedyWeatherEmulator.jl provides simple functions for saving and reloading data containers.

### Unified File Paths
All saved objects — whether simulation data, trained emulators, or loss histories — are uniquely identified by their simulation parameters (`SimPara`).  
The helper functions `data_path` and `delete_data` ensure that every dataset receives a consistent folder or file name:

- Raw data are stored in dedicated folders with one subfolder per run.  
- All other types (`SimData`, `Emulator`, `Losses`) are saved as single `.jld2` files.

The file name includes truncation, number of timesteps, number of initial conditions, and the optional `id_key`, e.g.

```text
data/sim_data/sim_data_T5_ndata50_IC200_IDdemo.jld2
```

### Saving Data
Any supported container can be saved with a single call:

```julia
save_data(sim_data)
save_data(emu; overwrite=true)
```

The type of the object (`SimData`, `Emulator`, or `Losses`) is detected automatically.
By default, existing files are not overwritten. If the same identifier is used twice, saving will be canceled unless `overwrite=true` is specified.

### Loading Data
Previously saved objects can be reloaded at any time:

```julia
sim_data_loaded = load_data(sim_para; type="sim_data")
emu_loaded      = load_data(sim_para; type="emulator")
```

The only requirement is that the `SimPara` matches the one originally used for saving.
The function returns the object in its original type, making it seamless to continue training, evaluate a stored emulator, or plot old loss curves.



## Evaluation and Visualization

After training an emulator, it is often useful to visualize its performance and inspect example vorticity fields.  
SpeedyWeatherEmulator.jl provides a small set of plotting functions for these purposes.  
The plotting routines are intentionally kept simple: apart from an optional `title`, no additional styling parameters are exposed, in order to keep the interface clear and focused.

### Loss Curves
The function `plot_losses` visualizes the training history stored in a `Losses` object.  
It displays:

1. the raw training loss per batch (log–log scale),  
2. the mean training loss per epoch,  
3. the mean validation loss per epoch.  

This makes it easy to diagnose whether the network is converging properly and whether overfitting occurs.

```julia
emu, losses = train_emulator(nn, fd)
plot_losses(losses; title="Training history (T5)")
```

The number of epochs is inferred automatically from the batch size and dataset split.
The returned plot object can be further customized or saved using the standard Plots.jl interface.

### Vortivity Heatmaps
To inspect actual states of the barotropic model, the function `plot_heatmap` reconstructs a vorticity field from a spectral coefficient vector and shows it as a heatmap.
This requires specifying the spectral truncation to interpret the coefficient layout correctly:

```julia
vec = rand(Float32, 54)     # random coeffs for trunc=5
plot_heatmap(vec; trunc=5, title="Random vorticity field")
```

Internally, the coefficients are converted into a lower-triangular matrix and then transformed into a physical-space grid.
The resulting heatmap provides an intuitive view of the spatial vorticity pattern represented by the spectral state.