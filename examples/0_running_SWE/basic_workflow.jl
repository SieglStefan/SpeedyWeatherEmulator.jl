using SpeedyWeatherEmulator

# Definie simulation parameters
sim_para = SimPara(trunc=5, n_data=20, n_ic=500, id_key="_basic_workflow")

# Generate raw simulation data
generate_raw_data(sim_para)

# Create formatted simulation data
sim_data = SimData(sim_para)
fd = FormattedData(sim_data)

# Define neural network and train the emulator
nn = NeuralNetwork()
em, losses = train_emulator(nn, fd)

# Plot the loss curve for inspection
display(plot_losses(losses))

# Define vorticity for comparison
vor0 = sim_data.data[:,10,500]
vorSW = sim_data.data[:,13,500]
vorEM = em(em(em(vor0)))

# Plot vorticity heatmaps
plot_heatmap(vor0, trunc=5, title="Initial Vorticity vor0")
plot_heatmap(vorSW, trunc=5, title="Real SpeedyWeather.jl Vorticity vorSW")
plot_heatmap(vorEM, trunc=5, title="Predicted Emulator Vorticity vorEM")