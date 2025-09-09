using SpeedyWeatherEmulator
using Plots, CairoMakie
using Random

##### Plots for the chapters "Basic Workflow" of "Running SpeedyWeahterEmulator.jl"
        # (raw_data is manually deleted after plot saving)

# Fix the seed for reproducibility
Random.seed!(1234)


# Definie simulation parameters
sim_para = SimPara(trunc=5, n_data=24, n_ic=800, id_key="_basic_workflow")

# Generate raw simulation data
generate_raw_data(sim_para)

# Create formatted simulation data
sim_data = SimData(sim_para)
fd = FormattedData(sim_data)

# Define neural network and train the emulator
nn = NeuralNetwork()
em, losses = train_emulator(nn, fd)

# Plot and save the loss curve for inspection
p = plot_losses(losses)
display(p)
#Plots.savefig(p, joinpath(@__DIR__, "plots", "losses_BWF.pdf"))

# Define test vorticity for comparison
vor0 = fd.data_pairs.x_test[:,10,1]
vor_sw = fd.data_pairs.y_test[:,10+6,1]

# Calculate emulator vorticity
vor_em = vor0
for _ in 1:6
    global vor_em = em(vor_em)
end

# Plot and save vorticity heatmaps
p1 = plot_heatmap(vor0, trunc=5, title="Initial Vorticity vor0", range=(-2.5e-5, +2.5e-5))
p2 = plot_heatmap(vor_sw, trunc=5, title="Target Vorticity vor_sw", range=(-2.5e-5, +2.5e-5))
p3 = plot_heatmap(vor_em, trunc=5, title="Emulated Vorticity vor_em", range=(-2.5e-5, +2.5e-5))

display(p1)
display(p2)
display(p3)


#CairoMakie.save(joinpath(@__DIR__, "plots", "vor0_BWF.pdf"), p1)
#CairoMakie.save(joinpath(@__DIR__, "plots", "vor_sw_BWF.pdf"), p2)
#CairoMakie.save(joinpath(@__DIR__, "plots", "vor_em_BWF.pdf"), p3)
