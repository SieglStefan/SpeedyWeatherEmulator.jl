using SpeedyWeather

spectral_grid = SpectralGrid(trunc=5, nlayers=1,Grid=FullGaussianGrid)
model = BarotropicModel(spectral_grid1)
simulation = initialize!(model)

# Comparing different IC
simulation1 = initialize!(model)
run!(simulation1, period=Day(5))
vor1 = vec(simulation1.prognostic_variables.vor[:,1,1])
model1 = model

simulation2 = initialize!(model)
run!(simulation2, period=Day(5))
vor2 = vec(simulation2.prognostic_variables.vor[:,1,1])
model2 = model


println("Same Simulation: ", vor1 == vor2)
println("Same Model: ", model1 == model2)


