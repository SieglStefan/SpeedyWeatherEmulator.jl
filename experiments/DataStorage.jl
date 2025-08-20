using SpeedyWeather, JLD2

spectral_grid = SpectralGrid(trunc=5, nlayers=1)
output = JLD2Output(output_dt = Hour(1))
model = BarotropicModel(spectral_grid; output=output)

simulation = initialize!(model)


SpeedyWeather.VorticityOutput()


@doc JLD2Output