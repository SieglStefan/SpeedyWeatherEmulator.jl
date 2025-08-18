module SpeedyWeatherEmulator

using CairoMakie
using SpeedyWeather

# Core
include("core/basic_structs.jl")
using .BasicStructs

# Data
include("data/sim_data_handling.jl")
using .SimDataHandling
include("data/data_formatting.jl")
using .DataFormatting

# model
include("model/zscore_trafo.jl")
using .ZscoreTrafo
include("model/model_structs.jl")
using .ModelStructs
include("model/train_model.jl")
using .TrainModel
include("model/model_data_handling.jl")
using .ModelDataHandling

# evaluation
include("evaluation/plot_losses.jl")
using .PlotLosses
include("evaluation/plot_forecast_test.jl")
using .PlotForecastTest
include("evaluation/compare_emulator.jl")
using .CompareEmulator
include("evaluation/plot_vor_heatmap.jl")
using .PlotVorHeatmap



sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000, storage_key="_KEY1")
sim_data = load_sim_data(sim_para)

tm = load_model(sim_para)
losses = load_losses(sim_para)

#display(plot_losses(losses))
#display(plot_forecast_test(sim_para))

vor0 = sim_data.data[:, 4, 1]
vorSW = sim_data.data[:, 5, 1]
vorEM = tm(vor0)

#display(plot_vor_heatmap(vor0, sim_data.sim_para.trunc))
#display(plot_vor_heatmap(vorSW, sim_data.sim_para.trunc))
#display(plot_vor_heatmap(vorEM, sim_data.sim_para.trunc))
#display(plot_vor_heatmap(vor0, sim_data.sim_para.trunc))

sim_para_comp = SimPara(trunc=5, n_steps=8, n_ic=1000)
sim_data_comp = SimData(sim_para_comp)

compare_emulator(tm, sim_data_comp = sim_data_comp, all_coeff=true)


end
