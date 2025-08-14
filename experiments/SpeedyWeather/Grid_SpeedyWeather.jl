using SpeedyWeather
using GLMakie, GeoMakie

# parameters
T = 5
N_lat_half = T - 1

fig = globe(FullGaussianGrid, N_lat_half; interactive=true)
save("grid_T5_SW.png", fig);
fig


