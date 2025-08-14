using GeoMakie, CairoMakie

lons = -180:180
lats = -90:90
field = [exp(cosd(l)) + 3(y/90) for l in lons, y in lats]

fig = Figure()
ax = GeoAxis(fig[1, 1],
    # Setting a beautiful view
    dest = "+proj=ortho +lat_0=20 +lon_0=+00",
    # No ticks visable (buggy)
    xticksvisible = false,    
    yticksvisible = false,
    xticklabelsvisible = false, 
    yticklabelsvisible = false,
    # Setting ticks matching SpeedyWeather.jl 
    xticks = -191.25:22.5:180,  
    yticks = -90:22.5:90 
)

surface!(ax, lons, lats, field;
    shading = NoShading,
    color = rotr90(GeoMakie.earth())
)

save("grid_T5.png", fig)
