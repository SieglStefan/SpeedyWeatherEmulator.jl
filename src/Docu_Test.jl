###  STARTING POINT
# starting point:
sim_para = SimPara(trunc=5, n_steps=8, n_ic=1000)
# Now we also need the timestep and the spinup time, for that we test the timesteps:
display(plot_forecast_time(sim_para, t_step=6.0, t_spinup=30.0))
display(plot_forecast_time(sim_para, t_step=24.0, t_spinup=30.0))
# We see, 24h timesteps are too much for an easy T5 model


### SIMULATION DATA
# Create simulation data, save and load it:
sim_data = create_sim_data(sim_para)
save_sim_data(sim_data)
sim_data = load_sim_data(sim_para)
println("---------------------------------------------")
# Or do in one go:
sim_data = prepare_sim_data(sim_para)
println("---------------------------------------------")
# Format the simulation data, either calling the saved sim_data:
fd = FormattedData(sim_para)
# or giving it as argument:
fd = FormattedData(sim_data)


### TRAIN THE MODEL
# Define the structure of the neural network
nn = NeuralNetwork()
# Then, train the nn with some formatted data
tm, losses = train_model(nn, fd)
# Now you can save the model and losses for later purposes:
save_model(tm)
save_losses(losses)
# To access the saved model and losses, you need again the sim_para:
tm = load_model(sim_para)
losses = load_losses(sim_para)


### EVALUTATION
# Now we want to plot the losses of trained model:
display(plot_losses(losses))
# Now we want to know what the relative errors of 1 timestep are. We can either use the sim_para of the trainedmodel for the 
# test set:
compare_emulator(tm, all_coeff=true)
# or define our own:
#sim_para_comp = SimPara(trunc=5, n_steps=8, n_ic=10)
#compare_emulator(tm, all_coeff=false, sim_para=sim_para_comp)
# We can also define our own initial conditions, as here with the RossbyHaurwitz wave:

#m = 4
#ω = 7.848e-6
#K = 7.848e-6
#ζ(λ, θ, σ) = Float32.(2ω*sind(θ) - K*sind(θ)*cosd(θ)^m*(m^2 + 3m + 2)*cosd(m*λ))

#compare_emulator(tm, all_coeff=true, initial_cond = ζ)


sim_para_comp = SimPara(trunc=5, n_steps=8, n_ic=2)
sim_data = create_sim_data(sim_para_comp)
vec0 = sim_data.data[:,5,1]
vec1 = sim_data.data[:,6,1]

vec2 = sim_data.data[:,5,2]
vec3 = sim_data.data[:,6,2]

plot_vor_heatmap(vec0, 5)
plot_vor_heatmap(vec1, 5)
plot_vor_heatmap(tm(vec0),5)

plot_vor_heatmap(vec2, 5)
plot_vor_heatmap(vec3, 5)
plot_vor_heatmap(tm(vec2),5)







display(plot_losses(losses))

vor0 = sim_data.data[:,5,500]
vorSW = sim_data.data[:,6,500]
vorEM = em(vor0)

for vor in [vor0, vorSW, vorEM]
    display(plot_heatmap(vor, trunc=5))
end