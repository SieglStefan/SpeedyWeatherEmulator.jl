sim_para = SimPara(trunc = 5, n_steps=8, n_ic = 10)

m = 4
ω = 7.848e-6
K = 7.848e-6
ζ(λ, θ, σ) = Float32.(2ω*sind(θ) - K*sind(θ)*cosd(θ)^m*(m^2 + 3m + 2)*cosd(m*λ))

sim_data = create_sim_data(sim_para, t_step=6.0, initial_cond=ζ)
data = sim_data.data

for i in 1:8
    display(plot_vor_heatmap(data[:,i,1], 5, title="plot $i"))
end