using SpeedyWeatherEmulator


const TRUNC = 5
const N_DATA = 48
const N_IC = 1000

sim_para_loading = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC)
sim_data = load_data(sim_para_loading, type="sim_data")
fd = FormattedData(sim_data)


L_list = [1,2,3]
W_list = [64, 128, 256, 512]


nn_warmup = NeuralNetwork(io_dim=2*calc_n_coeff(trunc=TRUNC),
                          hidden_dim=8,
                          n_hidden=1)

_, _ = train_emulator(nn_warmup, fd; n_epochs=1)

for L in L_list, W in W_list

    id = "_hyperpara_L$(L)_W$(W)"
    sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC, id_key=id)

    nn = NeuralNetwork( io_dim=2*calc_n_coeff(trunc=TRUNC),
                        hidden_dim = W,
                        n_hidden = L)

    em, losses = train_emulator(nn, fd; sim_para=sim_para)

    save_data(em)
    save_data(losses)
end