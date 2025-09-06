using SpeedyWeatherEmulator


### This Code is very similar to the counterpart "hyperpara_opt_coarse_eval.jl":
#       - Parts where the code differs, are marked with comments
#       - Defining a separate function for the hyperparameter optimization is not 
#               necessary for this simple consideration
#       - Moreover, keeping the function general (e.g. number of hyperparameters, 
#               different plot symbols, different forecast lengths) would be too 
#               complicated for this simple consideration.


Random.seed!(1234)

const TRUNC = 5
const N_DATA = 48
const N_IC = 1000

sim_para_loading = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC)
sim_data = load_data(sim_para_loading, type="sim_data")
fd = FormattedData(sim_data)


# Different parameters for the fine optimization
L_list = [1]                                                        # number of hidden layers
W_list_fine = [256, 384, 512, 640, 768, 896, 1024, 1536, 2048]      # number of neurons per hidden layer
W_list_coarse = [64, 128, 256, 512]                                 # number of neurons per hidden layer (coarse)


nn_warmup = NeuralNetwork(io_dim=2*calc_n_coeff(trunc=TRUNC),
                          hidden_dim=8,
                          n_hidden=1)

_, _ = train_emulator(nn_warmup, fd; n_epochs=1)


for L in L_list, W in W_list_fine

    # Skip training if W was already calculated in the coarse routine
    if W in W_list_coarse
        continue
    end


    id = "_hyperpara_L$(L)_W$(W)"
    sim_para = SimPara(trunc=TRUNC, n_data=N_DATA, n_ic=N_IC, id_key=id)

    nn = NeuralNetwork( io_dim=2*calc_n_coeff(trunc=TRUNC),
                        hidden_dim = W,
                        n_hidden = L)

    em, losses = train_emulator(nn, fd; sim_para=sim_para)

    save_data(em)
    save_data(losses)
end