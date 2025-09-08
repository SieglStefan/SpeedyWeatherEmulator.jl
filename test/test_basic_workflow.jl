using Test
using SpeedyWeatherEmulator

@testset "test_basic_workflow" begin
    sim = SimPara(trunc=5, n_data=8, n_ic=4, n_spinup=1, id_key="_TEST")

    # Creating a temporary folder
    tmp = mktempdir()
    try
        
        # 1: Creating raw data
        generate_raw_data(sim; overwrite=true, path=tmp)
        raw_path = data_path(sim; type="raw_data", path=tmp)
        @test isdir(raw_path)

        # 2: Create SimData
        simdata = SimData(sim, tmp)
        n_coeff = calc_n_coeff(trunc=sim.trunc)
        @test size(simdata.data) == (2*n_coeff, sim.n_data, sim.n_ic)

        # 3: Create FormattedData
        fd = FormattedData(simdata; splits=(train=0.6, valid=0.2, test=0.2))
        @test size(fd.data_pairs.x_train, 1) == 2*n_coeff

        # 4: Train emulator
        nn = NeuralNetwork(io_dim=2*n_coeff, hidden_dim=4, n_hidden=1)
        emu, losses = train_emulator(nn, fd; batchsize=1, n_epochs=1)

        @test emu isa Emulator
        @test losses isa Losses
    finally
        # Forcing the deletion of the temporary data 
        GC.gc()
        rm(tmp; recursive=true, force=true)
    end
end
