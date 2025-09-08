using Test
using SpeedyWeatherEmulator
using Statistics

@testset "test_generate_raw_data" begin
    sim = SimPara(trunc=2, n_data=2, n_ic=1, n_spinup=0, id_key="_TEST")
    
    # Testing raw data generatiom
    tmp = mktempdir()
    try                                     
        # Creation of raw data
        generate_raw_data(sim; overwrite=true, path=tmp)

        raw_root = data_path(sim; type="raw_data", path=tmp)
        run_dir  = joinpath(raw_root, "run_0001")
        out_file = joinpath(run_dir, "output.jld2")

        @test isdir(raw_root)                           # simulation folder exists
        @test isdir(run_dir)                            # first run folder exists
        @test isfile(out_file)                          # ouput.jld2 file exists
        @test filesize(out_file) > 0                    # ouput is not empty

        # Cancelation of raw data generation
        result = generate_raw_data(sim; overwrite=false, path=tmp)
        @test result == false                           # new data is not generated
        @test isdir(raw_root)                           # simulation folder still exists
        @test isfile(out_file)                          # ouput.jld2 file still exists

    finally
        # Forcing the deletion of the temporary data 
        GC.gc()                                     
        rm(tmp; recursive=true, force=true)
    end
end


@testset "test_simdata" begin
    sim = SimPara(trunc=5, n_data=7, n_ic=3)
    n_coeff = calc_n_coeff(trunc=sim.trunc)  
    sim_data = SimData(sim, zeros(Float32, 2*n_coeff, sim.n_data, sim.n_ic))

    # Testing the right size of the data field of SimData according to the simulation parameters
    @test size(sim_data.data) == (2*n_coeff, sim.n_data, sim.n_ic)      
end


@testset "test_formatteddata" begin
    sim = SimPara(trunc=5, n_data=6, n_ic=2)
    n_coeff = calc_n_coeff(trunc=sim.trunc)
    sim_data = SimData(sim, zeros(Float32, 2*n_coeff, sim.n_data, sim.n_ic))

    splits = (train=0.6, valid=0.2, test=0.2)
    fd = FormattedData(sim_data; splits=splits)
    dp = fd.data_pairs

    # Testing the overall number of datapairs
    n_pairs = (sim.n_data - 1) * sim.n_ic
    @test n_pairs == 10                                 # number of all datapairs

    # Testing the if the data-split is done right
    n_train = size(dp.x_train, 2)
    n_valid = size(dp.x_valid, 2)
    n_test  = size(dp.x_test,  2)

    @test n_train + n_valid + n_test == n_pairs
    @test n_train == 6                                  # number of datapairs in the training set
    @test n_valid == 2                                  # number of datapairs in the validation set
    @test n_test  == 2                                  # number of datapairs in the testing set

    # Testing the length of the columns (spectral coefficients)
    @test size(dp.x_train, 1) == 2*n_coeff              # number of real and imagninary coefficients
    @test size(dp.y_train, 1) == 2*n_coeff
    @test size(dp.x_valid, 1) == 2*n_coeff
    @test size(dp.y_valid, 1) == 2*n_coeff
    @test size(dp.x_test,  1) == 2*n_coeff
    @test size(dp.y_test,  1) == 2*n_coeff


    # Testing odd number of datapairs and splits
    sim = SimPara(trunc=5, n_data=39, n_ic=43)            
    sim_data = SimData(sim, zeros(Float32, 2*n_coeff, sim.n_data, sim.n_ic))

    splits = (train=11.4, valid=3.5, test=2.86)          # corresponds to (0.64..., 0.20..., 0.16...)
    fd = FormattedData(sim_data; splits=splits)
    dp = fd.data_pairs

    n_pairs = (sim.n_data - 1) * sim.n_ic               # =1634
    n_train = size(dp.x_train, 2)
    n_valid = size(dp.x_valid, 2)
    n_test  = size(dp.x_test,  2)

    @test n_train + n_valid + n_test == n_pairs
    @test n_train == 1049                               # number of datapairs in the training set
    @test n_valid == 322                                # number of datapairs in the validation set
    @test n_test  == 263                                # number of datapairs in the testing set
end