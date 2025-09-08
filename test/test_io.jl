using Test
using SpeedyWeatherEmulator
    

@testset "test_data_path" begin
    # Testing path creation for different simulation parameters (only checking path endings)
    sim1 = SimPara(trunc=5, n_data=48, n_ic=1000)
    path1 = data_path(sim1, type="raw_data")
    @test endswith(path1, "raw_data_T5_ndata48_IC1000_ID")

    sim2 = SimPara(trunc=5, n_data=48, n_ic=1000, id_key="_TEST")
    path2 = data_path(sim2, type="sim_data")
    @test endswith(path2, "sim_data_T5_ndata48_IC1000_ID_TEST.jld2")

    sim3 = SimPara(trunc=45, n_data=96, n_ic=2000)
    path3 = data_path(sim3, type="emulator")
    @test endswith(path3, "emulator_T45_ndata96_IC2000_ID.jld2")

    sim4 = SimPara(trunc=5, n_data=48, n_ic=1000, t_step=2.0)
    path4 = data_path(sim4, type="losses")
    @test endswith(path4, "losses_T5_ndata48_IC1000_ID.jld2")

    # Testing if a different timestep still leads to the same path
    path5 = data_path(sim3, type="emulator")
    @test path3 == path5

    # Testing if the whole path is correct using a temporary path tmp
    mktempdir() do tmp
        path = data_path(sim1; type="raw_data", path=tmp)
        expected = joinpath(tmp, "raw_data_T5_ndata48_IC1000_ID")
        @test path == expected                      # different t_step do NOT lead to different paths
    end
end


@testset "test_delete_data" begin
    sim = SimPara(trunc=5, n_data=1, n_ic=1)

    mktempdir() do tmp
        # 1st case: No folder exists
        path1, cancel1 = delete_data(sim; type="raw_data", path=tmp)
        @test cancel1 == false                       # no cancelation of data process
        @test isdir(path1) == false                  # data does not exist anymore

        # 2nd case: Folder exists but overwrite=false
        mkpath(path1)                                # Create folder
        path2, cancel2 = delete_data(sim; type="raw_data", overwrite=false, path=tmp)
        @test cancel2 == true                       # cancel data process
        @test isdir(path2) == true                  # data still exists                 

        # 3rd case: Folder exists and overwrite=true
        #mkpath(path)
        path3, cancel3 = delete_data(sim; type="raw_data", overwrite=true, path=tmp)
        @test cancel3 == false                      # no cancelation of data process
        @test isdir(path3) == false                 # data does not exist anymore


        # Testing if overwriting is possible for files
        file, _ = delete_data(sim; type="emulator", path=tmp)
        # Simulate existing file
        mkpath(dirname(file)); touch(file)          # create file
        file, cancel = delete_data(sim; type="emulator", overwrite=true, path=tmp)
        @test cancel == false                       # no cancelation of data process
        @test isfile(file) == false                 # file is deleted
    end
end


@testset "test_save_data_load_data" begin
    sim = SimPara(trunc=5, n_data=1, n_ic=1)
    sim_data = SimData(sim, rand(Float32, 2, 3, 1))

    mktempdir() do tmp
        # Save and load data
        save_data(sim_data; overwrite=true, path=tmp)
        loaded = load_data(sim; type="sim_data", path=tmp)

        # Comparing the saved/loaded data with the actual data
        @test typeof(loaded) == SimData
        @test loaded.sim_para == sim_data.sim_para
        @test loaded.data == sim_data.data
    end
end

