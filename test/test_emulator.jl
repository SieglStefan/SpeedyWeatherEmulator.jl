using Test
using SpeedyWeatherEmulator
using Flux, Statistics


@testset "test_zscore" begin
    μ = Float32[0, 1]
    σ = Float32[1, 2]
    stats = ZscorePara(μ, σ)

    # Testing transformation for simple data
    x = Float32[0 2; 1 3]   
    z = zscore(x, stats)
    x_back = inv_zscore(z, stats)

    @test z[1,1] ≈ (x[1,1] - μ[1]) / σ[1]           # expecting z = [(x_i - μ)/σ]
    @test z[2,2] ≈ (x[2,2] - μ[2]) / σ[2]           # expecting z = [(x_i - μ)/σ]

    @test x_back ≈ x                                # backtrafo is approx. equal to origin data
end


@testset "test_emulator_struct" begin
    sim = SimPara(trunc=5, n_data=2, n_ic=1)
    nn = NeuralNetwork(io_dim=4, hidden_dim=8, n_hidden=2)
    stats = ZscorePara(zeros(Float32, 4), ones(Float32, 4))

    em = Emulator(nn, stats, sim)

    # Testing conversion of input
    x = rand(Float32, 4)
    y = em(x)
    @test size(y) == size(x)                        # input = output size

    X = rand(Float32, 4, 7)
    Y = em(X)
    @test size(Y) == size(X)                        # input = output size
end


@testset "test_compare_emulator" begin
    sim = SimPara(trunc=5, n_data=2, n_ic=1)
    nn = NeuralNetwork(io_dim=2, hidden_dim=2, n_hidden=1)
    stats = ZscorePara(zeros(Float32, 2), ones(Float32, 2))

    emu_id = Emulator(sim, Chain(x -> x), stats)    # identiy chain

    # Test if identiy emulator works
    x_test = Float32[1 2; 3 4]
    y_test = copy(x_test)
    err = compare_emulator(emu_id; x_test=x_test, y_test=y_test)
    @test isapprox(err, 0.0; atol=1e-4)             # base and target are approx. equal (identiy)
end


@testset "test_train_emulator" begin
    sim = SimPara(trunc=5, n_data=5, n_ic=2)   
    n_coeff = calc_n_coeff(trunc=sim.trunc)

    data = rand(Float32, 2*n_coeff, sim.n_data, sim.n_ic)
    sim_data = SimData(sim, data)
    fd = FormattedData(sim_data; splits=(train=0.5, valid=0.25, test=0.25))

    nn = NeuralNetwork(io_dim=2*n_coeff, hidden_dim=4, n_hidden=1)

    # Test if training creates a emulator object and nonempty losses
    emu, losses = train_emulator(nn, fd; batchsize=1, n_epochs=2, η0=0.01)      # small test parameters

    @test typeof(emu) == Emulator
    @test isempty(losses.train) == false
    @test isempty(losses.valid) == false
end
