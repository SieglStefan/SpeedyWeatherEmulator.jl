using Test
using SpeedyWeatherEmulator
using SpeedyWeather

@testset "test_vec_to_ltm" begin
    trunc = 5
    n = calc_n_coeff(trunc=trunc)
    vec = rand(Float32, 2*n)

    L = vec_to_ltm(vec, trunc)

    # Testing the right type
    @test isa(L, LowerTriangularMatrix{ComplexF32})

    # Testing the number of spectral coefficients in L
    @test length(L) == n
end