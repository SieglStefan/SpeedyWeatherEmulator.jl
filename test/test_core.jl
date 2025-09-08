using Test
using SpeedyWeatherEmulator

@testset "test_calc_n_coeff" begin
    # Testing the right amount of complex spectral coefficients for specific truncations
    @test calc_n_coeff(trunc=5) == 27
    @test calc_n_coeff(trunc=45) == 1127
end