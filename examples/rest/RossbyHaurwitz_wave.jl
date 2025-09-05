using SpeedyWeatherEmulator

# Define Rossby-Haurwitz wave
m = 4
ω = 7.848e-6
K = 7.848e-6

ζ(λ, θ, σ) = 2ω*sind(θ) - K*sind(θ)*cosd(θ)^m*(m^2 + 3m + 2)*cosd(m*λ)


