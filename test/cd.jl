using Random, LinearAlgebra, Statistics
using Flux, Zygote, FiniteDifferences, Test
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: first2, diffdrop, init_weights!, Data
