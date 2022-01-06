using Test, Random, LinearAlgebra, Statistics, DelimitedFiles
import Zygote, Flux, Distributions, SpecialFunctions, LogExpFunctions, QuadGK, NPZ
import RestrictedBoltzmannMachines as RBMs

#= As far as I know, Github Actions uses Intel CPUs.
So it is faster to use MKL than OpenBLAS. =#
using MKL
