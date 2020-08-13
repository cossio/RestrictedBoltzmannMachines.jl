using Base: tail, front
using Test, Random
using RestrictedBoltzmannMachines

n = ((3,2,4), (4,3), (2,3))
B = 3

dbm = DBM((Binary(nl...) for nl in n)...)
randn!.(dbm.weights)
for layer in dbm.layers
    randn!(layer.θ)
end
rbms = Tuple(dbm)
for (l, rbm) in enumerate(rbms)
    @test RBM(dbm, Val(l)) == rbm
end

@test length(dbm) == length(n)
@test length(dbm.layers) == length(n)
@test size.(dbm.layers) == n
@test length(dbm.weights) == length(n) - 1
for l = 1:length(dbm)
    @test ndims(dbm.layers[l]) == ndims(dbm, l)
    @test ndims(dbm, l) == length(n[l])
end
for (l, w) in enumerate(dbm.weights)
    @test size(w) == (n[l]..., n[l+1]...)
end

x = ntuple(l -> rand(Bool, n[l]..., B), length(dbm))
@test size(energy(dbm, x)) == (B,)
energy(dbm, x) ≈ sum(energy.(rbms, front(x), tail(x)))
@inferred energy(dbm, x)

I1 = inputs_even_to_odd(dbm::DBM, x::Tuple)
I2 = inputs_odd_to_even(dbm::DBM, x::Tuple)
I = ntuple(l -> isodd(l) ? I1[l] : I2[l], length(dbm))
for l = 1:length(dbm)
    @test I[l] ≈ inputs_to(dbm, x, Val(l))
    if iseven(l)
        @test isnothing(I1[l])
    else
        @test isnothing(I2[l])
    end
end

@test I[1] ≈ inputs_h_to_v(rbms[1], x[2])
for l = 2:length(dbm)-1
    @test I[l] ≈ inputs_v_to_h(rbms[l-1], x[l-1]) + inputs_h_to_v(rbms[l], x[l+1])
end
@test I[end] ≈ inputs_v_to_h(rbms[end], x[end-1])

@inferred inputs_even_to_odd(dbm::DBM, x::Tuple)
@inferred inputs_odd_to_even(dbm::DBM, x::Tuple)

x1 = sample_even_from_odd(dbm, x)
@test size.(x1) == size.(x)
x1 = sample_odd_from_even(dbm, x)
@test size.(x1) == size.(x)

@inferred sample_even_from_odd(dbm, x)
@inferred sample_odd_from_even(dbm, x)
