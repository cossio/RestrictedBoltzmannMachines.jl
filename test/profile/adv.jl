using Flux, Zygote, Statistics, Test, Random, LinearAlgebra
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: first2, Data, OADAM

N = 100
q = 21
M = 200
nobs = 10000
batchsize = 64

function prepare_data(::Type{T}) where {T}
    v = T.(rand(1:q, N, nobs) |> OneHot.encode)
    y = T.(rand(1:2, nobs) |> OneHot.encode)
    w = rand(T, nobs)
    labeled_data = Data((v=v, w=w, y=y); batchsize=batchsize)
    unlabel_data = Data((v=v, w=w); batchsize=batchsize)
    return labeled_data, unlabel_data
end

function advtrain(::Type{T}; iters) where {T}
    println("Running $T")
    labeled_data, unlabel_data = prepare_data(T)
    rbm = RBM(Potts{T}(q, N), dReLU{T}(M))
    init!(rbm, labeled_data.tensors.v)
    adv = build_adv(T)
    advtrain!(rbm, adv, labeled_data, unlabel_data; iters=iters)
    advtrain!(rbm, adv, labeled_data, unlabel_data; iters=iters,
        rbmopt=OADAM(0.0001, (0.5,0.99)),
        advopt=OADAM(0.0001, (0.5,0.99)))
    return rbm, adv, labeled_data
end

advfun(::Type{<:Float64}) = f64
advfun(::Type{<:Float32}) = f32
function build_adv(::Type{T}) where {T}
    f = advfun(T)
    return Chain(
        Flux.flatten,
        f(Dense(N * q, 30, relu)),
        f(Dense(30, 16, relu)),
        f(Dense(16, 2))
    )
end

rbm32, adv32, data32 = advtrain(Float32; iters=100 * 64)
rbm64, adv64, data64 = advtrain(Float64; iters=100 * 64)
using Profile
Profile.init(n=10^7, delay=0.01)
@profiler rbm32, adv32, data32 = advtrain(Float32; iters=100 * 64)
@profiler rbm64, adv64, data64 = advtrain(Float64; iters=100 * 64)
