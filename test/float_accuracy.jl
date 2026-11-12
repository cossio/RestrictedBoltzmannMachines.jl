#= Floating-point accuracy of the scalar kernels of the layers, against rigorous references.

Each kernel is evaluated in Float32 and Float64 over a sweep of its inputs, and compared
with an enclosure of the exact value computed in ball arithmetic (Arblib): an interval that
is guaranteed to contain the true real number. The floating-point inputs convert exactly, so
the reference is the mathematical function at the very inputs the kernel received. Unlike a
BigFloat evaluation, whose own cancellation errors are invisible, an enclosure carries its
error bound: `eps_error` refuses a reference that is not tight to 40 digits.

The tolerances are the accuracy the kernels deliver today, in units of `eps(T)`, with a
margin of at least 2 over the worst error of the sweep. They are far below what a wrong
formula would produce, and they record where accuracy is knowingly lost: below the threshold
from which `tnmeanvar` evaluates the continued fraction, it uses a formula that cancels, and
loses about `a^4` of the relative accuracy of the truncated-normal variance and `a^2` of the
excess mean `E[Z | Z > a] - a`. That threshold is `a = 1` in Float32 and `a = 3.91` in
Float64 (see `_tn_tail_levels` for the choice), so that the variance loses at most a factor
of 4 in Float32, and of 265 in Float64, where its relative error is then about `2e-13`.

The statistics of a dReLU unit are held to the tolerances of a ReLU however deep the wells
of its two sides: the mixture weights do not lose accuracy with the magnitude of the
cumulant generating functions of the sides.
=#

using Test: @test, @testset
using Arblib: Arblib, Arb
using SpecialFunctions: erfc
using RestrictedBoltzmannMachines: tnmeanvar, relu_cgf, relu_meanvar, drelu_cgf, drelu_meanvar,
    drelu_mean_abs, drelu_moments, binary_var, binary_std, spin_cgf, gauss_cgf

setprecision(Arb, 512)

arb(x::Real) = Arb(Float64(x)) # exact for the floating-point types tested here

# exp(x^2) erfc(x). For x > 1 through the Tricomi function, erfcx(x) = U(1/2, 1/2, x^2) / √π,
# which stays tight where exp(x^2) and erfc(x) leave the range of the exponents.
function erfcx_ref(x::Arb)
    x > 1 || return exp(x^2) * erfc(x)
    u = Arb()
    Arblib.hypgeom_u!(u, Arb(1 // 2), Arb(1 // 2), x^2)
    return u / sqrt(Arb(π))
end

# mean and variance of the standard normal distribution truncated to (a, ∞)
function tn_ref(a::Arb)
    μ = sqrt(2 / Arb(π)) / erfcx_ref(a / sqrt(Arb(2)))
    return μ, 1 - (μ - a) * μ
end

# log ∫ exp(θh - |γ|h²/2) dh over h ≥ 0
function relu_cgf_ref(θ::Arb, γ::Arb)
    g = abs(γ)
    return log(sqrt(Arb(π) / 2g) * erfcx_ref(-θ / sqrt(2g)))
end

function relu_meanvar_ref(θ::Arb, γ::Arb)
    σ = 1 / sqrt(abs(γ))
    a = -θ * σ
    μ, ν = tn_ref(a)
    return σ * (μ - a), σ^2 * ν
end

# A dReLU unit is a mixture of a ReLU on h ≥ 0 and a mirrored ReLU on h < 0.
function drelu_ref(θp::Arb, θn::Arb, γp::Arb, γn::Arb)
    Γp = relu_cgf_ref(θp, γp)
    Γn = relu_cgf_ref(-θn, γn)
    pp = 1 / (1 + exp(Γn - Γp))
    pn = 1 / (1 + exp(Γp - Γn))
    μp, νp = relu_meanvar_ref(θp, γp)
    μn, νn = relu_meanvar_ref(-θn, γn)
    return (;
        Γp, Γn, cgf = Γp + log(1 + exp(Γn - Γp)),
        mean = pp * μp - pn * μn,
        # within-side plus between-side variance, free of cancellation
        var = pp * νp + pn * νn + pp * pn * (μp + μn)^2,
        mean_abs = pp * μp + pn * μn,
        moments = (pp * μp, -pn * μn, pp * (νp + μp^2), pn * (νn + μn^2)),
    )
end

# Error of `x` with respect to the enclosure `ref`, in units of `eps(T) * scale`.
function eps_error(x::T, ref::Arb, scale::Arb = abs(ref)) where {T <: AbstractFloat}
    @assert Arblib.radius(ref) < scale * 1.0e-40 "the reference enclosure $ref is not tight"
    isfinite(x) || return Inf
    Float64(scale) < floatmin(T) && return 0.0 # below the normal range of T
    return Float64(abs(arb(x) - ref) / scale) / eps(T)
end

# Records the inputs at which an error exceeds its tolerance (both in units of eps), so
# that a failing `@test isempty(violations)` shows where the kernel is off, and by how much.
function check!(violations::Vector, at, err::Real, tol::Real)
    err ≤ tol || push!(violations, (; at, err, tol))
    return violations
end

# `tnmeanvar` evaluates the continued fraction from this threshold on
tail_threshold(::Type{Float32}) = 1
tail_threshold(::Type{Float64}) = 3.91

# the accuracy lost to cancellation below that threshold, at threshold `a`
tail_cancellation(::Type{T}, a::Real) where {T} = a < tail_threshold(T) ? 1 + max(a, 0)^2 : 1

@testset "tnmeanvar accuracy ($T)" for T in (Float32, Float64)
    as = T[
        -40:0.0625:8; prevfloat(T(8)); nextfloat(T(8));
        8.25:0.25:60; exp10.(2:0.25:15)
    ]
    mean_violations = []
    var_violations = []
    for a in as
        μ, ν = tnmeanvar(a)
        μ_ref, ν_ref = tn_ref(arb(a))
        # for a < 0 the mean is of order exp(-a²/2), with condition number a² in `a`
        check!(mean_violations, a, eps_error(μ, μ_ref), 16 * (1 + min(a, 0)^2))
        check!(var_violations, a, eps_error(ν, ν_ref), 64 * tail_cancellation(T, a)^2)
        @test a ≤ μ
        @test 0 ≤ ν ≤ 1
    end
    @test isempty(mean_violations)
    @test isempty(var_violations)
end

@testset "ReLU kernels accuracy ($T)" for T in (Float32, Float64)
    θs = T[0; exp10.(-3:0.125:4); -exp10.(-3:0.125:4)]
    γs = T[exp10.(-3:0.5:3); -2]
    cgf_violations = []
    mean_violations = []
    var_violations = []
    for θ in θs, γ in γs
        at = (; θ, γ)
        a = -Float64(θ) / sqrt(abs(Float64(γ))) # threshold of the standardized unit
        Γ_ref = relu_cgf_ref(arb(θ), arb(γ))
        μ_ref, ν_ref = relu_meanvar_ref(arb(θ), arb(γ))
        μ, ν = relu_meanvar(θ, γ)
        # a log-partition function matters up to an absolute error
        check!(cgf_violations, at, eps_error(relu_cgf(θ, γ), Γ_ref, max(abs(Γ_ref), Arb(1))), 16)
        check!(mean_violations, at, eps_error(μ, μ_ref), 16 * tail_cancellation(T, a))
        check!(var_violations, at, eps_error(ν, ν_ref), 64 * tail_cancellation(T, a)^2)
    end
    @test isempty(cgf_violations)
    @test isempty(mean_violations)
    @test isempty(var_violations)
end

@testset "dReLU kernels accuracy ($T)" for T in (Float32, Float64)
    # The sweep includes sides with equally deep wells, whose weights are then comparable:
    # θp = -θn with γp = γn, and θ = 300 with γ = 0.1 against θ = 3000 with γ = 10, whose
    # cgfs are about 4.5e5 and differ by log(10). The deepest well has a cgf of 4.5e7.
    θs = T[0; [s * θ for θ in (0.1, 0.5, 1, 2, 3, 5, 7, 10, 30, 100, 300, 3000) for s in (1, -1)]]
    γs = T[0.1, 1, 10]
    names = (:cgf, :mean, :var, :mean_abs, :moments)
    violations = NamedTuple{names}(ntuple(_ -> [], length(names)))
    for θp in θs, θn in θs, γp in γs, γn in γs
        at = (; θp, θn, γp, γn)
        ref = drelu_ref(arb(θp), arb(θn), arb(γp), arb(γn))
        # thresholds of the standardized ReLUs of the two sides
        ap = -Float64(θp) / sqrt(Float64(γp))
        an = Float64(θn) / sqrt(Float64(γn))
        c = max(tail_cancellation(T, ap), tail_cancellation(T, an))
        μ, ν = drelu_meanvar(θp, θn, γp, γn)
        check!(violations.cgf, at, eps_error(drelu_cgf(θp, θn, γp, γn), ref.cgf, max(abs(ref.cgf), Arb(1))), 16)
        # the mean can vanish by symmetry: its natural scale is the mean absolute value
        check!(violations.mean, at, eps_error(μ, ref.mean, ref.mean_abs), 16 * c)
        check!(violations.var, at, eps_error(ν, ref.var), 64 * c^2)
        check!(violations.mean_abs, at, eps_error(drelu_mean_abs(θp, θn, γp, γn), ref.mean_abs), 16 * c)
        # the moments of each side, on the scale of the moment of that order of the unit
        m = drelu_moments(θp, θn, γp, γn)
        m_ref = ref.moments
        for i in 1:4
            scale, tol = i ≤ 2 ? (ref.mean_abs, 16 * c) : (m_ref[3] + m_ref[4], 64 * c^2)
            check!(violations.moments, (; at..., i), eps_error(m[i], m_ref[i], scale), tol)
        end
    end
    for name in names
        @test isempty(violations[name])
    end
end

@testset "Binary, Spin and Gaussian kernels accuracy ($T)" for T in (Float32, Float64)
    violations = []
    for θ in T[0; exp10.(-6:0.25:1.5); -exp10.(-6:0.25:1.5)]
        x = arb(θ)
        logistic_ref = 1 / (1 + exp(-x))
        var_ref = logistic_ref * (1 - logistic_ref)
        check!(violations, (:binary_var, θ), eps_error(binary_var(θ), var_ref), 8)
        check!(violations, (:binary_std, θ), eps_error(binary_std(θ), sqrt(var_ref)), 8)
        check!(violations, (:spin_cgf, θ), eps_error(spin_cgf(θ), log(exp(x) + exp(-x))), 8)
        for γ in T[0.01, 1, -3, 100]
            g = abs(arb(γ))
            cgf_ref = x^2 / 2g + log(2 * Arb(π) / g) / 2
            check!(violations, (:gauss_cgf, θ, γ), eps_error(gauss_cgf(θ, γ), cgf_ref, max(abs(cgf_ref), Arb(1))), 8)
        end
    end
    @test isempty(violations)
end
