struct UnbiasedSample{V}
    vhist::Vector{V}
    vchist::Vector{V}
    discarded::Int
    met::Bool
end

_binary_visible_logits(rbm::RBM{<:Binary,<:Binary}, h) = rbm.visible.θ .+ inputs_v_from_h(rbm, h)
_binary_hidden_logits(rbm::RBM{<:Binary,<:Binary}, v) = rbm.hidden.θ .+ inputs_h_from_v(rbm, v)

function _binary_sample_from_logits(logits::AbstractArray)
    u = rand!(similar(logits))
    return binary_rand.(logits, u)
end

_binary_sample_from_logits(logits::AbstractArray, u::AbstractArray) = binary_rand.(logits, u)

_binary_logprob(logits::AbstractArray, x::AbstractArray) = sum(x .* logits .- log1pexp.(logits))

_zeros_like(x::AbstractArray) = fill!(similar(x), zero(eltype(x)))

function _zero_gradient(rbm::RBM)
    return ∂RBM(_zeros_like(rbm.visible.par), _zeros_like(rbm.hidden.par), _zeros_like(rbm.w))
end

function _maximal_coupling_step(
    rbm::RBM{<:Binary,<:Binary},
    vc0::AbstractArray,
    hc0::AbstractArray,
    v1::AbstractArray,
    h1::AbstractArray;
    max_tries::Int = 10,
)
    v2_logits = _binary_visible_logits(rbm, h1)
    v2 = _binary_sample_from_logits(v2_logits)
    h2 = sample_h_from_v(rbm, v2)

    if v1 == vc0 && h1 == hc0
        return copy(v2), copy(h2), v2, h2, 0
    end

    vc1_logits = _binary_visible_logits(rbm, hc0)
    if randexp() ≥ _binary_logprob(v2_logits, v2) - _binary_logprob(vc1_logits, v2)
        return copy(v2), copy(h2), v2, h2, 0
    end

    v2 = nothing
    vc1 = nothing
    discarded = 0
    for attempt in 1:max_tries
        uv = rand!(similar(v2_logits))
        if isnothing(v2)
            candidate = _binary_sample_from_logits(v2_logits, uv)
            if attempt == max_tries || randexp() < _binary_logprob(v2_logits, candidate) - _binary_logprob(vc1_logits, candidate)
                v2 = candidate
            end
        end
        if isnothing(vc1)
            candidate = _binary_sample_from_logits(vc1_logits, uv)
            if attempt == max_tries || randexp() < _binary_logprob(vc1_logits, candidate) - _binary_logprob(v2_logits, candidate)
                vc1 = candidate
            end
        end
        if !isnothing(v2) && !isnothing(vc1)
            discarded = attempt - 1
            break
        end
    end

    h2_logits = _binary_hidden_logits(rbm, v2)
    hc1_logits = _binary_hidden_logits(rbm, vc1)
    uh = rand!(similar(h2_logits))
    h2 = _binary_sample_from_logits(h2_logits, uh)
    hc1 = _binary_sample_from_logits(hc1_logits, uh)

    return vc1, hc1, v2, h2, discarded
end

"""
    unbiased_sample(rbm, v0; min_steps = 1, max_steps = 100, max_tries = 10)

Run two coupled Gibbs chains for a binary-binary RBM, starting from the visible
configuration `v0`, and return their histories.
"""
function unbiased_sample(
    rbm::RBM{<:Binary,<:Binary},
    v0::AbstractArray;
    min_steps::Int = 1,
    max_steps::Int = 100,
    max_tries::Int = 10,
)
    @assert size(v0) == size(rbm.visible)
    @assert min_steps > 0
    @assert max_steps ≥ min_steps
    @assert max_tries > 0

    vc = copy(v0)
    hc = sample_h_from_v(rbm, vc)
    v = sample_v_from_h(rbm, hc)
    h = sample_h_from_v(rbm, v)

    vhist = [v]
    vchist = typeof(v)[]
    discarded = 0
    met = false

    for step in 1:max_steps
        vc, hc, v, h, disc = _maximal_coupling_step(rbm, vc, hc, v, h; max_tries)
        discarded += disc
        push!(vhist, v)
        push!(vchist, vc)
        if step ≥ min_steps && v == vc && h == hc
            met = true
            break
        end
    end

    return UnbiasedSample(vhist, vchist, discarded, met)
end

"""
    unbiased_estimator(f, sample; burnin = 1)

Form the unbiased MCMC estimator associated with a coupled sample history.
"""
function unbiased_estimator(f, sample::UnbiasedSample; burnin::Int = 1)
    @assert 1 ≤ burnin ≤ length(sample.vhist)
    estimate = f(sample.vhist[burnin])
    for t in (burnin + 1):length(sample.vhist)
        estimate += f(sample.vhist[t]) - f(sample.vchist[t - 1])
    end
    return estimate
end

"""
    ucd!(rbm, data)

Train a binary-binary RBM on data using Unbiased Contrastive Divergence.
"""
function ucd!(
    rbm::RBM{<:Binary,<:Binary},
    data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1,
    wts::Union{AbstractVector, Nothing} = nothing,
    nchains::Int = batchsize,
    min_steps::Int = 1,
    max_steps::Int = 100,
    max_tries::Int = 10,
    optim::AbstractRule = Adam(),
    moments = moments_from_samples(rbm.visible, data; wts),
    l2_fields::Real = 0,
    l1_weights::Real = 0,
    l2_weights::Real = 0,
    l2l1_weights::Real = 0,
    zerosum::Bool = true,
    rescale::Bool = true,
    callback = Returns(nothing),
    shuffle::Bool = true,
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps),
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)
    @assert nchains > 0

    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    wts_mean = isnothing(wts) ? 1 : mean(wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)

        ∂m = _zero_gradient(rbm)
        meeting_steps = 0
        discarded = 0
        nbatch = size(vd, ndims(vd))
        for chain in 1:nchains
            v0 = copy(vd[.., mod1(chain, nbatch)])
            sample = unbiased_sample(rbm, v0; min_steps, max_steps, max_tries)
            ∂m += unbiased_estimator(v -> ∂free_energy(rbm, v), sample; burnin = min_steps)
            meeting_steps += length(sample.vchist)
            discarded += sample.discarded
        end
        ∂m /= nchains

        ∂ = ∂d - ∂m

        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, state, iter, vd, wd, meeting_steps = meeting_steps / nchains, discarded = discarded / nchains)
    end
    return state, ps
end
