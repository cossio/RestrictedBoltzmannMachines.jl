struct WeightNorm{Tg<:AbstractArray, Tv<:AbstractArray}
    g::Tg
    v::Tv
    """
        WeightNorm(g, v)

    Implements the the weight normalization heuristic for training neural networks.
    See Salimans & Kingma 2016,
    <https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html>.
    """
    function WeightNorm(g::AbstractArray, v::AbstractArray)
        @assert all((size(g) .== size(v)) .| (size(g) .== 1))
        return new{typeof(g), typeof(v)}(g, v)
    end
end

@doc raw"""
        WeightNorm(rbm, λ = 1)

Returns a re-parameterization of `rbm.w`, defined by:

```math
\mathbf{w}_\mu = g_\mu^2 \frac{\mathbf{v}_\mu}{\|\mathbf{v}_\mu\|}
```

where the parameter ``g_\mu^2`` encodes the norm of the weight pattern attached to
hidden unit ``\mu``, while ``\mathbf{v}_\mu`` encodes its direction.

The constructor `WeightNorm(rbm, λ)` initializes ``g_\mu^2`` and ``\mathbf{v}_\mu``
assuming that the vectors ``\mathbf{v}_\mu`` have norms `λ`.
More precisely, we assume that:

```julia
λ = sqrt.(sum(abs2, v; dims=1:ndims(rbm.visible)))
```

If not provided `λ` defaults to ones.
"""
function WeightNorm(rbm::RBM, λ::AbstractArray)
    @assert size(λ) == (ntuple(_ -> 1, ndims(rbm.visible))..., size(rbm.hidden)...)
    g = weight_norms(rbm)
    v = λ .* rbm.w ./ g
    return WeightNorm(g, v)
end

function WeightNorm(rbm::RBM, λ::Real = 1)
    sz = (ntuple(_ -> 1, ndims(rbm.visible))..., size(rbm.hidden)...)
    λs = FillArrays.Fill(λ, sz)
    return WeightNorm(rbm, λs)
end

"""
    weight_norms(rbm)

Norms of weight patterns attached to each hidden unit.
"""
weight_norms(rbm::RBM) = sqrt.(sum(abs2, rbm.w; dims=1:ndims(rbm.visible)))

@doc raw"""
    ∂wnorm(∂w, g, v)

Given the gradients `∂w` of a function `f(w)` with respect to `w`,
returns the gradients `∂g, ∂v` of `f` with respect to the re-parameterization:

```math
\mathbf{w} = g \frac{\mathbf{v}}{\|\mathbf{v}\|}
```

See Salimans & Kingma 2016,
<https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html>.
"""
function ∂wnorm(∂w::AbstractArray, g::AbstractArray, v::AbstractArray)
    @assert size(∂w) == size(v)
    # see Eqs. (3) and (4) of Salimans & Kingma 2016
    vn = sqrt.(sum!(similar(g), abs2.(v)))
    ∂g = sum!(similar(g), ∂w .* v) ./ vn
    ∂v = (∂w - ∂g .* v ./ vn) .* g ./ vn
    return (g = ∂g, v = ∂v)
end

function w_from_gv(g::AbstractArray, v::AbstractArray)
    vn = sqrt.(sum!(similar(g), abs2.(v)))
    return g .* v ./ vn
end

function ∂contrastive_divergence(
    rbm::RBM, wn::WeightNorm, args...; kwargs...
)
    ∂ = ∂contrastive_divergence(rbm, args...; kwargs...)
    ∂wn = ∂wnorm(∂.w, wn.g, wn.v)
    return (visible = ∂.visible, hidden = ∂.hidden, g = ∂wn.g, v = ∂wn.v)
end

function update!(optimizer, rbm::RBM, wn::WeightNorm, ∂::NamedTuple)
    update!(optimizer, rbm.visible, ∂.visible)
    update!(optimizer, rbm.hidden, ∂.hidden)
    update!(optimizer, wn.g, ∂.g)
    update!(optimizer, wn.v, ∂.v)
    rbm.w .= w_from_gv(wn.g, wn.v)
    return rbm
end

"""
    pcd!(rbm, wn::WeightNorm, data)

Trains the RBM on data, using the weight normalization heuristic.
See Salimans & Kingma 2016,
<https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html>.
"""
function pcd!(rbm::RBM, wn::WeightNorm, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(), # stores training log
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = sufficient_statistics(rbm.visible, data; wts)

    # initialize fantasy chains by sampling visible layer
    vm = transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)
            # compute contrastive divergence gradient
            ∂ = ∂contrastive_divergence(rbm, wn, vd, vm; wd = wd, stats)
            # update parameters using gradient
            update!(optimizer, rbm, wn, ∂)
        end

        lpl = wmean(log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vn, maximum(abs, wn.v))

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end
    return history
end
