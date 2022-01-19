"""
    fpcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with fast weights.
See http://dl.acm.org/citation.cfm?id=1553374.1553506.
"""
function fpcd!(rbm::RBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    optimfast = Flux.ADAM(), # optimizer algorithm for fast weights
    decayfast::Real = 19/20  # weight decay of fast parameters
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = sufficient_statistics(rbm.visible, data; wts)

    # initialize fantasy chains by sampling visible layer
    vm = transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))

    # store fast weights
    rbmfast = deepcopy(rbm)
    # (Actually, the parameters of rbmfast are the sums θ_regular + θ_fast)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm = sample_v_from_v(rbmfast, vm; steps = steps)
            # compute contrastive divergence gradient
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd, stats)
            # update parameters using gradient
            update!(optimizer, rbm, ∂)
            update!(optimfast, rbmfast, ∂)
            # decays parameters of rbmfast towards those of rbm
            decayfast!(rbmfast, rbm; decay=decayfast)
            # store gradient norms
            push!(history, :∂, gradnorms(∂))
        end

        lpl = wmean(log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_")
        end
    end
    return history
end

#= Decays parameters of `rbmfast` towards those of `rbm`.
In other words, writing the parametrs of `rbmfast` as
ω_regular + ω_fast, where ω_regular are the prameters of
`rbm`, then here we decay the ω_fast part towards zero. =#
function decayfast!(rbmfast::M, rbm::M; decay::Real) where {M<:RBM}
    decayfast!(rbmfast.visible, rbm.visible; decay)
    decayfast!(rbmfast.hidden, rbm.hidden; decay)
    decayfast!(rbmfast.w, rbm.w; decay)
end

function decayfast!(fast::L, regular::L; decay::Real) where {L<:Union{Binary,Spin,Potts}}
    @assert size(fast) == size(regular)
    decayfast!(fast.θ, regular.θ; decay)
end

function decayfast!(fast::L, regular::L; decay::Real) where {L<:Union{Gaussian,ReLU}}
    @assert size(fast) == size(regular)
    decayfast!(fast.θ, regular.θ; decay)
    decayfast!(fast.γ, regular.γ; decay)
end

function decayfast!(fast::L, regular::L; decay::Real) where {L<:dReLU}
    @assert size(fast) == size(regular)
    decayfast!(fast.θp, regular.θp; decay)
    decayfast!(fast.θn, regular.θn; decay)
    decayfast!(fast.γp, regular.γp; decay)
    decayfast!(fast.γn, regular.γn; decay)
end

function decayfast!(ωfast::AbstractArray, ωregular::AbstractArray; decay::Real)
    @assert size(ωfast) == size(ωregular)
    ωfast .= decay .* ωfast + (1 - decay) .* ωregular
end
