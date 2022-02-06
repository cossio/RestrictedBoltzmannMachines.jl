"""
    fpcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with fast weights.
See http://dl.acm.org/citation.cfm?id=1553374.1553506.
"""
function fpcd!(rbm::AbstractRBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    optimfast = Flux.ADAM(), # optimizer algorithm for fast parameters
    decayfast::Real = 19/20  # weight decay of fast parameters
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = suffstats(visible(rbm), data; wts)
    vm = transfer_sample(visible(rbm), falses(size(visible(rbm))..., batchsize))

    # store fast parameters
    rbmfast = deepcopy(rbm)
    # (Actually, the parameters of rbmfast are the sums θ_regular + θ_fast)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm = sample_v_from_v(rbmfast, vm; steps = steps)
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd, stats)
            push!(history, :∂, gradnorms(∂))
            update!(rbm, update!(∂, rbm, optim))
            # update fast parameters
            update!(rbmfast, update!(∂, rbmfast, optimfast))
            decayfast!(rbmfast, rbm; decay=decayfast)
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        Δt_ = round(Δt, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s)"
    end
    return history
end

#= Decays parameters of `rbmfast` towards those of `rbm`.
In other words, writing the parametrs of `rbmfast` as
ω_regular + ω_fast, where ω_regular are the prameters of
`rbm`, then here we decay the ω_fast part towards zero. =#
function decayfast!(rbmfast::M, rbm::M; decay::Real) where {M<:RBM}
    decayfast!(visible(rbmfast), visible(rbm); decay)
    decayfast!(hidden(rbmfast), hidden(rbm); decay)
    decayfast!(weights(rbmfast), weights(rbm); decay)
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
