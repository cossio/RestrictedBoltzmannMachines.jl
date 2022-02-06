"""
    cdad!(rbm, data)

Trains the RBM on data using contrastive divergence.
Computes gradients with Zygote.
"""
function cdad!(rbm::AbstractRBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            vm = sample_v_from_v(rbm, vd; steps = steps)
            gs = Zygote.gradient(rbm) do rbm
                loss = contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :cd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end
            update!(rbm, update!(only(gs), rbm, optim))
            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = wmean(log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"

    end
    return history
end
