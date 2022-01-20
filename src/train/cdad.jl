"""
    cdad!(rbm, data)

Trains the RBM on data using contrastive divergence.
Computes gradients with Zygote.
"""
function cdad!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    lossadd = (_...) -> 0, # regularization
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # fantasy chains
            _idx = rand(1:_nobs(data), batchsize)
            _vm = copy(selectdim(data, ndims(data), _idx))
            vm = sample_v_from_v(rbm, _vm; steps = steps)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(rbm) do rbm
                loss = contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :cd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            update!(optimizer, rbm, only(gs))

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
