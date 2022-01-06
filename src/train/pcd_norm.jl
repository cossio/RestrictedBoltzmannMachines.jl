"""
    pcd_norm!(rbm, data)

Trains the RBM on data, using the weight normalization heuristic.
See Salimans & Kingma 2016,
<https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html>.
"""
function pcd_norm!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = ADAM(), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts::Wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., _nobs(data))
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # v, g notation from Salimans & Kingma 2016
    wg = sqrt.(sum(abs2, rbm.w; dims=1:ndims(rbm.visible))) # weight norms
    wv = rbm.w ./ wg # normalized directions
    vnorm = sqrt.(sum(abs2, wv; dims=1:ndims(rbm.visible)))

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)
            # estimate gradients
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd)
            # update visible and hidden layers as usual
            update!(optimizer, rbm.visible, ∂.visible)
            update!(optimizer, rbm.hidden, ∂.hidden)

            #= Only the update of the weights is affected by the normalization
            scheme of Salimans & Kingma 2016. We first update v, g following
            Eq.(3) of the paper. =#
            ∂g = sum(∂.w .* wv ./ vnorm; dims=1:ndims(rbm.visible))
            ∂v = ∂.w .* wg ./ vnorm - ∂g .* wg .* wv ./ vnorm.^2

            Flux.update!(optimizer, rbm.w, ∂.w)
            update!(optimizer, wg, ∂g)
            update!(optimizer, wv, ∂v)

            # Update the RBM weights using the parameters v, g
            vnorm .= sqrt.(sum(abs2, wv; dims=1:ndims(rbm.visible)))
            rbm.w .= wg .* wv ./ vnorm

            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = batch_mean(log_pseudolikelihood(rbm, data), wts)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end
    return history
end
