function center(rbm::RBM, λv::AbstractArray, λh::AbstractArray)
    inputs_h = inputs_v_to_h(rbm, λv)
    inputs_v = inputs_h_to_v(rbm, λh)
    visible = center(rbm.visible, inputs_v)
    hidden  = center(rbm.hidden,  inputs_h)
    return RBM(visible, hidden, rbm.w)
end

center(layer::Binary, inputs::AbstractArray) = Binary(layer.θ + inputs)

function center!(rbm::RBM, λv::AbstractArray, λh::AbstractArray)
    inputs_h = inputs_v_to_h(rbm, λv)
    inputs_v = inputs_h_to_v(rbm, λh)
    center!(rbm.visible, inputs_v)
    center!(rbm.hidden,  inputs_h)
    return rbm
end

function center!(layer::Binary, inputs::AbstractArray)
    layer.θ .+= inputs
    return layer
end

uncenter(rbm::RBM, λv::AbstractArray, λh::AbstractArray) = center(rbm, -λv, -λh)
uncenter!(rbm::RBM, λv::AbstractArray, λh::AbstractArray) = center!(rbm, -λv, -λh)
uncenter(layer, inputs::AbstractArray) = center(layer, -inputs)
uncenter!(layer, inputs::AbstractArray) = center!(layer, -inputs)

"""
    centered_pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with centered gradients.
"""
function centered_pcd!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts::Wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    α::Real = 0.5,
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    hdat = mean_h_from_v(rbm, data)
    λv = batch_mean(data, wts)
    λh = batch_mean(hdat, wts)

    center!(rbm, λv, λh)
    rbm_ = uncenter(rbm, λv, λh)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm_, vm; steps = steps)
            inputs = inputs_v_to_h(rbm_, vd)
            hd = transfer_mean(rbm_.hidden, inputs)
            λh = (1 - α) * λh + α * batch_mean(hd, wd)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(ps) do
                rbm_ = uncenter(rbm, λv, λh)
                loss = contrastive_divergence(rbm_, vd, vm; wd)
                regu = lossadd(rbm_, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :cd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            Flux.update!(optimizer, ps, gs)

            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = batch_mean(log_pseudolikelihood(rbm_, data), wts)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end

    hdat = mean_h_from_v(rbm_, data)
    λv = mean_(data; dims=ndims(data))
    λh = mean_(hdat; dims=ndims(hdat))
    uncenter!(rbm, λv, λh)

    return history
end

function center_gradients!(∂::NamedTuple, rbm::RBM{<:Binary,<:Binary}, )
    @assert size(∂w) == size(rbm.w)
    @assert size(λv) == size(rbm.visible)
    @assert size(λh) == size(rbm.hidden)

end
