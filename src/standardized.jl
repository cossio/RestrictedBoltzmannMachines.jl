struct StandardizedRBM{V,H,W,Ov,Oh,Sv,Sh}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    scale_v::Sv
    scale_h::Sh
    function StandardizedRBM(
        visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray,
        scale_v::AbstractArray, scale_h::AbstractArray
    )
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(visible) == size(offset_v) == size(scale_v)
        @assert size(hidden)  == size(offset_h) == size(scale_h)
        V, H, W = typeof(visible), typeof(hidden), typeof(w)
        Ov, Oh, Sv, Sh = typeof(offset_v), typeof(offset_h), typeof(scale_v), typeof(scale_h)
        return new{V,H,W,Ov,Oh,Sv,Sh}(visible, hidden, w, offset_v, offset_h, scale_v, scale_h)
    end
end

function StandardizedRBM(
    rbm::RBM,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    StandardizedRBM(rbm.visible, rbm.hidden, rbm.w, offset_v, offset_h, scale_v, scale_h)
end

function StandardizedRBM(rbm::RBM)
    offset_v = (similar(rbm.w, size(rbm.visible)) .= 0)
    offset_h = (similar(rbm.w, size(rbm.hidden)) .= 0)
    scale_v = (similar(rbm.w, size(rbm.visible)) .= 1)
    scale_h = (similar(rbm.w, size(rbm.hidden)) .= 1)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

RBM(rbm::StandardizedRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

standardize_v(rbm::StandardizedRBM, v::AbstractArray) = (v .- rbm.offset_v) ./ rbm.scale_v
standardize_h(rbm::StandardizedRBM, h::AbstractArray) = (h .- rbm.offset_h) ./ rbm.scale_h

function interaction_energy(rbm::StandardizedRBM, v::AbstractArray, h::AbstractArray)
    std_v = standardize_v(rbm, v)
    std_h = standardize_h(rbm, h)
    return interaction_energy(RBM(rbm), std_v, std_h)
end

function inputs_h_from_v(rbm::StandardizedRBM, v::AbstractArray)
    std_v = standardize_v(rbm, v)
    inputs = inputs_h_from_v(RBM(rbm), std_v)
    return inputs ./ rbm.scale_h
end

function inputs_v_from_h(rbm::StandardizedRBM, h::AbstractArray)
    scaled_h = standardize_h(rbm, h)
    inputs = inputs_v_from_h(RBM(rbm), scaled_h)
    return inputs ./ rbm.scale_v
end

function mirror(rbm::StandardizedRBM)
    _rbm = mirror(RBM(rbm))
    return StandardizedRBM(_rbm, rbm.offset_h, rbm.offset_v, rbm.scale_h, rbm.scale_v)
end

function free_energy(rbm::StandardizedRBM, v::AbstractArray)
    E = energy(rbm.visible, v)
    inputs = inputs_h_from_v(rbm, v)
    F = -cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_h), inputs)
    return E + F - ΔE
end

function free_energy_h(rbm::StandardizedRBM, h::AbstractArray)
    E = energy(rbm.hidden, h)
    inputs = inputs_v_from_h(rbm, h)
    F = -cgf(rbm.visible, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_v), inputs)
    return E + F - ΔE
end

function ∂free_energy(
    rbm::StandardizedRBM, v::AbstractArray;
    wts = nothing, moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    ∂v = ∂energy_from_moments(rbm.visible, moments)
    ∂Γ = ∂cgfs(rbm.hidden, inputs)
    h = grad2ave(rbm.hidden, ∂Γ)

    ∂h = reshape(wmean(-∂Γ; wts, dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)), size(rbm.hidden.par))
    ∂w = ∂interaction_energy(rbm, v, h; wts)

    return ∂RBM(∂v, ∂h, ∂w)
end

function ∂free_energy_v(rbm::StandardizedRBM, v::AbstractArray; kwargs...)
    return ∂free_energy(rbm, v; kwargs...)
end

function ∂free_energy_h(
    rbm::StandardizedRBM, h::AbstractArray;
    wts = nothing, moments = moments_from_samples(rbm.hidden, h; wts)
)
    inputs = inputs_v_from_h(rbm, h)
    ∂h = ∂energy_from_moments(rbm.hidden, moments)
    ∂Γ = ∂cgfs(rbm.visible, inputs)
    v = grad2ave(rbm.visible, ∂Γ)

    ∂v = reshape(wmean(-∂Γ; wts, dims = (ndims(rbm.visible.par) + 1):ndims(∂Γ)), size(rbm.visible.par))
    ∂w = ∂interaction_energy(rbm, v, h; wts)

    return ∂RBM(∂v, ∂h, ∂w)
end


function ∂interaction_energy(
    rbm::StandardizedRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    std_v = standardize_v(rbm, v)
    std_h = standardize_h(rbm, h)
    ∂w = ∂interaction_energy(RBM(rbm), std_v, std_h; wts)
    return ∂w
end

function log_pseudolikelihood(rbm::StandardizedRBM, v::AbstractArray)
    return log_pseudolikelihood(unstandardize(rbm), v)
end

# function ∂regularize!(∂::∂RBM, rbm::StandardizedRBM; kwargs...)
#     ∂regularize!(∂, RBM(rbm); kwargs...)
# end

function ∂regularize!(
    ∂::∂RBM, rbm::StandardizedRBM;
    l2_fields::Real = 0,
    l1_weights::Real = 0,
    l2_weights::Real = 0,
    l2l1_weights::Real = 0,
)
    urbm = unstandardize(rbm)

    offset_h = reshape(rbm.offset_h, map(one, size(rbm.scale_v))..., size(rbm.scale_h)...)
    scale_v = reshape(rbm.scale_v, size(rbm.scale_v)..., map(one, size(rbm.scale_h))...)
    scale_h = reshape(rbm.scale_h, map(one, size(rbm.scale_v))..., size(rbm.scale_h)...)
    scale_w = scale_v .* scale_h

    if !iszero(l2_fields)
        visible_reg = ∂regularize_fields(urbm.visible; l2_fields)
        ∂.visible .+= visible_reg
        ∂regularize_add_visible_offset!(∂, visible_reg, offset_h, scale_w, rbm.visible)
    end
    if !iszero(l1_weights)
        ∂.w .+= l1_weights * sign.(urbm.w) ./ scale_w
    end
    if !iszero(l2_weights)
        ∂.w .+= l2_weights * urbm.w ./ scale_w
    end
    if !iszero(l2l1_weights)
        dims = ntuple(identity, ndims(rbm.visible))
        ∂.w .+= l2l1_weights * sign.(urbm.w) .* mean(abs, urbm.w; dims) ./ scale_w
    end
end

function ∂regularize_add_visible_offset!(∂::∂RBM, visible_regularization::AbstractArray, offset_h::AbstractArray, scale_w::AbstractArray, ::dReLU)
    ∂.w .-= (visible_regularization[1, ..] + visible_regularization[2, ..]) .* offset_h ./ scale_w
end

function ∂regularize_add_visible_offset!(∂::∂RBM, visible_regularization::AbstractArray, offset_h::AbstractArray, scale_w::AbstractArray, ::Union{Binary,Spin,Potts,Gaussian,ReLU,xReLU,pReLU})
    ∂.w .-= visible_regularization[1, ..] .* offset_h ./ scale_w
end

function rescale_hidden_activations!(rbm::StandardizedRBM)
    if rescale_activations!(rbm.hidden, rbm.scale_h)
        rbm.offset_h ./= rbm.scale_h
        rbm.scale_h ./= rbm.scale_h
        return true
    end
    return false
end

function zerosum!(rbm::StandardizedRBM)
    zerosum!(RBM(rbm))
    return rbm
end

"""
    delta_energy(rbm)

Compute the (constant) energy shift with respect to the equivalent normal RBM.
"""
delta_energy(rbm::RBM) = 0

function delta_energy(rbm::StandardizedRBM)
    return interaction_energy(rbm, zero(rbm.offset_v), zero(rbm.offset_h))
end

unstandardize(rbm::StandardizedRBM) = RBM(standardize(rbm))
unstandardize(rbm::RBM) = rbm

standardize(rbm::StandardizedRBM) = standardize(rbm, zero.(rbm.offset_v), zero.(rbm.offset_h), one.(rbm.scale_v), one.(rbm.scale_h))
standardize(rbm::RBM) = StandardizedRBM(rbm)

function standardize(
    rbm::StandardizedRBM,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    std_rbm = standardize_visible(rbm, offset_v, scale_v)
    return standardize_hidden(std_rbm, offset_h, scale_h)
end

standardize(
    rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray
) = standardize(standardize(rbm), offset_v, offset_h, scale_v, scale_h)

function standardize_visible(std_rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(std_rbm.visible) == size(offset_v) == size(scale_v)

    cv = reshape(scale_v ./ std_rbm.scale_v, size(std_rbm.visible)..., map(one, size(std_rbm.hidden))...)
    Δθ = inputs_h_from_v(std_rbm, offset_v)

    hid = shift_fields(std_rbm.hidden, Δθ)
    w = std_rbm.w .* cv
    rbm = RBM(std_rbm.visible, hid, w)

    return StandardizedRBM(rbm, offset_v, std_rbm.offset_h, scale_v, std_rbm.scale_h)
end

function standardize_hidden(std_rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(std_rbm.hidden) == size(offset_h) == size(scale_h)

    ch = reshape(scale_h ./ std_rbm.scale_h, map(one, size(std_rbm.visible))..., size(std_rbm.hidden)...)
    Δθ = inputs_v_from_h(std_rbm, offset_h)

    vis = shift_fields(std_rbm.visible, Δθ)
    w = std_rbm.w .* ch
    rbm = RBM(vis, std_rbm.hidden, w)

    return StandardizedRBM(rbm, std_rbm.offset_v, offset_h, std_rbm.scale_v, scale_h)
end

standardize_visible(rbm::RBM, offset_v::AbstractArray, scale_v::AbstractArray) = standardize_visible(standardize(rbm), offset_v, scale_v)
standardize_hidden(rbm::RBM, offset_h::AbstractArray, scale_h::AbstractArray) = standardize_hidden(standardize(rbm), offset_h, scale_h)

standardize_visible(rbm::StandardizedRBM) = standardize_visible(rbm, zero(rbm.offset_v), ones.(rbm.scale_v))
standardize_hidden(rbm::StandardizedRBM) = standardize_hidden(rbm, zero(rbm.offset_h), ones.(rbm.scale_h))
standardize_visible(rbm::RBM) = standardize(rbm)
standardize_hidden(rbm::RBM) = standardize(rbm)

function standardize!(rbm::StandardizedRBM, offset_v::AbstractArray, offset_h::AbstractArray, scale_v::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)
    standardize_visible!(rbm, offset_v, scale_v)
    standardize_hidden!(rbm, offset_h, scale_h)
    return rbm
end

function standardize_visible!(rbm::StandardizedRBM, offset_v::AbstractArray, scale_v::AbstractArray)
    @assert size(rbm.visible) == size(offset_v) == size(scale_v)

    cv = reshape(scale_v ./ rbm.scale_v, size(rbm.visible)..., map(one, size(rbm.hidden))...)
    Δθ = inputs_h_from_v(rbm, offset_v)

    shift_fields!(rbm.hidden, Δθ)
    rbm.w .= rbm.w .* cv
    rbm.offset_v .= offset_v
    rbm.scale_v .= scale_v

    return rbm
end

function standardize_hidden!(rbm::StandardizedRBM, offset_h::AbstractArray, scale_h::AbstractArray)
    @assert size(rbm.hidden) == size(offset_h) == size(scale_h)

    ch = reshape(scale_h ./ rbm.scale_h, map(one, size(rbm.visible))..., size(rbm.hidden)...)
    Δθ = inputs_v_from_h(rbm, offset_h)

    shift_fields!(rbm.visible, Δθ)
    rbm.w .= rbm.w .* ch
    rbm.offset_h .= offset_h
    rbm.scale_h .= scale_h

    return rbm
end

function standardize_visible_from_data!(rbm::StandardizedRBM, data::AbstractArray; wts = nothing, ϵ::Real = 0)
    μ = batchmean(rbm.visible, data; wts)
    ν = batchvar(rbm.visible, data; wts, mean=μ)
    return standardize_visible!(rbm, μ, sqrt.(ν .+ ϵ))
end

function standardize_hidden_from_inputs!(rbm::StandardizedRBM, inputs::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    μ, ν = hidden_statistics_from_inputs(rbm.hidden, inputs; wts)
    offset_h = (1 - damping) .* rbm.offset_h + damping .* μ
    scale_h = sqrt.((1 - damping) .* rbm.scale_h.^2 + damping .* (ν .+ ϵ))
    return standardize_hidden!(rbm, offset_h, scale_h)
end

function standardize_hidden_from_v!(rbm::StandardizedRBM, v::AbstractArray; wts = nothing, damping::Real = 0, ϵ::Real = 0)
    inputs = inputs_h_from_v(rbm, v)
    standardize_hidden_from_inputs!(rbm, inputs; damping, wts, ϵ)
end

function hidden_statistics_from_inputs(layer::AbstractLayer, inputs::AbstractArray; wts = nothing)
    h_ave = mean_from_inputs(layer, inputs)
    h_var = var_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts)
    ν_ext = batchvar(layer, h_ave; wts, mean = μ)
    ν = ν_int + ν_ext # law of total variance
    return (; μ, ν)
end

function potts_to_gumbel(rbm::StandardizedRBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

function gumbel_to_potts(rbm::StandardizedRBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

function pcd!(
    rbm::StandardizedRBM,
    data::AbstractArray;

    batchsize::Int = 1,
    shuffle::Bool = true,

    iters::Int = 1, # number of gradient updates

    steps::Int = 1,
    vm::AbstractArray = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),

    moments = moments_from_samples(rbm.visible, data), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # "pseudocount" for estimating variances of v and h and damping
    damping::Real = 1//100,
    ϵv::Real = 0, ϵh::Real = 0,

    # optimiser
    optim::AbstractRule = Adam(),
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps),

    # Absorb the scale_h into the hidden unit activation (for continuous hidden units).
    # Results in hidden units with var(h) ~ 1.
    rescale_hidden::Bool = true,

    zerosum::Bool = true, # zerosum gauge for Potts layers

    # called for every gradient update
    callback = Returns(nothing)
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert 0 ≤ damping ≤ 1

    standardize_visible_from_data!(rbm, data; ϵ = ϵv)
    zerosum && zerosum!(rbm)

    for (iter, (vd,)) in zip(1:iters, infinite_minibatches(data; batchsize, shuffle))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # update standardization
        standardize_hidden_from_v!(rbm, vd; damping, ϵ=ϵh)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        rescale_hidden && rescale_hidden_activations!(rbm)
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, state, ps, iter, vm, vd, ∂)
    end

    return state, ps
end

function BinaryStandardizedRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    rbm = BinaryRBM(a, b, w)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

function BinaryStandardizedRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = BinaryRBM(a, b, w)
    return standardize(rbm)
end

function SpinStandardizedRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray,
    scale_v::AbstractArray, scale_h::AbstractArray
)
    rbm = SpinRBM(a, b, w)
    return StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)
end

function SpinStandardizedRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = SpinRBM(a, b, w)
    return standardize(rbm)
end

function log_partition(rbm::StandardizedRBM)
    v = ChainRulesCore.ignore_derivatives() do
        collect_states(rbm.visible)
    end
    return logsumexp(-free_energy(rbm, v))
end
