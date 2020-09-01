export CD, PCD, train!, contrastive_divergence, mean_free_energy

# Throw this from a callback to force an early stop of training
# (or just call stop())
struct EarlyStop <: Exception end
stop() = throw(EarlyStop())

# Contrastive Divergence
@kwdef struct CD
    steps::Int = 1 # update steps of the MC chains per iteration
end

# Persistent Contrastive Divergence
@kwdef struct PCD
    steps::Int = 1 # update steps of the MC chains per iteration
end

"""
    train!(rbm, data, [cd])

We measure training time in units of observations presented to the model.
For example, if you want to train for 100 epochs, then set `iters = 100 * data.nobs`,
where `data.nobs` is the number of observations in the dataset.
"""
function train!(rbm::RBM, data::Data, cd::Union{CD,PCD} = PCD();
                iters::Int, opt = ADAM(), ps::Params = params(rbm),
                vm::AbstractArray = update_chains(rbm, cd, first(data).v), # Markov chains
                history = nothing, # stores training history
                callback = () -> (),
                tests_data::Data = Data(),
                reg = no_regularization, λw::Real = 0, λh::Real = 0, λg::Real = 0 # regularization
            )
    checkdims(rbm.vis, vm)
    progress_bar = Progress(length(1:data.batchsize:iters))
    for (iter, datum, tests_datum) in zip(1:data.batchsize:iters, data, tests_data)
        # update model samples
        vm = update_chains(rbm, cd, datum.v, vm)
        # train RBM
        gs = gradient(ps) do
            loss = contrastive_divergence(rbm, datum.v, vm, datum.w)
            rbm_reg = reg(rbm)
            hl1 = hidden_l1(rbm, datum)
            wl1l2 = weights_l1l2(rbm)
            gl2 = fields_l2(rbm.vis)
            if !isnothing(history)
                ignore() do
                    push!(history, :loss, iter, loss)
                    push!(history, :reg, iter, rbm_reg)
                    push!(history, :wl1l2, iter, wl1l2)
                    push!(history, :hl1, iter, hl1)
                    push!(history, :gl2, iter, gl2)
                end
            end
            loss + rbm_reg + λh * hl1 + λg * gl2 + λw/2 * wl1l2
        end
        isnothing(history) || push!(history, :grad_norm, iter, [norm(gs[p]) for p in ps])
        
        #= record log_pseudolikelihood =#
        if isnothing(history) && (iter % 50data.batchsize == 0)
            lpl_train = log_pseudolikelihood_rand(rbm, datum.v, 1, data.tensors.w)
            lpl_tests = log_pseudolikelihood_rand(rbm, tests_datum.v, 1, tests_datum.w)
            push!(history, :lpltrain, iter, lpl_train)
            push!(history, :lpltests, iter, lpl_tests)
        end

        # update RBM parameters
        Flux.update!(opt, ps, gs)

        # callback
        try
            callback()
        catch ex
            if ex isa EarlyStop
                break
            else
                rethrow(ex)
            end
        end
        
        # update progress bar
        next!(progress_bar)
    end
    return nothing
end

update_chains(rbm::RBM, cd::CD,  vd::AbstractArray, vm::AbstractArray = vd, β = 1) =
    sample_v_from_v(rbm, vd, β; steps = cd.steps)::typeof(vd)

update_chains(rbm::RBM, cd::PCD, vd::AbstractArray, vm::AbstractArray = vd, β = 1) =
    sample_v_from_v(rbm, vm, β; steps = cd.steps)::typeof(vm)

"""
    contrastive_divergence(rbm, vd, vm, wd = 1, wm = 1)

Contrastive divergence, defined as free energy difference between data (vd) and
model sample (vm). The (optional) `wd,wm` are weights for the batches.
"""
function contrastive_divergence(rbm::RBM, vd::AbstractArray, vm::AbstractArray, wd = 1, wm = 1)
    Fd = mean_free_energy(rbm, vd, wd)
    Fm = mean_free_energy(rbm, vm, wm)
    return (Fd - Fm) / length(rbm.vis)
end

"""
    mean_free_energy(rbm, v, w = 1)

Mean free energy across batches. The optional `w` specifies weights for the
batches.
"""
mean_free_energy(rbm::RBM, v::AbstractArray, w = 1) = wmean(free_energy(rbm, v), w)
