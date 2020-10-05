export CD, PCD, train!,
    contrastive_divergence_v, contrastive_divergence_h,
    mean_free_energy_v, mean_free_energy_h

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
    train!(rbm, data)

We measure training time in units of observations presented to the model.
For example, if you want to train for 100 epochs, then set `iters = 100 * data.nobs`,
where `data.nobs` is the number of observations in the dataset.
"""
function train!(rbm::RBM, data::Data; cd::Union{CD,PCD} = PCD(),
                iters::Int, # number of iterations (number of examples seen during training)
                opt = ADAM(), # optimizer algorithm
                ps::Params = params(rbm), # subset of optimized parameters
                vm::NumArray = update_chains_v(rbm, cd, first(data).v), # Markov chains
                history = nothing, # stores training history
                callback = () -> (), # callback function called on each iteration
                tests_data::Data = data, # validation dataset
                reg = no_regularization, λw::Real = 0, λh::Real = 0, λg::Real = 0, # regularization
                min_lpl = -Inf, # minimum log-pseudolikelihood
                lpl_interval = 50data.batchsize, # iterations to wait before computing log-pseudolikelihood
                print_interval = 200data.batchsize # iterations to wait before printing log-pseudolikelihood
            )
    checkdims(rbm.vis, vm)
    progress_bar = Progress(length(1:data.batchsize:iters))
    for (iter, datum, tests_datum) in zip(1:data.batchsize:iters, data, tests_data)
        # update model samples
        vm = update_chains_v(rbm, cd, datum.v, vm)
        # train RBM
        gs = gradient(ps) do
            loss = contrastive_divergence_v(rbm, datum.v, vm, datum.w)
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
        isnothing(history) || push!(history, :grad_norm, iter, IdDict(x => norm(gs[x]) for x in ps))
        
        #= record log_pseudolikelihood =#
        if !isnothing(history) && (iter % lpl_interval < data.batchsize)
            lpl_train = log_pseudolikelihood_rand(rbm, datum.v, 1, datum.w)
            lpl_tests = log_pseudolikelihood_rand(rbm, tests_datum.v, 1, tests_datum.w)
            push!(history, :lpltrain, iter, lpl_train)
            push!(history, :lpltests, iter, lpl_tests)
            if iter % print_interval < data.batchsize
                println("iter=$iter, lpl_train=$lpl_train")
            end
            if !(lpl_train > min_lpl && lpl_tests > min_lpl)
                @error "lpl_train=$lpl_train or lpl_tests=$lpl_tests less than min_lpl=$min_lpl; stopping (iter=$iter)"
                throw(RBMs.EarlyStop())
            end
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

update_chains_v(rbm::RBM, cd::CD,  vd::NumArray, vm::NumArray = vd, β::Num = 1) =
    sample_v_from_v(rbm, vd, β; steps = cd.steps)::typeof(vd)
update_chains_v(rbm::RBM, cd::PCD, vd::NumArray, vm::NumArray = vd, β::Num = 1) =
    sample_v_from_v(rbm, vm, β; steps = cd.steps)::typeof(vm)

update_chains_h(rbm::RBM, cd::CD,  hd::NumArray, hm::NumArray = hd, β::Num = 1) =
    sample_h_from_h(rbm, hd, β; steps = cd.steps)::typeof(hd)
update_chains_h(rbm::RBM, cd::PCD, hd::NumArray, hm::NumArray = hd, β::Num = 1) =
    sample_h_from_h(rbm, hm, β; steps = cd.steps)::typeof(hm)

"""
    contrastive_divergence_v(rbm, vd, vm, wd = 1, wm = 1)

Contrastive divergence, defined as free energy difference between data (vd) and
model sample (vm). The (optional) `wd,wm` are weights for the batches.
"""
function contrastive_divergence_v(rbm::RBM, vd::NumArray, vm::NumArray, wd::Num = 1, wm::Num = 1)
    Fd = mean_free_energy_v(rbm, vd, wd)
    Fm = mean_free_energy_v(rbm, vm, wm)
    return (Fd - Fm) / length(rbm.vis)
end

"""
    contrastive_divergence_h(rbm, hd, hm, wd = 1, wm = 1)

Contrastive divergence, defined as free energy difference between data (hd) and
model sample (hm). The (optional) `wd,wm` are weights for the batches.
"""
function contrastive_divergence_h(rbm::RBM, hd::NumArray, hm::NumArray, wd::Num = 1, wm::Num = 1)
    Fd = mean_free_energy_h(rbm, hd, wd)
    Fm = mean_free_energy_h(rbm, hm, wm)
    return (Fd - Fm) / length(rbm.hid)
end

"""
    mean_free_energy_v(rbm, v, w = 1)

Mean free energy across batches of visible configurations.
The optional `w` specifies weights for the batches.
"""
mean_free_energy_v(rbm::RBM, v::NumArray, w::Num = 1) = wmean(free_energy_v(rbm, v), w)

"""
    mean_free_energy_h(rbm, h, w = 1)

Mean free energy across batches of hidden configurations.
The optional `w` specifies weights for the batches.
"""
mean_free_energy_h(rbm::RBM, h::NumArray, w::Num = 1) = wmean(free_energy_h(rbm, h), w)
