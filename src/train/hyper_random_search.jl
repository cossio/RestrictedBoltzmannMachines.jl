function _cd_hyper_random(rbm_init::RBM, train_data::Data; tests_data::Data=train_data, decay,
                            λws = (@. 10^(-4 * (0:0.1:1))), λgs = (@. 10^(-5 * (0:0.1:1))), λhs = (@. 10^(-4 * (0:0.1:1))),
                            w0s = 2 .* (0:0.01:1.0), β0s=0:1e-6:1, η0s = 0:1e-6:0.05, cd_steps = 1:10,
                            no_iters = 30train_data.nobs, min_lpl = -10)
    # sample hyper-parameters
    λw = rand(λws); λg = rand(λgs); λh = rand(λhs);
    w0 = rand(w0s); β0 = rand(β0s); η0 = rand(η0s);
    cdsteps = rand(cd_steps)

    println("Training: λw=$λw, λg=$λg, λh=$λh, w0=$w0, β0=$β0, η0=$η0, lrdecay=$(decay.decay), cdsteps=$cdsteps")

    # init
    rbm = deepcopy(rbm_init)
    init!(rbm, train_data.tensors.v; w=w0)

    # train
    opt = Optimiser(decay, ADAM(η0, (β0, 0.999)))
    history = MVHistory()
    train!(rbm, train_data; cd = RBMs.PCD(cdsteps), iters=no_iters, tests_data=tests_data,
            history=history, opt=opt, λw=λw, λg=λg, λh=λh, min_lpl=min_lpl)
    
    # result
    lltrain = log_pseudolikelihood_rand(rbm, train_data)
    lltests = log_pseudolikelihood_rand(rbm, tests_data)
    return (λw = λw, λg = λg, λh = λh, w0 = w0, β0 = β0, η0 = η0,
            cdsteps = cdsteps, lltrain = lltrain, lltests = lltests)
end

function cd_hyper_random_sqrt_lr_decay(rbm_init::RBM, data::Data;
                                       lrdecays = (@. 10^(2 + 2 * (0:0.1:1))),
                                       no_iters = 30data.nobs, kwargs...)
    lrdecay = rand(lrdecays)
    decay = SqrtDecay(decay = lrdecay * data.batchsize / no_iters)
    nt = _cd_hyper_random(rbm_init, data; decay=decay, no_iters=no_iters, kwargs...)
    return (nt..., lrdecay = lrdecay)
end

function cd_hyper_random_exp_lr_decay(rbm_init::RBM, data::Data;
                                      lrdecays = (@. 10^(-3 * (0:0.1:1) - 1)),
                                      no_iters = 30data.nobs, kwargs...)
    lrdecay = rand(lrdecays)
    decay = GeometricDecay(decay = lrdecay^(data.batchsize / no_iters))
    nt = _cd_hyper_random(rbm_init, data; decay=decay, no_iters=no_iters, kwargs...)
    return (nt..., lrdecay = lrdecay)
end
