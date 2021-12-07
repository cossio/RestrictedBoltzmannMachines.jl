var documenterSearchIndex = {"docs":
[{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"EditURL = \"https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/docs/src/literate/MNIST.jl\"","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"We begin by importing the required packages. We load MNIST via MLDatasets.jl. Here we also plot some of the first digits.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"import RestrictedBoltzmannMachines as RBMs\nusing CairoMakie\nimport MLDatasets\n\nfig = Figure(resolution=(500, 500))\nfor i in 1:5, j in 1:5\n    ax = Axis(fig[i,j])\n    hidedecorations!(ax)\n    heatmap!(ax, first(MLDatasets.MNIST.traindata(5 * (i - 1) + j)))\nend\nfig","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"First we load the MNIST dataset.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"train_x, train_y = MLDatasets.MNIST.traindata()\ntests_x, tests_y = MLDatasets.MNIST.testdata()\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"We will train an RBM with binary (0,1) visible and hidden units. Therefore we binarize the data first.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"train_x = float(train_x .≥ 0.5)\ntests_x = float(tests_x .≥ 0.5)\ntrain_y = float(train_y)\ntests_y = float(tests_y)\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"In the previous code block, notice how we converted train_x, train_y, ..., and so on, to floats using float, which in this case converts to Float64. The RBM we will define below also uses Float64 to store weights (it's the default). This is important if we want to hit blas matrix multiplies, which are much faster than, e.g., using a BitArray to store the data. Thus be careful that the data and the RBM weights have the same float type.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Plot some examples of the binarized data (same digits as above).","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"fig = Figure(resolution=(500, 500))\nfor i in 1:5, j in 1:5\n    ax = Axis(fig[i,j])\n    hidedecorations!(ax)\n    heatmap!(ax, train_x[:,:,5 * (i - 1) + j])\nend\nfig","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Initialize an RBM with 100 hidden units. It is recommended to initialize the weights as random normals with zero mean and standard deviation = 1/sqrt(number of hidden units). See Glorot & Bengio 2010.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"rbm = RBMs.RBM(RBMs.Binary(28,28), RBMs.Binary(100), randn(28, 28, 100) / 28)\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Train the RBM on the data. This returns a MVHistory object (from ValueHistories.jl), containing things like the pseudo-likelihood of the data during training. We print here the time spent in the training as a rough benchmark.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"history = RBMs.train!(rbm, train_x; epochs=100, batchsize=128)\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Plot log-pseudolikelihood during learning.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"lines(get(history, :lpl)...)","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Generate some RBM samples.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"dream_x = RBMs.sample_v_from_v(rbm, train_x; steps=20);\n\nfig = Figure(resolution=(500, 500))\nfor i in 1:5, j in 1:5\n    ax = Axis(fig[i,j])\n    hidedecorations!(ax)\n    heatmap!(ax, dream_x[:,:,5 * (i - 1) + j])\nend\nfig","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"This page was generated using Literate.jl.","category":"page"},{"location":"reference/#RestrictedBoltzmannMachines.jl-Reference","page":"Reference","title":"RestrictedBoltzmannMachines.jl Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [RBMs]","category":"page"},{"location":"reference/#RestrictedBoltzmannMachines.Binary","page":"Reference","title":"RestrictedBoltzmannMachines.Binary","text":"Binary(θ)\n\nBinary layer, with external fields θ.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RestrictedBoltzmannMachines.Gaussian","page":"Reference","title":"RestrictedBoltzmannMachines.Gaussian","text":"Gaussian(θ, γ)\n\nGaussian layer, with location parameters θ and scale parameters γ.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RestrictedBoltzmannMachines.Potts","page":"Reference","title":"RestrictedBoltzmannMachines.Potts","text":"Potts(θ)\n\nPotts layer, with external fields θ. Encodes categorical variables as one-hot vectors. The number of classes is the size of the first dimension.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RestrictedBoltzmannMachines.RBM","page":"Reference","title":"RestrictedBoltzmannMachines.RBM","text":"RBM\n\nRepresents a restricted Boltzmann Machine.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RestrictedBoltzmannMachines.ReLU","page":"Reference","title":"RestrictedBoltzmannMachines.ReLU","text":"ReLU(θ, γ)\n\nReLU layer, with location parameters θ and scale parameters γ.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RestrictedBoltzmannMachines.Spin","page":"Reference","title":"RestrictedBoltzmannMachines.Spin","text":"Spin(θ)\n\nSpin layer, with external fields θ.\n\n\n\n\n\n","category":"type"},{"location":"reference/#RestrictedBoltzmannMachines.batchmul-Tuple{AbstractArray, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.batchmul","text":"batchmul(fields, x)\n\nForms the product fields * x, summing over the fields dimension while leaving the batch dimensions.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.categorical_rand-Tuple{AbstractVector}","page":"Reference","title":"RestrictedBoltzmannMachines.categorical_rand","text":"categorical_rand(ps)\n\nRandomly draw i with probability ps[i]. You must ensure that ps defines a proper probability distribution.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.categorical_sample-Tuple{AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.categorical_sample","text":"categorical_sample(P)\n\nGiven a probability array P of size (q, *), returns an array C of size (*), such that C[i] ∈ 1:q is a random sample from the categorical distribution P[:,i]. You must ensure that P defines a proper probability distribution.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.categorical_sample_from_logits-Tuple{AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.categorical_sample_from_logits","text":"categorical_sample_from_logits(logits)\n\nGiven a logits array logits of size (q, *) (where q is the number of classes), returns an array X of size (*), such that X[i] is a categorical random sample from the distribution with logits logits[:,i].\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.categorical_sample_from_logits_gumbel-Tuple{AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.categorical_sample_from_logits_gumbel","text":"categorical_sample_from_logits_gumbel(logits)\n\nLike categoricalsamplefrom_logits, but using the Gumbel trick.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.energy-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.energy","text":"energy(rbm, v, h)\n\nEnergy of the rbm in the configuration (v,h).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.flip_layers-Tuple{RestrictedBoltzmannMachines.RBM}","page":"Reference","title":"RestrictedBoltzmannMachines.flip_layers","text":"flip_layers(rbm)\n\nReturns a new RBM with viible and hidden layers flipped.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.free_energy","page":"Reference","title":"RestrictedBoltzmannMachines.free_energy","text":"free_energy(rbm, v, β=1)\n\nFree energy of visible configuration (after marginalizing hidden configurations).\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.generate_sequences","page":"Reference","title":"RestrictedBoltzmannMachines.generate_sequences","text":"generate_sequences(n, A = 0:1)\n\nRetruns an iterator over all sequences of length n out of the alphabet A.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.init!-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.init!","text":"init!(rbm, data; ϵ = 1e-6)\n\nInits the RBM, computing average visible unit activities from data.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.init_weights!-Tuple{RestrictedBoltzmannMachines.RBM}","page":"Reference","title":"RestrictedBoltzmannMachines.init_weights!","text":"init_weights!(rbm; w=1)\n\nRandom initialization of weights, as independent normals with variance 1/N. All patterns are of norm 1.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.inputs_h_to_v-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.inputs_h_to_v","text":"inputs_h_to_v(rbm, h)\n\nInteraction inputs from hidden to visible layer.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.inputs_v_to_h-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.inputs_v_to_h","text":"inputs_v_to_h(rbm, v)\n\nInteraction inputs from visible to hidden layer.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.interaction_energy-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.interaction_energy","text":"interaction_energy(rbm, v, h)\n\nWeight mediated interaction energy.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.log_likelihood","page":"Reference","title":"RestrictedBoltzmannMachines.log_likelihood","text":"log_likelihood(rbm, v, β=1)\n\nLog-likelihood of v under rbm, with the partition function compued by extensive enumeration. For discrete layers, this is exponentially slow for large machines.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.log_partition","page":"Reference","title":"RestrictedBoltzmannMachines.log_partition","text":"log_partition(rbm, β = 1)\n\nLog-partition of the rbm at inverse temperature β, computed by extensive enumeration of  states (except for particular cases such as Gaussian-Gaussian) RBM). This is exponentially slow for large machines.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.log_pseudolikelihood","page":"Reference","title":"RestrictedBoltzmannMachines.log_pseudolikelihood","text":"log_pseudolikelihood(rbm, v, sites, β=1)\n\nLog-pseudolikelihood of a site conditioned on the other sites, where sites is an array of site indices (CartesianIndex), one for each batch. Returns an array of log-pseudolikelihood, for each batch.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.log_pseudolikelihood-2","page":"Reference","title":"RestrictedBoltzmannMachines.log_pseudolikelihood","text":"log_pseudolikelihood_rand(rbm, v, β=1)\n\nLog-pseudolikelihood of randomly chosen sites conditioned on the other sites. For each configuration choses a samplefrominputs site, and returns the mean of the computed pseudo-likelihoods.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.log_site_trace","page":"Reference","title":"RestrictedBoltzmannMachines.log_site_trace","text":"log_site_trace(site, rbm, v, β=1)\n\nLog of the trace over configurations of site. Here v must consist of a single batch.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.log_site_traces","page":"Reference","title":"RestrictedBoltzmannMachines.log_site_traces","text":"log_site_traces(rbm, v, sites, β=1)\n\nLog of the trace over configurations of sites, where sites is an array of site indices (CartesianIndex), for each batch. Returns an array of the log-traces for each batch.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.lognormcdf-Tuple{Real, Real}","page":"Reference","title":"RestrictedBoltzmannMachines.lognormcdf","text":"lognormcdf(a, b)\n\nComputes log(normcdf(a, b)), but retaining accuracy.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.mean_-Tuple{AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.mean_","text":"mean_(A, dims)\n\nTakes the mean of A across dimensions dims and drops them.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.mills-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.mills","text":"mills(x::Real)\n\nMills ratio of the standard normal distribution. Defined as (1 - cdf(x)) / pdf(x).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.minibatch_count-Tuple{Int64}","page":"Reference","title":"RestrictedBoltzmannMachines.minibatch_count","text":"minibatch_count(nobs; batchsize)\n\nNumber of minibatches.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.minibatch_count-Tuple{Vararg{AbstractArray}}","page":"Reference","title":"RestrictedBoltzmannMachines.minibatch_count","text":"minibatch_count(data; batchsize)\n\nNumber of minibatches.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.minibatches-Tuple{Int64}","page":"Reference","title":"RestrictedBoltzmannMachines.minibatches","text":"minibatches(nobs; batchsize, shuffle = true)\n\nPartition nobs into minibatches of length n. If necessary repeats some observations to complete last batches. (Therefore all batches are of the same size n).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.minibatches-Tuple{Vararg{AbstractArray}}","page":"Reference","title":"RestrictedBoltzmannMachines.minibatches","text":"minibatches(datas...; batchsize)\n\nSplits the given datas into minibatches. Each minibatch is a tuple where each entry is a minibatch from the corresponding data within datas. All minibatches are of the same size batchsize (if necessary repeating some samples at the last minibatches).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.normalize_weights-Tuple{RestrictedBoltzmannMachines.RBM}","page":"Reference","title":"RestrictedBoltzmannMachines.normalize_weights","text":"normalize_weights(rbm)\n\nRescales weights so that norm(w[:,μ]) = 1 for all μ (making individual weights ~ 1/√N). The resulting RBM shares layer parameters with the original, but weights are a new array.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.normcdf-Tuple{Real, Real}","page":"Reference","title":"RestrictedBoltzmannMachines.normcdf","text":"normcdf(a, b)\n\nProbablity that a ≤ Z ≤ b, where Z is a standard normal samplefrominputs variable. WARNING: Silently returns a negative value if a > b.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.normcdf-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.normcdf","text":"normcdf(x)\n\nProbablity that Z ≤ x, where Z is a standard normal samplefrominputs variable.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.normcdfinv-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.normcdfinv","text":"normcdfinv(x)\n\nInverse of normcdf.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.onehot_decode-Tuple{AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.onehot_decode","text":"onehot_decode(X)\n\nGiven a onehot encoded array X of N + 1 dimensions, returns the equivalent categorical array of N dimensions.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.onehot_encode-Union{Tuple{AbstractArray{T}}, Tuple{T}, Tuple{AbstractArray{T}, Any}} where T","page":"Reference","title":"RestrictedBoltzmannMachines.onehot_encode","text":"onehot_encode(A, code)\n\nGiven an array A of N dimensions, returns a one-hot encoded array of N + 1 dimensions where single entries of the first dimension are one.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.randgumbel-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Reference","title":"RestrictedBoltzmannMachines.randgumbel","text":"randgumbel(T = Float64)\n\nGenerates a random Gumbel variate.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.randnt-Tuple{Random.AbstractRNG, Real}","page":"Reference","title":"RestrictedBoltzmannMachines.randnt","text":"randnt([rng], a)\n\nRandom standard normal lower truncated at a (that is, Z ≥ a).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.randnt_half-Tuple{Random.AbstractRNG, Real, Real}","page":"Reference","title":"RestrictedBoltzmannMachines.randnt_half","text":"randnt_half([rng], μ, σ)\n\nSamples the normal distribution with mean μ and standard deviation σ truncated to positive values.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.reconstruction_error","page":"Reference","title":"RestrictedBoltzmannMachines.reconstruction_error","text":"reconstruction_error(rbm, v, β = 1; steps = 1)\n\nStochastic reconstruction error of v.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.sample_h_from_h","page":"Reference","title":"RestrictedBoltzmannMachines.sample_h_from_h","text":"sample_h_from_h(rbm, h, β = 1; steps = 1)\n\nSamples a hidden configuration conditional on another hidden configuration h.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.sample_h_from_v","page":"Reference","title":"RestrictedBoltzmannMachines.sample_h_from_v","text":"sample_h_from_v(rbm, v, β=1)\n\nSamples a hidden configuration conditional on the visible configuration v.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.sample_v_from_h","page":"Reference","title":"RestrictedBoltzmannMachines.sample_v_from_h","text":"sample_v_from_h(rbm, h, β = 1)\n\nSamples a visible configuration conditional on the hidden configuration h.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.sample_v_from_v","page":"Reference","title":"RestrictedBoltzmannMachines.sample_v_from_v","text":"sample_v_from_v(rbm, v, β = 1; steps = 1)\n\nSamples a visible configuration conditional on another visible configuration v.\n\n\n\n\n\n","category":"function"},{"location":"reference/#RestrictedBoltzmannMachines.sqrt1half-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.sqrt1half","text":"sqrt1half(x)\n\nAccurate computation of sqrt(1 + (x/2)^2) + |x|/2.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.sum_-Tuple{AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.sum_","text":"sum_(A, dims)\n\nSums A over dimensions dims and drops them.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.tnmean-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.tnmean","text":"tnmean(a)\n\nMean of the standard normal distribution, truncated to the interval (a, +∞).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.tnstd-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.tnstd","text":"tnstd(a)\n\nStandard deviation of the standard normal distribution, truncated to the interval (a, +∞).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.tnvar-Tuple{Real}","page":"Reference","title":"RestrictedBoltzmannMachines.tnvar","text":"tnvar(a)\n\nVariance of the standard normal distribution, truncated to the interval (a, +∞). WARNING: Fails for very very large values of a.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.train!-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.train!","text":"train!(rbm, data)\n\nTrains the RBM on data.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.tuplen-Union{Tuple{Val{N}}, Tuple{N}} where N","page":"Reference","title":"RestrictedBoltzmannMachines.tuplen","text":"tuplen(Val(N))\n\nConstructs the tuple (1, 2, ..., N).\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.unwhiten-Tuple{RestrictedBoltzmannMachines.RBM, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.unwhiten","text":"unwhiten(rbm, data)\n\nGiven an RBM trained on whitened data, returns an RBM that can look at original data.\n\n\n\n\n\n","category":"method"},{"location":"reference/#RestrictedBoltzmannMachines.weighted_mean-Tuple{AbstractArray, AbstractArray}","page":"Reference","title":"RestrictedBoltzmannMachines.weighted_mean","text":"weighted_mean(v, w)\n\nMean of v with weights w.\n\n\n\n\n\n","category":"method"},{"location":"math/#Mathematical-introduction-to-Restricted-Boltzmann-Machines","page":"Mathematical introduction","title":"Mathematical introduction to Restricted Boltzmann Machines","text":"","category":"section"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"A restricted Boltzmann machine (RBM) with visible units mathbfv = (v_1 ldots v_N) and hidden units mathbfh = (h_1 ldots h_M) has an energy function defined by:","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"E(mathbfv mathbfh) = sum_i mathcalV_i(v_i) + sum_mumathcalU_mu(h_mu) - sum_imu w_imu v_i h_mu","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"where mathcalV_i(v_i) and mathcalU_mu(h_mu) are the unit potentials and w_imu the interaction weights. The probability of a configuration is:","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"P(mathbfv mathbfh) = frac1Zmathrme^-beta E(mathbfvmathbfh)","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"where","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"Z = sum_mathbfv mathbfh mathrme^-beta E(mathbfv mathbfh)","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"is the partition function and beta the inverse temperature. The machine assigns a likelihood:","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"P(mathbfv) = undersetmathbfhsum P (mathbfv mathbfh) =\nfrac1Z mathrme^-beta E_textrmeff(mathbfv)","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"to visible configurations, where E_textrmeff(mathbfv) is the free energy:","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"E_textrmeff(mathbfv) = sum_i mathcalV_i(v_i) - sum_mu\nGamma_mu left(sum_i w_i mu v_i right)","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"and","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"Gamma_mu(I) = frac1beta ln sum_h_mu mathrme^beta(I h_mu - mathcalU_mu(h_mu))","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"are the cumulant generating functions associated to the hidden unit potentials.","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"Note that beta refers to the inverse temperature in the distribution P(mathbfvmathbfh). If instead we want to sample the marginal P(mathbfv) at a different inverse temperature beta_v,  we would have to use the distribution:","category":"page"},{"location":"math/","page":"Mathematical introduction","title":"Mathematical introduction","text":"P_beta_v(mathbfv) = fracmathrme^- beta_v E_textrmeff\n(mathbfv)sum_mathbfv mathrme^-beta_mathrmv E_textrmeff\n(mathbfv)","category":"page"},{"location":"#RestrictedBoltzmannMachines.jl-Documentation","page":"Home","title":"RestrictedBoltzmannMachines.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia package to train and simulate Restricted Boltzmann Machines.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package doesn't export any symbols. It is recommended to import the package as:","category":"page"},{"location":"","page":"Home","title":"Home","text":"import RestrictedBoltzmannMachines as RBMs","category":"page"},{"location":"","page":"Home","title":"Home","text":"to avoid typing the long name.","category":"page"}]
}
