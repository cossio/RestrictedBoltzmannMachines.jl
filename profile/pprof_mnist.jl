using Random, Test, Profile
import RestrictedBoltzmannMachines as RBMs
import MLDatasets, PProf

println("Loading MNIST data ...")

# load MNIST dataset
train_x, train_y = MLDatasets.MNIST.traindata();
tests_x, tests_y = MLDatasets.MNIST.testdata();

# floating type we will use
Float = Float64

# since we train a binary RBM, we binarize the data first
train_x = Float.(train_x .≥ 0.5);
tests_x = Float.(tests_x .≥ 0.5);
train_y = Float.(train_y);
tests_y = Float.(tests_y);

# initialize RBM with 100 hidden units
rbm = RBMs.RBM(
    RBMs.Binary(Float, 28, 28),
    RBMs.Binary(Float, 100),
    randn(Float, 28, 28, 100) / 28
);

println("Initial quick run to pre-compile things ...")

# short run to pre-compile things before collecting profile
@time RBMs.pcd!(rbm, train_x[:,:,1:64]; epochs=2, batchsize=16);

println("Profiling ...")

# collect profile
history = @profile RBMs.pcd!(rbm, train_x; epochs=10, batchsize=128);

#=
This prints a link to a local webserver where you can inspect the profile
you've collected. It produces a file called profile.pb.gz in the pprof format,
and then opens the pprof tool in interactive "web" mode.
Source: https://github.com/JuliaPerf/PProf.jl
=#
PProf.pprof()
