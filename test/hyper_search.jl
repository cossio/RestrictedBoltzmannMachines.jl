using Test, Random, RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: cd_hyper_random_sqrt_lr_decay, cd_hyper_random_exp_lr_decay

@testset "Hyper random search, with simple teacher / student setup" begin
    Random.seed!(3)

    # generate data
    N = 10; B = 1000
    teacher = RBM(Binary(N), Gaussian(1))
    randn!(teacher.weights)
    teacher.weights .*= 1/√N
    train_x = rand((0., 1.), N, B)
    train_x = sample_v_from_v(teacher, train_x; steps=1000)
    train_data = Data((v = train_x, w = ones(B)); batchsize=32)

    student = RBM(Binary(N), Gaussian(1))
    result = cd_hyper_random_sqrt_lr_decay(student, train_data)
    # funny test ... we just to see that this does not error
    @test result.lltrain < 0
end
