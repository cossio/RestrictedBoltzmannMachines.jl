using RestrictedBoltzmannMachines: RBM, pcd!, Binary

rbm = RBM(Binary(; θ=zeros(5)), Binary(; θ=zeros(3)), randn(5,3))
