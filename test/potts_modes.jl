import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset

@testset "Potts conditional modes with full ties" begin
    for Layer in (RBMs.Potts, RBMs.PottsGumbel), sites in ((), (2,), (2, 2)),
            batch in ((), (4,), (2, 3)), field in (0.0, 1.0)
        layer = Layer((3, sites...))
        layer.θ .= field
        rbm = RBMs.RBM(RBMs.Binary((1,)), layer, zeros(1, size(layer)...))
        x = zeros(1, batch...)
        expected = falses(size(layer)..., batch...)
        selectdim(expected, 1, 1) .= true

        for mode in (RBMs.mode_h_from_v(rbm, x), RBMs.mode_v_from_h(RBMs.mirror(rbm), x))
            @test size(mode) == size(expected)
            @test eltype(mode) == Bool
            @test mode == expected
            @test all(sum(mode; dims = 1) .== 1)
            @test all(RBMs.energy(layer, mode) .== -field * prod(sites))
        end
        isempty(batch) && @test RBMs.mode_from_inputs(layer) == expected
    end
end

@testset "Potts conditional modes with partial ties and unique maxima" begin
    for Layer in (RBMs.Potts, RBMs.PottsGumbel)
        layer = Layer(; θ = repeat([0.0 3.0; 2.0 0.0; 2.0 3.0], 1, 1, 2))
        w = reshape(repeat([2.0 -3.0; 0.0 3.0; 0.0 0.0], 1, 1, 2), 1, 3, 2, 2)
        rbm = RBMs.RBM(RBMs.Binary((1,)), layer, w)
        # x = 0 gives partial ties; x = 1 gives full and partial ties;
        # x = 2 gives unique maxima. Batches mix these cases across samples.
        for x in ([0.0], [1.0], [2.0], [0.0 1.0 2.0], reshape([0.0, 1.0, 2.0, 1.0, 2.0, 0.0], 1, 2, 3))
            logits = layer.θ .+ RBMs.inputs_h_from_v(rbm, x)
            for (model, conditional_mode) in ((rbm, RBMs.mode_h_from_v), (RBMs.mirror(rbm), RBMs.mode_v_from_h))
                mode = conditional_mode(model, x)
                @test size(mode) == size(logits)
                @test eltype(mode) == Bool
                @test all(sum(mode; dims = 1) .== 1)
                @test sum(mode .* logits; dims = 1) == maximum(logits; dims = 1)
                @test mode == conditional_mode(model, x)
            end
        end
    end
end
