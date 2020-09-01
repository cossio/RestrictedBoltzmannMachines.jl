using RestrictedBoltzmannMachines: Data

@testset "data, basic tests" begin
    train_x = rand((1, 2, 3), 100, 20)
    train_y = rand(Bool, 20)
    train_w = rand(20)
    d = Data((v=train_x, w=train_w, y=train_y); batchsize=4)

    @inferred iterate(d)
    @inferred first(d)
    @test size(first(d).v) == (100, 4)
    @test size(first(d).y) == (4,)
    @test size(first(d).w) == (4,)
end

@testset "data consistent batches" begin
    x = rand(10, 20, 100)
    y = dropdims(sum(x; dims=(1,2)); dims=(1,2))
    z = dropdims(sum(sin.(x); dims=(2,)); dims=(2,))
    d = Data((x=x, y=y, z=z); batchsize=16)

    for (iter, batch) in zip(1:100, d)
        @test batch.y ≈ dropdims(sum(batch.x; dims=(1,2)), dims=(1,2))
        @test batch.z ≈ dropdims(sum(sin.(batch.x); dims=(2,)), dims=(2,))
    end
end

@testset "data contains all batches" begin
    x = rand(5, 3, 4)
    for b = 1:4
        x[1,:,b] .= b
    end
    y = dropdims(sum(x; dims=(1,2)); dims=(1,2))
    z = dropdims(sum(sin.(x); dims=(2,)); dims=(2,))
    d = Data((x=x, y=y, z=z); batchsize=2)
    @test d.nobs == 4

    collected_x = Array{Float64}[]
    collected_y = Array{Float64}[]
    collected_z = Array{Float64}[]
    for (i,p) in zip(1:2, d)
        @test size(p.x) == (5,3,2)
        @test size(p.y) == (2,)
        @test size(p.z) == (5,2)
        push!(collected_x, p.x)
        push!(collected_y, p.y)
        push!(collected_z, p.z)
    end
    @test Set(cat(collected_x...; dims=3)[1,1,:]) == Set([1,2,3,4])

    collected_x = Array{Float64}[]
    collected_y = Array{Float64}[]
    collected_z = Array{Float64}[]
    for (i,p) in zip(1:4, d)
        @test size(p.x) == (5,3,2)
        @test size(p.y) == (2,)
        @test size(p.z) == (5,2)
        push!(collected_x, p.x)
        push!(collected_y, p.y)
        push!(collected_z, p.z)
    end
    @test Set(cat(collected_x...; dims=3)[1,1,1:4]) == Set([1,2,3,4])
    @test Set(cat(collected_x...; dims=3)[1,1,5:8]) == Set([1,2,3,4])
end

@testset "empty data" begin
    data = Data()
    @test isempty(first(data))
    for (i,d) in enumerate(data)
        break
    end
end
