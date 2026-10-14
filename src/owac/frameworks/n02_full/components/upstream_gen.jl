# P0 numerical reference. Executes the pinned Gen particle filter on exactly the
# same finite HMM and observations as smc_ref.py; this is not a P3 mechanism.
using Gen, Random, JSON, SHA

payload = JSON.parsefile(ARGS[1])
expected_root = realpath(ARGS[3])
@assert startswith(realpath(pathof(Gen)), expected_root * "/")
transition = reduce(vcat, permutedims.(payload["transition"]))
emission = reduce(vcat, permutedims.(payload["emission"]))
initial = Float64.(payload["init"])
obs = Int.(payload["observations"]) .+ 1
exact = reduce(vcat, permutedims.(payload["exact"]))

@gen function hmm(T::Int)
    z = 0
    for t in 1:T
        probabilities = t == 1 ? initial : Float64.(transition[z, :])
        z = @trace(categorical(probabilities), (:z, t))
        @trace(categorical(Float64.(emission[z, :])), (:y, t))
    end
    return z
end

function measure_filter(obs, n)
    state = initialize_particle_filter(hmm, (1,), choicemap(((:y, 1), obs[1])), n)
    marginals = Vector{Vector{Float64}}()
    resampling = Bool[]
    for t in eachindex(obs)
        if t > 1
            push!(resampling, maybe_resample!(state))
            particle_filter_step!(state, (t,), (UnknownChange(),), choicemap(((:y, t), obs[t])))
        end
        lw = get_log_weights(state)
        weights = exp.(lw .- maximum(lw)); weights ./= sum(weights)
        marginal = zeros(5)
        for (trace, weight) in zip(get_traces(state), weights)
            marginal[trace[(:z, t)]] += weight
        end
        push!(marginals, marginal)
    end
    return marginals, resampling
end

Random.seed!(payload["seed"])
started = time()
marginals, resampling = measure_filter(obs, payload["num_particles"])
tv = [sum(abs.(marginals[t] - exact[t, :])) / 2 for t in eachindex(obs)]
source_hashes = Dict(name => bytes2hex(sha256(read(joinpath(expected_root, name)))) for name in
    ["src/inference/particle_filter.jl", "src/inference/trace_translators.jl"])
result = Dict(
    "upstream_repository" => "probcomp/Gen.jl",
    "upstream_commit" => "2a27298b392e89fa67837f8871b624a41d901d19",
    "julia_version" => string(VERSION), "gen_source" => pathof(Gen),
    "source_hashes" => source_hashes, "input_sha256" => bytes2hex(sha256(read(ARGS[1]))),
    "seed" => payload["seed"], "num_particles" => payload["num_particles"],
    "marginals" => marginals, "total_variation_by_step" => tv,
    "tv_mean" => sum(tv) / length(tv), "tv_max" => maximum(tv),
    "resampled" => resampling, "elapsed_seconds" => time() - started,
    "status" => maximum(tv) < 0.05 ? "pass" : "fail",
    "tolerance" => 0.05, "scope" => "finite HMM reference; no variable-dimension proposal or P3 behavior claim")
open(ARGS[2], "w") do f
    JSON.print(f, result, 2)
end
println("Gen SMC: tv_mean=", result["tv_mean"], " tv_max=", result["tv_max"], " status=", result["status"])
result["status"] == "pass" || error("Gen SMC outside frozen finite-sample bound")
