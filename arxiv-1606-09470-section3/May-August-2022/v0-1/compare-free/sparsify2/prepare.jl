# Prepare for experiments with DMM training

io = open("log-file.txt", "w")

include("utilities.jl")

include("dmm-lite.jl")

include("handcrafted.jl")

include("feedforward_with_accums.jl")

include("loss.jl")

include("TreeADAM.jl")

function min_ind_dict(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    min_v = 1.0f8
    indices = ("not found", "not found", "not found", "not found")
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    v = x[i][j][m][n]
                    if abs(v) < min_v
                        min_v = abs(v)
                        indices = (i, j, m, n)
                    end
    end end end end
	(min_v, indices)
end

function trim_network(trainable::Dict{String, Dict{String}}, opt::TreeADAM, lim::Float32)
    (y, next1, next2) = sparsecopy(trainable["network_matrix"], lim, opt.mt, opt.vt)
    trainable["network_matrix"] = y
    opt.mt = next1
    opt.vt = next2	
end

using JSON3

open("initial-matrix.json", "w") do f
    JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
    println(f) # not really necessary
end

using Serialization

serialize("initial-matrix.ser", trainable["network_matrix"])



# sparse = deserialize("run-3/sparse_2588.ser")

# sparse = sparsecopy(a_1032, 0.001f0) 

# trainable["network_matrix"] = sparse

# opt = TreeADAM(trainable["network_matrix"], 0.1f0, (0.9f0, 0.999f0), EPS)

opt = TreeADAM(trainable["network_matrix"], 0.001f0)

# opt = TreeADAM(trainable["network_matrix"])

println("DEFINED: opt")

# adam_step!(opt, trainable["network_matrix"], 
#          convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
#                  gradient(loss, trainable)[1]["network_matrix"]))

println("SKIPPED: adam_step!")

# ------------------------------------------------------------------------------------------

function reset_dicts!()
    reset_network_dicts!(trainable)
    reset_network_dicts!(handcrafted)
end

function training_step!()
    reset_dicts!()
    adam_step!(opt, trainable["network_matrix"], 
               convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
                       gradient(loss, trainable)[1]["network_matrix"]))
end

using Dates

function steps!(n_steps)
    printlog_v(io, now())
    for i in 1:n_steps
        printlog_v(io, "STEP ", i, " ================================")
        training_step!()
    end
    printlog_v(io, now())
end

function sparsifying_steps!(n_steps)
    printlog_v(io, now())
    for i in 1:n_steps
        printlog_v(io, "STEP ", i, " ================================")
        training_step!()
        lim = min_abs_dict(trainable["network_matrix"])
        trim_network(trainable, opt, lim)
        printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
    end
    printlog_v(io, now())
end

# one sparsifying step out of N
function interleaving_steps!(n_steps, N=2)
    printlog_v(io, now())
    for i in 1:n_steps
        printlog_v(io, "STEP ", i, " ================================")
        training_step!()
        if i%N == 1
            lim = min_abs_dict(trainable["network_matrix"])
            trim_network(trainable, opt, lim)
            printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
        end
    end
    printlog_v(io, now())
end

trainable["network_matrix"]

# println("The network is set to 'sparse', ready to train, use 'steps!(N)'")
println("The network is ready to train, use 'steps!(N)'")







