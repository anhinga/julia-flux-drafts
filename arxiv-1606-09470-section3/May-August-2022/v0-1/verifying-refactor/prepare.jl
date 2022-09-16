# Prepare for experiments with DMM training

io = open("log-file.txt", "w")

include("utilities.jl")

include("dmm-lite.jl")

include("handcrafted.jl")

include("recurrent.jl")

include("loss.jl")

include("TreeADAM.jl")

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

# opt = TreeADAM(trainable["network_matrix"], 0.1f0)

opt = TreeADAM(trainable["network_matrix"])

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

trainable["network_matrix"]

# println("The network is set to 'sparse', ready to train, use 'steps!(N)'")
println("The network is ready to train, use 'steps!(N)'")







