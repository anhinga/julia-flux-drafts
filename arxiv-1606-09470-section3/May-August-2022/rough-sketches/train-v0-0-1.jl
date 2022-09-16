# Experiments with DMM training

io = open("log-file.txt", "w")

function printlog(io, xs...)
    println(io, xs...)
    flush(io)
end

function printlog_v(io, xs...)
    printlog(io, xs...)
	println(xs...)
end

include("two-networks.jl")

using JSON3

open("initial-matrix.json", "w") do f
    JSON3.pretty(f, JSON3.write(trainable["network_matrix"]))
    println(f) # not really necessary
end

using Serialization

serialize("initial-matrix.ser", trainable["network_matrix"])

function count(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    d = 0
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    d += 1
    end end end end
    d
end

function count_interval(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, min_lim::Float32, max_lim::Float32)
    d = 0
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    v = x[i][j][m][n]
                    if v >= min_lim && v <= max_lim
                        d += 1
                    end 
    end end end end
    d
end

function count_neg_interval(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, min_lim::Float32, max_lim::Float32, vocal = false)
    d = 0
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    v = x[i][j][m][n]
                    if v <= min_lim || v >= max_lim
                        d += 1
                        if vocal
                            println(i, " ", j, " ",  m, " ", n, " ", v)
                        end
                    end 
    end end end end
    d
end

# pseudo-sparsification (just zeros small things out, while we really need to remove them)
function filtercopy(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, lim::Float32)
    y = deepcopy(x)
    for i in keys(y)
        for j in keys(y[i])
            for m in keys(y[i][j])
                for n in keys(y[i][j][m])
                    if abs(y[i][j][m][n]) < lim
                        y[i][j][m][n] = 0.0f0
    end end end end end
    y
end

# a version of function from two-networks.jl TODO: eliminate code duplication
function link!(d::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
               to_neuron::String, to_field::String, from_neuron::String, from_field::String, 
               weight::Float32 = 1.0f0) # additive!
    for s in [to_neuron, to_field, from_neuron]
        if !haskey(d, s)
            d[s] = Dict()
        end
        d = d[s]
    end
    d[from_field] = get_N(d, from_field) + weight # additive to what was there, if anything!
end

# true sparsification
function sparsecopy(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, lim::Float32)
    y = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    if abs(x[i][j][m][n]) >= lim
                        link!(y, i, j, m, n, x[i][j][m][n])
    end end end end end
    y
end

# transpose
function transpose_network_matrix(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
    y = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()
    for i in keys(x)
        for j in keys(x[i])
            for m in keys(x[i][j])
                for n in keys(x[i][j][m])
                    link!(y, m, n, i, j, x[i][j][m][n])
    end end end end
    y
end

function reset_dicts!()
    reset_network_dicts!(trainable)
    reset_network_dicts!(handcrafted)
end

sparse = deserialize("run-3/sparse_2588.ser")

# sparse = sparsecopy(a_1032, 0.001f0) 

trainable["network_matrix"] = sparse

# opt = TreeADAM(trainable["network_matrix"], 0.1f0, (0.9f0, 0.999f0), EPS)

# opt = TreeADAM(trainable["network_matrix"], 0.1f0)

opt = TreeADAM(trainable["network_matrix"])

println("DEFINED: opt")

# adam_step!(opt, trainable["network_matrix"], 
#          convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
#                  gradient(loss, trainable)[1]["network_matrix"]))

println("SKIPPED: adam_step!")

println("The network is set to 'sparse', ready to train, use 'steps!(N)'")
# println("The network is ready to train, use 'steps!(N)'")

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

# strange effects here
#
# the rate of regularization penalty reduction does not seem to depend 
# on the regularization coefficient; is it a strange ADAM property or a bug?
#
# the main part of the loss function goes down at first, but then starts to grow
#
# the question is again: an unfortunate training hyperparameters to be corrected,
# or a bug in the code?
#
# for i in 1:100
#    println("STEP ", i, " ================================")
#    training_step!()
# end
#
# or
# for i in 1:17
#    println("STEP-NO-RESET ", i, " >>>>>>>>>>>>>>>>>>>>>>>")
#    adam_step!(opt, trainable["network_matrix"],
#               convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
#				        gradient(loss, trainable)[1]["network_matrix"]))
# end
#
# We need to instrument this better to keep track of evolution of values
# For the time being, let's try to add L2 in addition to L1

# one training step is done 
# (oops, changing "timer" links too, and we did not mean to do that)

trainable["network_matrix"]






