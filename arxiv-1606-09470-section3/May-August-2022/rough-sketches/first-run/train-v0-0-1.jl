# Experiments with DMM training

include("two-networks.jl")

function reset_dicts!()
    reset_network_dicts!(trainable)
    reset_network_dicts!(handcrafted)
end

# opt = TreeADAM(trainable["network_matrix"], 0.1f0, (0.9f0, 0.999f0), EPS)

# opt = TreeADAM(trainable["network_matrix"], 0.1f0)

opt = TreeADAM(trainable["network_matrix"])

println("DEFINED: opt")

adam_step!(opt, trainable["network_matrix"], 
           convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
                   gradient(loss, trainable)[1]["network_matrix"]))

println("DONE: adam_step!")

function training_step!()
    reset_dicts!()
    adam_step!(opt, trainable["network_matrix"], 
               convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
                       gradient(loss, trainable)[1]["network_matrix"]))
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






