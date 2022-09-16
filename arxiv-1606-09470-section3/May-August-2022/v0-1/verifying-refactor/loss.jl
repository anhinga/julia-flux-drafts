

#=
# and to run one might wish to use sleep(1) between cycles to slow it down and see what's happening
for i in 1:150
    two_stroke_cycle!(handcrafted)
    println("timer:   ", handcrafted["neurons"]["timer"].output_dict)
    println("input:   ", handcrafted["neurons"]["input"].output_dict)
    println("accum:   ", handcrafted["neurons"]["accum"].output_dict)
    println("norm:    ", handcrafted["neurons"]["norm"].output_dict)
    println("compare: ", handcrafted["neurons"]["compare"].output_dict)
    println("dot:     ", handcrafted["neurons"]["dot"].output_dict)
    println()
    println("OUTPUT: ", handcrafted["neurons"]["output"].input_dict)
    println()
    sleep(0.1)
end

for i in 1:150
    two_stroke_cycle!(trainable)
    println("timer:   ", trainable["neurons"]["timer"].output_dict)
    println("input:   ", trainable["neurons"]["input"].output_dict)
    println()
    println("OUTPUT: ", trainable["neurons"]["output"].input_dict)
    println()
    sleep(0.1)
end
=#

square(x::Float32) = x*x

function loss(dmm_lite::DMM_Lite_)
    l = 0.0f0
    for i in 1:35 # 1:12 # 1:35 ok speed (use 0.001 reg mult), 
	              # 1:140 works, but the slowdown is spectacular (use 0.01 reg mult)
        two_stroke_cycle!(dmm_lite)
        two_stroke_cycle!(handcrafted)
        target_1 = get_N(handcrafted["neurons"]["output"].input_dict["dict-1"], ":number")
        target_2 = get_N(handcrafted["neurons"]["output"].input_dict["dict-2"], ":number")
        l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-1"], ":number") - target_1)
        l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-2"], ":number") - target_2)
    end
    regularization = 0.0f0
    for i in keys(dmm_lite["network_matrix"])
	    if i != "timer" && i != "input"
            for j in keys(dmm_lite["network_matrix"][i])
                for m in keys(dmm_lite["network_matrix"][i][j])
                    for n in keys(dmm_lite["network_matrix"][i][j][m])
                        regularization += abs(dmm_lite["network_matrix"][i][j][m][n]) # + 
						                  # 10.0f0 * square(dmm_lite["network_matrix"][i][j][m][n])
    end end end end end
	reg_novel = 0.0f0
    for i in keys(dmm_lite["network_matrix"])
	    if i != "timer" && i != "input"
            for j in keys(dmm_lite["network_matrix"][i])
                weights_sum = 0.0f0
                for m in keys(dmm_lite["network_matrix"][i][j])
                    for n in keys(dmm_lite["network_matrix"][i][j][m])
                        weights_sum += dmm_lite["network_matrix"][i][j][m][n]
                end end
				reg_novel += square(weights_sum - 1.0f0)
    end end end

    printlog_v(io, "prereg loss ", l, " regularization ", regularization, " reg_novel ", reg_novel)
    l += 0.001f0 * regularization + 0.001f0 * reg_novel	
	printlog_v(io, "loss ", l)
    l
end

#this_loss = loss(handcrafted)
#this_loss = loss(trainable)

#println("loss: ", this_loss)



# try

#    this_grad = gradient(loss, handcrafted)

# this_grad = gradient(loss, trainable)

#catch e
#    println("caught!", e)
#end

# println("grad: ", this_grad[1]["network_matrix"])

# convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
#         this_grad[1]["network_matrix"])



