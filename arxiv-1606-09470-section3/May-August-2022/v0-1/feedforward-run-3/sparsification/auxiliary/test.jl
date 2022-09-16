function loss_k(dmm_lite::DMM_Lite_, k_steps)
    l = 0.0f0
    for i in 1:k_steps # 1:12 # 1:35 ok speed (use 0.001 reg mult), 
	              # 1:140 works, but the slowdown is spectacular (use 0.01 reg mult)
        two_stroke_cycle!(dmm_lite)
        two_stroke_cycle!(handcrafted)
        target_1 = get_N(handcrafted["neurons"]["output"].input_dict["dict-1"], ":number")
        target_2 = get_N(handcrafted["neurons"]["output"].input_dict["dict-2"], ":number")
        l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-1"], ":number") - target_1)
        l += square(get_N(dmm_lite["neurons"]["output"].input_dict["dict-2"], ":number") - target_2)
    end
    reg_l1 = 0.0f0
	reg_l2 = 0.0f0
    for i in keys(dmm_lite["network_matrix"])
	    if i != "timer" && i != "input"
            for j in keys(dmm_lite["network_matrix"][i])
                for m in keys(dmm_lite["network_matrix"][i][j])
                    for n in keys(dmm_lite["network_matrix"][i][j][m])
                        reg_l1 += abs(dmm_lite["network_matrix"][i][j][m][n])
                        reg_l2 += square(dmm_lite["network_matrix"][i][j][m][n])
    end end end end end
	#=
	reg_novel = 0.0f0
    for i in keys(dmm_lite["network_matrix"])
        for j in keys(dmm_lite["network_matrix"][i])
            weights_sum = 0.0f0
            for m in keys(dmm_lite["network_matrix"][i][j])
                for n in keys(dmm_lite["network_matrix"][i][j][m])
                    weights_sum += dmm_lite["network_matrix"][i][j][m][n]
            end end
            reg_novel += square(weights_sum - 1.0f0)
    end end
	=#

    printlog_v(io, "prereg loss ", l, " reg_l1 ", reg_l1, " reg_l2 ", reg_l2)
    l += 0.1f0 * reg_l1 # + 0.001f0 * reg_l2	
	printlog_v(io, "loss ", l)
    l
end

# input_dummy (from a fixed string, depending on the timer input)
#
function input_dummy(x::Dict{String, Dict{String, Float32}})
    t::Float32 = get_N(get_D(x, "timer"), ":number")
    printlog(io, "(driving input) timer: ", t)
    t = max(t, 0)
    s::String = "tets. str.ingtb str.ingtb"
    d::Dict{String, Float32} = Dict{String, Float32}()
    if t%10 == 0
        i = min(round(Int, t÷10) + 1, lastindex(s))
        Zygote.@ignore d[SubString(s, i, i)] = one_value
    end
    Dict{String, Dict{String, Float32}}("char" => d)
end

trainable["network_matrix"] = deserialize("sparse20-after-100-steps-matrix.ser")