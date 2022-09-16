# Utilities for experiments with DMM training

function printlog(io, xs...)
    println(io, xs...)
    flush(io)
end

function printlog_v(io, xs...)
    printlog(io, xs...)
	println(xs...)
end

# ------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------

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







