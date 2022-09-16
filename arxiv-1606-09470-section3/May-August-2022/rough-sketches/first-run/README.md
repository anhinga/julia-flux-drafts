# First successful training run, May 28-29, 2022

```julia
cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/rough-sketches")

include("train-v0-0-1.jl")

for i in 1:16
    println("STEP ", i, " ================================")
    training_step!()
end

# I am not 100% whether there has been another loop with length 16,
# my setup is still "less than perfect" in terms of reproducibity and such,
# but I am working to improve this aspect

for i in 1:500
    println("STEP ", i, " ================================")
    training_step!()
end
```

The training dynamics was quite interesting, rather unstable; I only
including the last 60 steps as file `tail-of-the-output.txt`.

The resulting network matrix is pretty-printed as `first-run.json`.

```julia
# To pretty-print (but recovery in exact same form is non-trivial, and it's not clear if any precision gets lost)

using JSON3

open("first-run.json", "w") do f
    JSON3.pretty(f, JSON3.write(a))
    println(f) # not really necessary
end

# To store-retrieve in binary (but compatibility between versions is not assured)

using Serialization

serialize("first-run.ser", a)

b = deserialize("first-run.ser") # types "b" correctly

a == b # returns true
```

Let's take a look at the resulting network matrix.

```julia
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
```

```
julia> count(b)
20308

julia> count_neg_interval(b, -0.8f0, 0.8f0)
2

julia> count_neg_interval(b, -0.8f0, 0.8f0, true)
timer timer timer timer 1.0
input timer timer timer 1.0
2

julia> count_neg_interval(b, -0.7f0, 0.7f0)
3

julia> count_neg_interval(b, -0.7f0, 0.7f0, true)
timer timer timer timer 1.0
input timer timer timer 1.0
accum-2 dict-2 const_1 const_1 -0.7053732
3

julia> count_neg_interval(b, -0.6f0, 0.6f0)
5

julia> count_neg_interval(b, -0.6f0, 0.6f0, true)
timer timer timer timer 1.0
accum-3 dict-2 norm-4 norm 0.6546106
input timer timer timer 1.0
accum-2 dict-2 const_1 const_1 -0.7053732
dot-5 dict-1 norm-2 norm 0.6443894
5

julia> count_neg_interval(b, -0.5f0, 0.5f0)
14

julia> count_neg_interval(b, -0.5f0, 0.5f0, true)
timer timer timer timer 1.0
output dict-2 norm-1 norm 0.51451546
output dict-1 norm-3 norm 0.5390576
dot-2 dict-1 input char 0.5743997
compare-5 dict-2 accum-4 dict -0.5037672
accum-3 dict-2 norm-4 norm 0.6546106
compare-4 dict-2 norm-2 norm -0.5951638
input timer timer timer 1.0
norm-3 dict accum-5 dict 0.5337626
accum-5 dict-2 compare-5 true 0.50991875
accum-5 dict-2 input char 0.56658715
accum-2 dict-2 const_1 const_1 -0.7053732
dot-4 dict-1 norm-3 norm 0.59004813
dot-5 dict-1 norm-2 norm 0.6443894
14

julia> count_neg_interval(b, -0.4f0, 0.4f0)
29

julia> count_neg_interval(b, -0.3f0, 0.3f0)
81

julia> count_neg_interval(b, -0.2f0, 0.2f0)
232

julia> count_neg_interval(b, -0.1f0, 0.1f0)
690

julia> count_neg_interval(b, -0.01f0, 0.01f0)
2171

julia> count_neg_interval(b, -0.001f0, 0.001f0)
2537

julia> count_neg_interval(b, -0.0001f0, 0.0001f0)
9151
```

One should be able to sparsify meaningfully based on this, but
we are really aiming towards a dozen non-zero links or so at the end
of the day.

What might make sense is to start with a smaller network
(and with a larger temporal interval) for initial experiments.

And then, of course, we want to end the practice of fitting
to one example; this was just a convergence study, we are aware
that more than one example is actually needed ;-)
