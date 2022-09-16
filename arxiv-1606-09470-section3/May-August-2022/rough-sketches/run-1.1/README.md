# Run 1.1 June 2-3; the first run, well instrumented and extended

Currently at 1032 steps; this is still "work-in-progress"

```
julia> a_1032 = deepcopy(trainable["network_matrix"])
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
[...]

julia> count_neg_interval(a_1032, -0.8f0, 0.8f0)
2

julia> count_neg_interval(a_1032, -0.7f0, 0.7f0)
3

julia> count_neg_interval(a_1032, -0.6f0, 0.6f0)
4

julia> count_neg_interval(a_1032, -0.5f0, 0.5f0)
11

julia> count_neg_interval(a_1032, -0.5f0, 0.5f0, true)
timer timer timer timer 1.0
output dict-1 norm-3 norm 0.57676715
dot-2 dict-1 input char 0.56877714
accum-3 dict-2 norm-4 norm 0.6519098
compare-4 dict-2 norm-2 norm -0.5916672
input timer timer timer 1.0
norm-3 dict accum-5 dict 0.51802754
accum-5 dict-2 input char 0.5664605
accum-2 dict-2 const_1 const_1 -0.70896107
dot-4 dict-1 norm-3 norm 0.58391464
dot-5 dict-1 norm-2 norm 0.56162167
11

julia> count_neg_interval(a_1032, -0.4f0, 0.4f0)
27

julia> count_neg_interval(a_1032, -0.3f0, 0.3f0)
69

julia> count_neg_interval(a_1032, -0.2f0, 0.2f0)
174

julia> count_neg_interval(a_1032, -0.1f0, 0.1f0)
394

julia> count_neg_interval(a_1032, -0.01f0, 0.01f0)
770

julia> count_neg_interval(a_1032, -0.001f0, 0.001f0)
851

julia> count_neg_interval(a_1032, -0.0001f0, 0.0001f0)
8436
```

Results of sparsification experiments (added to console output and to log file):

EXPLORATORY ACTIVITY, cutoff 0.1f0 - NOT GOOD, cutoff 0.01f0 - STILL GOOD

As we know from Run 2.1, this many links is a very decent network sizewise, 
so we should be able to do true sparsification 
to the level of 0.01f0 or to the level of 0.001f0
and rapidly train for a longer BPTT from there.
