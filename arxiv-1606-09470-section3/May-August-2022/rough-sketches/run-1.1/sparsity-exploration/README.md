# A bit of post run-1.1 sparsity and structure exploration

The changed `train` script with added utilities will be included into this subdirectory

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/rough-sketches")

julia> include("train-v0-0-1.jl")

[...]

julia> a_1032 = deserialize("run-1.1/1032-steps-matrix.ser")

[...]

julia> sparse = sparsecopy(a_1032, 0.001f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0219264), "norm-5"=>Dict("norm"=>0.25096), "accum-4"…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.199862), "accum-2"=>Dict("dict"=>0.201067), "accum-1"=>Di…
  "accum-4"   => Dict("dict"=>Dict("dot-2"=>Dict("false"=>0.0193636)), "true"=>Dict("compare-3"=>Dict("false"=>-0.01210…
  "dot-2"     => Dict("dict"=>Dict("norm-3"=>Dict("true"=>-0.00654772)), "true"=>Dict("accum-4"=>Dict("true"=>0.0083629…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.190032), "norm-5"=>Dict("norm"=>0.0691165), "accum-4"=>…
  "compare-5" => Dict("true"=>Dict("norm-2"=>Dict("dict-1"=>0.0433679)), "dot"=>Dict("accum-4"=>Dict("dot"=>0.0208008))…
  "accum-3"   => Dict("false"=>Dict("compare-4"=>Dict("norm"=>-0.00810362)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.155827), "norm-5"=>Dict("norm"=>-0.326115), "accum-4"=>…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.0902416)), "true"=>Dict("accum-5"=>Dict("dict"=>0.0888…
  "compare-4" => Dict("dot"=>Dict("accum-2"=>Dict("dot"=>0.00907417)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0643…
  "compare-2" => Dict("dict"=>Dict("dot-1"=>Dict("dict-1"=>0.00865799)), "true"=>Dict("accum-2"=>Dict("norm"=>0.0028621…
  "dot-1"     => Dict("true"=>Dict("compare-2"=>Dict("norm"=>-0.0135199)), "dot"=>Dict("accum-3"=>Dict("norm"=>0.018773…
  "dot-3"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0996243), "norm-5"=>Dict("dot"=>0.0478396, "norm"=>-…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.180737), "norm-5"=>Dict("norm"=>0.18749), "accum-4"=>D…
  "compare-3" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0231354), "norm-5"=>Dict("norm"=>-0.112434), "accum-…
  "accum-5"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.30937), "norm-5"=>Dict("norm"=>-0.0241003), "accum-4"…
  "accum-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.708961), "norm-5"=>Dict("norm"=>0.24083), "accum-4"=…
  "compare-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.164622), "norm-5"=>Dict("norm"=>0.129442), "accum-4"…
  "dot-4"     => Dict("dict"=>Dict("accum-2"=>Dict("false"=>0.0189236)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>0.21…
  "dot-5"     => Dict("dot"=>Dict("compare-3"=>Dict("false"=>0.0143029)), "false"=>Dict("norm-3"=>Dict("dict-1"=>0.1208…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.443442), "norm-5"=>Dict("norm"=>-0.0127115), "accum-4"=…

julia> count(sparse)
851

julia> sparse["output"]
Dict{String, Dict{String, Dict{String, Float32}}} with 2 entries:
  "dict-2" => Dict("const_1"=>Dict("const_1"=>-0.0219264), "norm-5"=>Dict("norm"=>0.25096), "accum-4"=>Dict("dict"=>-0.…
  "dict-1" => Dict("const_1"=>Dict("const_1"=>-0.0250755), "norm-5"=>Dict("norm"=>-0.0878835), "accum-4"=>Dict("dict"=>…

julia> sparse["output"]["dict-1"]
Dict{String, Dict{String, Float32}} with 21 entries:
  "const_1"   => Dict("const_1"=>-0.0250755)
  "norm-5"    => Dict("norm"=>-0.0878835)
  "accum-4"   => Dict("dict"=>-0.0383879)
  "dot-2"     => Dict("dot"=>0.0245676, "false"=>0.0020161, "norm"=>-0.00393641)
  "norm-1"    => Dict("norm"=>-0.225004)
  "compare-5" => Dict("true"=>-0.0445922, "false"=>0.086028)
  "accum-3"   => Dict("dict"=>-0.143714)
  "norm-4"    => Dict("norm"=>0.207288)
  "accum-1"   => Dict("dict"=>-0.0230568)
  "compare-4" => Dict("true"=>0.0229342)
  "compare-2" => Dict("true"=>0.203662, "false"=>-0.0367009)
  "dot-1"     => Dict("dot"=>0.06133)
  "dot-3"     => Dict("dot"=>-0.0499203)
  "norm-3"    => Dict("norm"=>0.576767)
  "compare-3" => Dict("true"=>0.40439, "false"=>0.1701)
  "accum-5"   => Dict("dict"=>-0.0682158, "true"=>0.0106147)
  "accum-2"   => Dict("dict"=>0.171961)
  "compare-1" => Dict("true"=>0.0396978, "false"=>0.410133)
  "dot-4"     => Dict("dot"=>-0.252035)
  "dot-5"     => Dict("dot"=>-0.0528345)
  "norm-2"    => Dict("norm"=>0.0512389)

julia> handcrafted["network_matrix"]["output"]
Dict{String, Dict{String, Dict{String, Float32}}} with 2 entries:
  "dict-2" => Dict("dot"=>Dict("dot"=>1.0))
  "dict-1" => Dict("compare"=>Dict("true"=>1.0))

julia> sparse_01 = sparsecopy(a_1032, 0.01f0)
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 23 entries:
  "timer"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "output"    => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0219264), "norm-5"=>Dict("norm"=>0.25096), "accum-4"…
  "norm-5"    => Dict("dict"=>Dict("accum-5"=>Dict("dict"=>-0.199862), "accum-2"=>Dict("dict"=>0.201067), "accum-1"=>Di…
  "accum-4"   => Dict("dict"=>Dict("dot-2"=>Dict("false"=>0.0193636)), "true"=>Dict("compare-3"=>Dict("false"=>-0.01210…
  "dot-2"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.208796), "norm-5"=>Dict("norm"=>-0.0586771), "accum-…
  "norm-1"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.190032), "norm-5"=>Dict("norm"=>0.0691165), "accum-4"=>…
  "compare-5" => Dict("true"=>Dict("norm-2"=>Dict("dict-1"=>0.0433679)), "dot"=>Dict("accum-4"=>Dict("dot"=>0.0208008))…
  "accum-3"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.01858), "norm-5"=>Dict("norm"=>-0.0618115), "accum-4…
  "norm-4"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.155827), "norm-5"=>Dict("norm"=>-0.326115), "accum-4"=>…
  "accum-1"   => Dict("dict"=>Dict("compare-1"=>Dict("dict-1"=>0.0902416)), "true"=>Dict("accum-5"=>Dict("dict"=>0.0888…
  "compare-4" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0643311), "norm-5"=>Dict("norm"=>-0.163086), "accum-4…
  "compare-2" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.277083), "norm-5"=>Dict("norm"=>-0.0641934), "accum-…
  "dot-1"     => Dict("true"=>Dict("compare-2"=>Dict("norm"=>-0.0135199)), "dot"=>Dict("accum-3"=>Dict("norm"=>0.018773…
  "dot-3"     => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0996243), "norm-5"=>Dict("dot"=>0.0478396, "norm"=>-…
  "input"     => Dict("timer"=>Dict("timer"=>Dict("timer"=>1.0)))
  "norm-3"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.180737), "norm-5"=>Dict("norm"=>0.18749), "accum-4"=>D…
  "compare-3" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.0231354), "norm-5"=>Dict("norm"=>-0.112434), "accum-…
  "accum-5"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.30937), "norm-5"=>Dict("norm"=>-0.0241003), "accum-4"…
  "accum-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.708961), "norm-5"=>Dict("norm"=>0.24083), "accum-4"=…
  "compare-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-0.164622), "norm-5"=>Dict("norm"=>0.129442), "accum-4"…
  "dot-4"     => Dict("dict"=>Dict("accum-2"=>Dict("false"=>0.0189236)), "dict-2"=>Dict("const_1"=>Dict("const_1"=>0.21…
  "dot-5"     => Dict("dot"=>Dict("compare-3"=>Dict("false"=>0.0143029)), "false"=>Dict("norm-3"=>Dict("dict-1"=>0.1208…
  "norm-2"    => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>0.443442), "norm-5"=>Dict("norm"=>-0.0127115), "accum-4"=…

julia> count(sparse_01)
770

julia> sparse_01["output"]["dict-1"]
Dict{String, Dict{String, Float32}} with 21 entries:
  "const_1"   => Dict("const_1"=>-0.0250755)
  "norm-5"    => Dict("norm"=>-0.0878835)
  "accum-4"   => Dict("dict"=>-0.0383879)
  "dot-2"     => Dict("dot"=>0.0245676)
  "norm-1"    => Dict("norm"=>-0.225004)
  "compare-5" => Dict("true"=>-0.0445922, "false"=>0.086028)
  "accum-3"   => Dict("dict"=>-0.143714)
  "norm-4"    => Dict("norm"=>0.207288)
  "accum-1"   => Dict("dict"=>-0.0230568)
  "compare-4" => Dict("true"=>0.0229342)
  "compare-2" => Dict("true"=>0.203662, "false"=>-0.0367009)
  "dot-1"     => Dict("dot"=>0.06133)
  "dot-3"     => Dict("dot"=>-0.0499203)
  "norm-3"    => Dict("norm"=>0.576767)
  "compare-3" => Dict("true"=>0.40439, "false"=>0.1701)
  "accum-5"   => Dict("dict"=>-0.0682158, "true"=>0.0106147)
  "accum-2"   => Dict("dict"=>0.171961)
  "compare-1" => Dict("true"=>0.0396978, "false"=>0.410133)
  "dot-4"     => Dict("dot"=>-0.252035)
  "dot-5"     => Dict("dot"=>-0.0528345)
  "norm-2"    => Dict("norm"=>0.0512389)

julia> sparsecopy(a_1032, 0.1f0)["output"]["dict-1"]
Dict{String, Dict{String, Float32}} with 9 entries:
  "compare-3" => Dict("true"=>0.40439, "false"=>0.1701)
  "accum-2"   => Dict("dict"=>0.171961)
  "compare-1" => Dict("false"=>0.410133)
  "dot-4"     => Dict("dot"=>-0.252035)
  "compare-2" => Dict("true"=>0.203662)
  "accum-3"   => Dict("dict"=>-0.143714)
  "norm-4"    => Dict("norm"=>0.207288)
  "norm-3"    => Dict("norm"=>0.576767)
  "norm-1"    => Dict("norm"=>-0.225004)

julia> sparsecopy(a_1032, 0.2f0)["output"]["dict-1"]
Dict{String, Dict{String, Float32}} with 7 entries:
  "compare-3" => Dict("true"=>0.40439)
  "compare-1" => Dict("false"=>0.410133)
  "dot-4"     => Dict("dot"=>-0.252035)
  "compare-2" => Dict("true"=>0.203662)
  "norm-4"    => Dict("norm"=>0.207288)
  "norm-3"    => Dict("norm"=>0.576767)
  "norm-1"    => Dict("norm"=>-0.225004)

julia> sparsecopy(a_1032, 0.3f0)["output"]["dict-1"]
Dict{String, Dict{String, Float32}} with 3 entries:
  "compare-3" => Dict("true"=>0.40439)
  "compare-1" => Dict("false"=>0.410133)
  "norm-3"    => Dict("norm"=>0.576767)

julia> sparsecopy(a_1032, 0.3f0)["output"]["dict-2"]
Dict{String, Dict{String, Float32}} with 2 entries:
  "dot-5"  => Dict("dot"=>-0.409062)
  "norm-1" => Dict("norm"=>0.487708)

julia> sparsecopy(a_1032, 0.2f0)["output"]["dict-2"]
Dict{String, Dict{String, Float32}} with 5 entries:
  "compare-3" => Dict("false"=>-0.206871)
  "accum-5"   => Dict("dict"=>0.236125)
  "dot-5"     => Dict("dot"=>-0.409062)
  "norm-5"    => Dict("norm"=>0.25096)
  "norm-1"    => Dict("norm"=>0.487708)

julia> sparsecopy(a_1032, 0.1f0)["output"]["dict-2"]
Dict{String, Dict{String, Float32}} with 13 entries:
  "norm-5"    => Dict("norm"=>0.25096)
  "accum-4"   => Dict("dict"=>-0.135338)
  "dot-2"     => Dict("dot"=>0.101428)
  "norm-1"    => Dict("norm"=>0.487708)
  "compare-5" => Dict("true"=>0.143606)
  "accum-3"   => Dict("dict"=>-0.101197)
  "compare-4" => Dict("true"=>-0.166904)
  "compare-2" => Dict("true"=>-0.103596)
  "dot-3"     => Dict("dot"=>-0.126692)
  "compare-3" => Dict("true"=>0.129184, "false"=>-0.206871)
  "accum-5"   => Dict("dict"=>0.236125)
  "accum-2"   => Dict("dict"=>0.181333)
  "dot-5"     => Dict("dot"=>-0.409062)

julia> sparsecopy(a_1032, 0.01f0)["output"]["dict-2"]
Dict{String, Dict{String, Float32}} with 21 entries:
  "const_1"   => Dict("const_1"=>-0.0219264)
  "norm-5"    => Dict("norm"=>0.25096)
  "accum-4"   => Dict("dict"=>-0.135338)
  "dot-2"     => Dict("dot"=>0.101428)
  "norm-1"    => Dict("norm"=>0.487708)
  "compare-5" => Dict("true"=>0.143606, "false"=>-0.0121215)
  "accum-3"   => Dict("dict"=>-0.101197)
  "norm-4"    => Dict("norm"=>-0.0996024)
  "accum-1"   => Dict("dict"=>0.0263056)
  "compare-4" => Dict("true"=>-0.166904)
  "compare-2" => Dict("true"=>-0.103596, "false"=>0.0581516)
  "dot-1"     => Dict("dot"=>-0.0556719)
  "dot-3"     => Dict("dot"=>-0.126692)
  "norm-3"    => Dict("norm"=>0.0146557)
  "compare-3" => Dict("true"=>0.129184, "false"=>-0.206871)
  "accum-5"   => Dict("dict"=>0.236125)
  "accum-2"   => Dict("dict"=>0.181333)
  "compare-1" => Dict("true"=>0.0455713, "false"=>0.0538826)
  "dot-4"     => Dict("dot"=>0.0319234)
  "dot-5"     => Dict("dot"=>-0.409062)
  "norm-2"    => Dict("norm"=>0.0597156)

julia> sparsecopy(a_1032, 0.001f0)["output"]["dict-2"]
Dict{String, Dict{String, Float32}} with 21 entries:
  "const_1"   => Dict("const_1"=>-0.0219264)
  "norm-5"    => Dict("norm"=>0.25096)
  "accum-4"   => Dict("dict"=>-0.135338)
  "dot-2"     => Dict("dot"=>0.101428)
  "norm-1"    => Dict("norm"=>0.487708)
  "compare-5" => Dict("true"=>0.143606, "false"=>-0.0121215)
  "accum-3"   => Dict("dict"=>-0.101197)
  "norm-4"    => Dict("norm"=>-0.0996024)
  "accum-1"   => Dict("dict"=>0.0263056)
  "compare-4" => Dict("true"=>-0.166904)
  "compare-2" => Dict("true"=>-0.103596, "false"=>0.0581516)
  "dot-1"     => Dict("dot"=>-0.0556719)
  "dot-3"     => Dict("dot"=>-0.126692)
  "norm-3"    => Dict("norm"=>0.0146557)
  "compare-3" => Dict("true"=>0.129184, "false"=>-0.206871)
  "accum-5"   => Dict("dict"=>0.236125)
  "accum-2"   => Dict("dict"=>0.181333)
  "compare-1" => Dict("true"=>0.0455713, "false"=>0.0538826)
  "dot-4"     => Dict("dot"=>0.0319234)
  "dot-5"     => Dict("dot"=>-0.409062)
  "norm-2"    => Dict("norm"=>0.0597156)

julia>
```

This also does not generalize past step 34, so it's time to do extended time window BPTT learning
on the semi-sparse system:

```
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 0.0
(getting on output) left: -0.025075538 right: -0.021926384
(driving input) timer: 1.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 1.0
(getting on output) left: 0.0020041186 right: 0.009932263
(driving input) timer: 2.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 2.0
(getting on output) left: 0.015913744 right: -0.00063895807
(driving input) timer: 3.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 3.0
(getting on output) left: -0.020521116 right: 0.009257581
(driving input) timer: 4.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 4.0
(getting on output) left: -0.00096144597 right: 0.0058544655
(driving input) timer: 5.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 5.0
(getting on output) left: 0.0040411367 right: -0.0080979075
(driving input) timer: 6.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 6.0
(getting on output) left: -0.020025926 right: 0.0010337958
(driving input) timer: 7.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 7.0
(getting on output) left: 0.017689684 right: -0.010965809
(driving input) timer: 8.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 8.0
(getting on output) left: 0.0666378 right: -0.00633436
(driving input) timer: 9.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 9.0
(getting on output) left: 0.031941842 right: -0.005156995
(driving input) timer: 10.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 10.0
(getting on output) left: -0.009956786 right: -0.017300656
(driving input) timer: 11.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 11.0
(getting on output) left: 0.0062990077 right: -0.00915846
(driving input) timer: 12.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 12.0
(getting on output) left: -0.005069119 right: -0.0024101082
(driving input) timer: 13.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 13.0
(getting on output) left: -0.002656617 right: 0.009366995
(driving input) timer: 14.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 14.0
(getting on output) left: -0.004641697 right: 0.012403952
(driving input) timer: 15.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 15.0
(getting on output) left: -0.0048153168 right: -0.0016102169
(driving input) timer: 16.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 16.0
(getting on output) left: -0.0025583198 right: -0.00093469676
(driving input) timer: 17.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 17.0
(getting on output) left: -0.007602124 right: -0.0057386723
(driving input) timer: 18.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 18.0
(getting on output) left: -0.022499738 right: -0.009449473
(driving input) timer: 19.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 19.0
(getting on output) left: -0.012498598 right: -0.0061179437
(driving input) timer: 20.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 20.0
(getting on output) left: -0.012645656 right: -0.0046085496
(driving input) timer: 21.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 21.0
(getting on output) left: -0.010834288 right: 0.0064888336
(driving input) timer: 22.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 22.0
(getting on output) left: -0.02786861 right: 0.007283493
(driving input) timer: 23.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 23.0
(getting on output) left: 0.06253914 right: -0.003705366
(driving input) timer: 24.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 24.0
(getting on output) left: 0.02039719 right: 0.011977442
(driving input) timer: 25.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 25.0
(getting on output) left: -0.012855141 right: 0.0059404597
(driving input) timer: 26.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 26.0
(getting on output) left: 0.0014753575 right: 0.0014176187
(driving input) timer: 27.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 27.0
(getting on output) left: -0.005510799 right: 0.0021167956
(driving input) timer: 28.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 28.0
(getting on output) left: -0.012667151 right: 0.0035380418
(driving input) timer: 29.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 29.0
(getting on output) left: -0.010659646 right: -0.0015875695
(driving input) timer: 30.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 30.0
(getting on output) left: -0.017519237 right: -0.013008345
(driving input) timer: 31.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 31.0
(getting on output) left: 0.012411004 right: -0.0029057432
(driving input) timer: 32.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 32.0
(getting on output) left: 0.011513874 right: 0.0071037654
(driving input) timer: 33.0
(getting on output) left: 0.0 right: 0.0
(driving input) timer: 33.0
(getting on output) left: 0.953433 right: 0.00014536176
(driving input) timer: 34.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 34.0
(getting on output) left: 0.22149149 right: -0.036859177
(driving input) timer: 35.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 35.0
(getting on output) left: -0.02539567 right: -0.040791538
(driving input) timer: 36.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 36.0
(getting on output) left: 0.026665047 right: -0.0071146227
(driving input) timer: 37.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 37.0
(getting on output) left: 0.010851711 right: 0.0121488925
(driving input) timer: 38.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 38.0
(getting on output) left: 0.009854076 right: 0.015862051
(driving input) timer: 39.0
(getting on output) left: 1.0 right: 0.0
(driving input) timer: 39.0
prereg loss 4.5865316 regularization 335.15085
loss 4.9216824
```

and here we can try to train "all at once" for a rather long time window in BPTT,
or we can try curriculum learning.
