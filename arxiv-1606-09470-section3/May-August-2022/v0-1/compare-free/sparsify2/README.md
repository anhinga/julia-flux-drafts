# Redo the sparsify experiment

In the process we discovered that Zygote.jl 0.6.41-0.6.43 does not work for our tasks,
so we had to revert to v0.6.40 used elsewhere (TODO item: fix the problems with updated Zygote.jl)

Reproducing the first 5 steps from `../sparsify` with small regularization (very close, but there are
minor differences):

```
julia> sparsifying_steps!(5)
2022-08-14T09:43:15.580
STEP 1 ================================
prereg loss 0.10171912 reg_l1 85.602264 reg_l2 26.10721
loss 0.1873214
cutoff 2.6698785e-9 network size 941
STEP 2 ================================
prereg loss 0.10161547 reg_l1 85.601875 reg_l2 26.108368
loss 0.18721735
cutoff 1.1145616e-7 network size 940
STEP 3 ================================
prereg loss 0.10137128 reg_l1 85.601395 reg_l2 26.109524
loss 0.18697268
cutoff 3.0488877e-7 network size 939
STEP 4 ================================
prereg loss 0.10104765 reg_l1 85.60012 reg_l2 26.110575
loss 0.18664777
cutoff 9.439236e-8 network size 938
STEP 5 ================================
prereg loss 0.10092122 reg_l1 85.59957 reg_l2 26.11157
loss 0.1865208
cutoff 3.1372474e-7 network size 937
2022-08-14T09:46:07.109
```

Now, let's go back to the intended regularization.

What we are immediately seeing is that that regularization might be too high
at this early stage of sparsification. We really need to develop a more
adaptive training, perhaps with limited backtracking.

That's an important major TODO task.

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/compare-free/sparsify2")

julia> include("prepare.jl")
FEEDFORWARD with local recurrence and skip connections INCLUDED
DEFINED: opt
SKIPPED: adam_step!
The network is ready to train, use 'steps!(N)'

julia> trainable["network_matrix"] = deserialize("cf-2500-steps-matrix.ser")
Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}} with 37 entries:
  "dot-1-1"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>-8.98799f-5), "input"=>Dict("char"=>-0.121564), "eos"=>…
  "norm-5-2"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.205666), "norm-2-1"=>Dict("norm"=>-0.332586), "dot-2-2"=>Di…
  "norm-2-1"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.157434), "dot-1-2"=>Dict("dot"=>0.105914), "const_1"=>Dict(…
  "dot-2-2"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.614051), "dot-1-2"=>Dict("dot"=>-0.624078), "const_1"=>D…
  "norm-3-1"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.199802), "norm-2-1"=>Dict("norm"=>0.0269759), "dot-2-2"=>Di…
  "accum-3-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>1.70603f-5), "norm-2-1"=>Dict("norm"=>-0.00449996), "dot-2-…
  "dot-3-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.25989), "norm-2-1"=>Dict("norm"=>0.00263599), "dot-2-2"=>…
  "norm-4-2"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.0384275), "accum-3-2"=>Dict("dict"=>-0.0380184), "norm-2-1"…
  "norm-5-1"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>0.544922), "norm-2-1"=>Dict("norm"=>0.122575), "dot-2-2"=>Dic…
  "accum-1-1" => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.0274521), "input"=>Dict("char"=>1.41784), "eos"=>Dict…
  "dot-5-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.223367), "norm-2-1"=>Dict("norm"=>-0.0316851), "dot-2-2"…
  "norm-1-2"  => Dict("dict"=>Dict("const_1"=>Dict("const_1"=>-0.102733), "input"=>Dict("char"=>0.000458776), "eos"=>Di…
  "norm-4-1"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.232692), "accum-3-2"=>Dict("dict"=>-0.0238939), "norm-2-1"…
  "accum-6-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.226458), "norm-5-2"=>Dict("norm"=>0.0347561), "norm-2-1"=…
  "dot-6-2"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.0531279), "norm-5-2"=>Dict("norm"=>-0.0495344), "norm-2-…
  "output"    => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0636645), "norm-5-2"=>Dict("norm"=>-0.0144621), "norm-2-1…
  "dot-1-2"   => Dict("dict-2"=>Dict("const_1"=>Dict("const_1"=>0.3258), "input"=>Dict("char"=>-0.0847738), "eos"=>Dict…
  "norm-2-2"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.000184463), "dot-1-2"=>Dict("dot"=>-2.50155f-6), "const_1"…
  "norm-6-1"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-0.000156793), "norm-5-2"=>Dict("norm"=>0.00916407), "norm-2-…
  "dot-2-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.504061), "dot-1-2"=>Dict("dot"=>-0.527366), "const_1"=>D…
  "accum-5-1" => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>8.76796f-6), "norm-2-1"=>Dict("norm"=>-8.24491f-6), "dot-2-…
  "norm-6-2"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-3.70854f-5), "norm-5-2"=>Dict("norm"=>-0.250683), "norm-2-1"…
  "dot-6-1"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>0.0206005), "norm-5-2"=>Dict("norm"=>-0.0554821), "norm-2-1…
  "norm-3-2"  => Dict("dict"=>Dict("dot-1-1"=>Dict("dot"=>-5.33715f-5), "norm-2-1"=>Dict("norm"=>0.0539498), "dot-2-2"=…
  "dot-4-2"   => Dict("dict-2"=>Dict("dot-1-1"=>Dict("dot"=>-0.17969), "accum-3-2"=>Dict("dict"=>-0.00833335), "norm-2-…
  ⋮           => ⋮

julia> opt = deserialize("cf-2500-steps-opt.ser")
TreeADAM(0.001f0, (0.9f0, 0.999f0), 1.0f-8, Float32[6.0f-45, 0.08190312], Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}("dot-1-1" => Dict("dict-2" => Dict("const_1" => Dict("const_1" => -5.770004f-5), "input" => Dict("char" => -0.0034075317), "eos" => Dict("char" => 0.0070785033)), "dict-1" => Dict("const_1" => Dict("const_1" => 9.797325f-5), "input" => Dict("char" => 0.0032413367), "eos" => Dict("char" => -0.0061501213))), "accum-6-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.0009847032), "norm-5-2" => Dict("norm" => 0.0024163178), "dot-2-2" => Dict("dot" => 0.00024581177), "norm-4-2" => Dict("norm" => -0.0010153516), "norm-3-1" => Dict("norm" => 0.0012299655), "accum-3-1" => Dict("dict" => -0.00053636543), "dot-3-1" => Dict("dot" => -0.0007385948), "const_1" => Dict("const_1" => -0.0009752396), "norm-5-1" => Dict("norm" => 0.0021950426), "dot-5-1" => Dict("dot" => -0.0003046066)…)), "dot-2-2" => Dict("dict-2" => Dict("dot-1-2" => Dict("dot" => -0.0010488313), "dot-1-1" => Dict("dot" => -0.0009843674), "const_1" => Dict("const_1" => 0.0010180117), "accum-1-1" => Dict("dict" => 0.05430817), "accum-1-2" => Dict("dict" => -0.0037326328), "norm-1-2" => Dict("norm" => 1.0072916f-5), "input" => Dict("char" => -0.016293708), "norm-1-1" => Dict("norm" => 0.00093402254), "eos" => Dict("char" => 0.0044985963)), "dict-1" => Dict("dot-1-2" => Dict("dot" => -0.0009947978), "dot-1-1" => Dict("dot" => -0.0009447557), "const_1" => Dict("const_1" => 0.0008618103), "accum-1-1" => Dict("dict" => 0.038395487), "accum-1-2" => Dict("dict" => 0.024491984), "norm-1-2" => Dict("norm" => 0.00093516964), "input" => Dict("char" => -0.02251342), "norm-1-1" => Dict("norm" => 0.0009187304), "eos" => Dict("char" => 0.006501956))), "norm-4-2" => Dict("dict" => Dict("dot-1-1" => Dict("dot" => 0.0007059941), "accum-3-2" => Dict("dict" => 0.11560786), "dot-2-2" => Dict("dot" => 0.0070980387), "norm-3-1" => Dict("norm" => 0.0056576156), "const_1" => Dict("const_1" => 0.0012945484), "accum-3-1" => Dict("dict" => 0.0004793811), "dot-3-1" => Dict("dot" => 0.0051138382), "norm-2-1" => Dict("norm" => 0.015503305), "eos" => Dict("char" => -0.009476785), "accum-1-1" => Dict("dict" => 0.019104902)…)), "norm-3-1" => Dict("dict" => Dict("dot-1-1" => Dict("dot" => -0.00030199828), "dot-2-2" => Dict("dot" => -0.006552912), "norm-2-1" => Dict("norm" => 0.004872574), "const_1" => Dict("const_1" => 0.004237505), "accum-1-1" => Dict("dict" => 0.0145171955), "norm-1-2" => Dict("norm" => -0.0004625025), "accum-2-1" => Dict("dict" => 0.013413459), "input" => Dict("char" => 0.00046299724), "dot-1-2" => Dict("dot" => -0.0008227276), "norm-2-2" => Dict("norm" => -0.011420894)…)), "output" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.0029012521), "accum-6-1" => Dict("dict" => -0.19943579), "dot-2-2" => Dict("dot" => 0.045536682), "norm-4-2" => Dict("norm" => -0.005702149), "norm-3-1" => Dict("norm" => 0.017490774), "accum-3-1" => Dict("dict" => -0.22631644), "dot-3-1" => Dict("dot" => 0.12396659), "const_1" => Dict("const_1" => -0.011789131), "norm-5-1" => Dict("norm" => 0.077658266), "dot-5-1" => Dict("dot" => 0.044922896)…), "dict-1" => Dict("dot-1-1" => Dict("dot" => -0.008959373), "accum-6-1" => Dict("dict" => -0.1811989), "dot-2-2" => Dict("dot" => 0.042875126), "norm-4-2" => Dict("norm" => 0.01745743), "norm-3-1" => Dict("norm" => 0.0027749185), "accum-3-1" => Dict("dict" => -0.21987179), "dot-3-1" => Dict("dot" => 0.10318519), "const_1" => Dict("const_1" => -0.002269214), "norm-5-1" => Dict("norm" => -0.0014256239), "dot-5-1" => Dict("dot" => -0.01119396)…)), "dot-3-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.0010563704), "dot-2-2" => Dict("dot" => -0.003030886), "norm-2-1" => Dict("norm" => -0.00079348934), "const_1" => Dict("const_1" => -0.00020989217), "accum-1-1" => Dict("dict" => -0.03873794), "norm-1-2" => Dict("norm" => -0.00019985341), "accum-2-1" => Dict("dict" => -1.9289553f-5), "input" => Dict("char" => -0.0017593454), "dot-1-2" => Dict("dot" => 0.0010817274), "norm-2-2" => Dict("norm" => 0.0008913665)…), "dict-1" => Dict("dot-1-1" => Dict("dot" => 0.0009791513), "dot-2-2" => Dict("dot" => -0.005224302), "norm-2-1" => Dict("norm" => -0.0007537239), "const_1" => Dict("const_1" => 6.4192456f-5), "accum-1-1" => Dict("dict" => -0.046739686), "norm-1-2" => Dict("norm" => 2.7150332f-5), "accum-2-1" => Dict("dict" => -0.020923685), "input" => Dict("char" => -0.0014608142), "dot-1-2" => Dict("dot" => 0.0009662804), "norm-2-2" => Dict("norm" => 0.0011845732)…)), "accum-3-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => -3.2801967f-5), "dot-2-2" => Dict("dot" => 0.00016074619), "norm-2-1" => Dict("norm" => 0.00039139445), "const_1" => Dict("const_1" => 0.00015367367), "accum-1-1" => Dict("dict" => -0.00078275736), "norm-1-2" => Dict("norm" => 0.0001414665), "accum-2-1" => Dict("dict" => -0.011617277), "input" => Dict("char" => 0.0018383897), "dot-1-2" => Dict("dot" => -0.00013978186), "norm-2-2" => Dict("norm" => 0.00035200908)…)), "norm-5-1" => Dict("dict" => Dict("dot-1-1" => Dict("dot" => 0.0004986307), "dot-2-2" => Dict("dot" => -0.0061848843), "norm-4-2" => Dict("norm" => -0.004616751), "norm-3-1" => Dict("norm" => 0.02509747), "accum-3-1" => Dict("dict" => 0.031205818), "dot-3-1" => Dict("dot" => -0.016132232), "const_1" => Dict("const_1" => -0.00038367353), "norm-2-1" => Dict("norm" => -0.013276614), "accum-1-1" => Dict("dict" => 0.007253157), "norm-4-1" => Dict("norm" => 0.019791685)…)), "dot-5-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => -0.0015103583), "dot-2-2" => Dict("dot" => -0.0030117887), "norm-4-2" => Dict("norm" => -0.01147454), "norm-3-1" => Dict("norm" => 0.003377399), "accum-3-1" => Dict("dict" => -0.15766364), "dot-3-1" => Dict("dot" => -0.002218132), "const_1" => Dict("const_1" => 0.0013927267), "norm-2-1" => Dict("norm" => 0.0013952726), "accum-1-1" => Dict("dict" => -0.017411789), "norm-4-1" => Dict("norm" => 0.010934227)…), "dict-1" => Dict("dot-1-1" => Dict("dot" => 0.00046246327), "dot-2-2" => Dict("dot" => -0.009528492), "norm-4-2" => Dict("norm" => 0.0067870007), "norm-3-1" => Dict("norm" => 0.0035122526), "accum-3-1" => Dict("dict" => -0.036472004), "dot-3-1" => Dict("dot" => 0.0033786648), "const_1" => Dict("const_1" => 0.004014804), "norm-2-1" => Dict("norm" => 0.00219474), "accum-1-1" => Dict("dict" => 0.011787247), "norm-4-1" => Dict("norm" => -0.0011674598)…))…), Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}("dot-1-1" => Dict("dict-2" => Dict("const_1" => Dict("const_1" => 1.9433992f-6), "input" => Dict("char" => 0.007906012), "eos" => Dict("char" => 1.1157497)), "dict-1" => Dict("const_1" => Dict("const_1" => 1.5914783f-6), "input" => Dict("char" => 0.010333488), "eos" => Dict("char" => 0.99683183))), "accum-6-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.0001295852), "norm-5-2" => Dict("norm" => 0.0428737), "dot-2-2" => Dict("dot" => 0.009204429), "norm-4-2" => Dict("norm" => 0.018533068), "norm-3-1" => Dict("norm" => 0.0040413863), "accum-3-1" => Dict("dict" => 1.6547575), "dot-3-1" => Dict("dot" => 0.021755667), "const_1" => Dict("const_1" => 0.007588342), "norm-5-1" => Dict("norm" => 0.020440884), "dot-5-1" => Dict("dot" => 0.0024414256)…)), "dot-2-2" => Dict("dict-2" => Dict("dot-1-2" => Dict("dot" => 0.0073624006), "dot-1-1" => Dict("dot" => 0.003714381), "const_1" => Dict("const_1" => 0.077038966), "accum-1-1" => Dict("dict" => 141.80067), "accum-1-2" => Dict("dict" => 55.562115), "norm-1-2" => Dict("norm" => 0.0005590872), "input" => Dict("char" => 0.010926552), "norm-1-1" => Dict("norm" => 0.00084992324), "eos" => Dict("char" => 0.93739355)), "dict-1" => Dict("dot-1-2" => Dict("dot" => 0.0070384177), "dot-1-1" => Dict("dot" => 0.0035924343), "const_1" => Dict("const_1" => 0.079647265), "accum-1-1" => Dict("dict" => 58.93405), "accum-1-2" => Dict("dict" => 370.20984), "norm-1-2" => Dict("norm" => 0.0005256647), "input" => Dict("char" => 0.04920384), "norm-1-1" => Dict("norm" => 0.00079320784), "eos" => Dict("char" => 4.5418944))), "norm-4-2" => Dict("dict" => Dict("dot-1-1" => Dict("dot" => 0.0033704527), "accum-3-2" => Dict("dict" => 109.9838), "dot-2-2" => Dict("dot" => 0.1885498), "norm-3-1" => Dict("norm" => 0.062184777), "const_1" => Dict("const_1" => 0.06500132), "accum-3-1" => Dict("dict" => 31.607635), "dot-3-1" => Dict("dot" => 0.5390998), "norm-2-1" => Dict("norm" => 0.011854542), "eos" => Dict("char" => 0.0022486502), "accum-1-1" => Dict("dict" => 0.38925573)…)), "norm-3-1" => Dict("dict" => Dict("dot-1-1" => Dict("dot" => 0.054977458), "dot-2-2" => Dict("dot" => 0.35474768), "norm-2-1" => Dict("norm" => 0.09519574), "const_1" => Dict("const_1" => 1.2106253), "accum-1-1" => Dict("dict" => 2.165502), "norm-1-2" => Dict("norm" => 0.00782644), "accum-2-1" => Dict("dict" => 620.54834), "input" => Dict("char" => 2.3209575f-5), "dot-1-2" => Dict("dot" => 0.10912487), "norm-2-2" => Dict("norm" => 2.643122)…)), "output" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.6443376), "accum-6-1" => Dict("dict" => 15262.182), "dot-2-2" => Dict("dot" => 38.77018), "norm-4-2" => Dict("norm" => 103.95623), "norm-3-1" => Dict("norm" => 14.294256), "accum-3-1" => Dict("dict" => 7296.86), "dot-3-1" => Dict("dot" => 111.15902), "const_1" => Dict("const_1" => 31.930878), "norm-5-1" => Dict("norm" => 93.08051), "dot-5-1" => Dict("dot" => 11.709267)…), "dict-1" => Dict("dot-1-1" => Dict("dot" => 1.2716154), "accum-6-1" => Dict("dict" => 28567.49), "dot-2-2" => Dict("dot" => 57.95102), "norm-4-2" => Dict("norm" => 154.3568), "norm-3-1" => Dict("norm" => 31.622713), "accum-3-1" => Dict("dict" => 11945.347), "dot-3-1" => Dict("dot" => 178.88228), "const_1" => Dict("const_1" => 110.96921), "norm-5-1" => Dict("norm" => 155.04694), "dot-5-1" => Dict("dot" => 24.311378)…)), "dot-3-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.004135021), "dot-2-2" => Dict("dot" => 0.03302159), "norm-2-1" => Dict("norm" => 0.01004145), "const_1" => Dict("const_1" => 0.08668085), "accum-1-1" => Dict("dict" => 6.640558), "norm-1-2" => Dict("norm" => 0.00068335474), "accum-2-1" => Dict("dict" => 406.45474), "input" => Dict("char" => 0.0002873331), "dot-1-2" => Dict("dot" => 0.008395449), "norm-2-2" => Dict("norm" => 0.20557758)…), "dict-1" => Dict("dot-1-1" => Dict("dot" => 0.005892867), "dot-2-2" => Dict("dot" => 0.06335916), "norm-2-1" => Dict("norm" => 0.017511135), "const_1" => Dict("const_1" => 0.1295575), "accum-1-1" => Dict("dict" => 10.269936), "norm-1-2" => Dict("norm" => 0.00096682325), "accum-2-1" => Dict("dict" => 950.0775), "input" => Dict("char" => 0.00052105787), "dot-1-2" => Dict("dot" => 0.011916737), "norm-2-2" => Dict("norm" => 0.2944866)…)), "accum-3-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 5.2178533f-5), "dot-2-2" => Dict("dot" => 0.0007241071), "norm-2-1" => Dict("norm" => 0.00021270548), "const_1" => Dict("const_1" => 0.018016888), "accum-1-1" => Dict("dict" => 0.14455467), "norm-1-2" => Dict("norm" => 1.4384026f-5), "accum-2-1" => Dict("dict" => 2.4332786), "input" => Dict("char" => 5.918561f-6), "dot-1-2" => Dict("dot" => 9.90023f-5), "norm-2-2" => Dict("norm" => 0.0028123483)…)), "norm-5-1" => Dict("dict" => Dict("dot-1-1" => Dict("dot" => 0.0030156788), "dot-2-2" => Dict("dot" => 1.6577742), "norm-4-2" => Dict("norm" => 3.1524215), "norm-3-1" => Dict("norm" => 0.09532637), "accum-3-1" => Dict("dict" => 96.12517), "dot-3-1" => Dict("dot" => 2.0831807), "const_1" => Dict("const_1" => 0.07741297), "norm-2-1" => Dict("norm" => 0.048369467), "accum-1-1" => Dict("dict" => 0.9424521), "norm-4-1" => Dict("norm" => 0.21086793)…)), "dot-5-1" => Dict("dict-2" => Dict("dot-1-1" => Dict("dot" => 0.022858806), "dot-2-2" => Dict("dot" => 1.3525406), "norm-4-2" => Dict("norm" => 3.772573), "norm-3-1" => Dict("norm" => 0.61242646), "accum-3-1" => Dict("dict" => 1219.6967), "dot-3-1" => Dict("dot" => 4.995826), "const_1" => Dict("const_1" => 0.5802993), "norm-2-1" => Dict("norm" => 0.14835563), "accum-1-1" => Dict("dict" => 18.433651), "norm-4-1" => Dict("norm" => 2.4541774)…), "dict-1" => Dict("dot-1-1" => Dict("dot" => 0.0004969726), "dot-2-2" => Dict("dot" => 0.22621252), "norm-4-2" => Dict("norm" => 0.40242723), "norm-3-1" => Dict("norm" => 0.02548213), "accum-3-1" => Dict("dict" => 77.87398), "dot-3-1" => Dict("dot" => 0.27263784), "const_1" => Dict("const_1" => 0.012915636), "norm-2-1" => Dict("norm" => 0.007444395), "accum-1-1" => Dict("dict" => 15.728376), "norm-4-1" => Dict("norm" => 0.06735789)…))…))

julia> sparsifying_steps!(100)
2022-08-14T10:03:55.088
STEP 1 ================================
prereg loss 0.10171912 reg_l1 85.602264 reg_l2 26.10721
loss 8.661945
cutoff 1.1295062e-5 network size 941
STEP 2 ================================
prereg loss 0.10318087 reg_l1 85.64886 reg_l2 26.077137
loss 8.668067
cutoff 2.3707296e-5 network size 940
STEP 3 ================================
prereg loss 0.10346201 reg_l1 85.47473 reg_l2 26.024055
loss 8.650935
cutoff 5.573893e-7 network size 939
STEP 4 ================================
prereg loss 0.105214715 reg_l1 85.15016 reg_l2 25.952309
loss 8.620232
cutoff 1.9082525e-5 network size 938
STEP 5 ================================
prereg loss 0.108139545 reg_l1 84.98379 reg_l2 25.866072
loss 8.606519
cutoff 1.9282888e-6 network size 937
STEP 6 ================================
prereg loss 0.111999005 reg_l1 84.76101 reg_l2 25.767862
loss 8.5880995
cutoff 2.0802716e-5 network size 936
STEP 7 ================================
prereg loss 0.11406999 reg_l1 84.427574 reg_l2 25.659489
loss 8.556828
cutoff 1.2051896e-6 network size 935
STEP 8 ================================
prereg loss 0.11827767 reg_l1 83.99721 reg_l2 25.54292
loss 8.517999
cutoff 1.9990111e-6 network size 934
STEP 9 ================================
prereg loss 0.12042808 reg_l1 83.56413 reg_l2 25.420298
loss 8.476842
cutoff 5.594251e-6 network size 933
STEP 10 ================================
prereg loss 0.12701333 reg_l1 83.22123 reg_l2 25.293112
loss 8.449137
cutoff 4.2478114e-6 network size 932
STEP 11 ================================
prereg loss 0.12988083 reg_l1 82.80264 reg_l2 25.16201
loss 8.410146
cutoff 1.7140119e-6 network size 931
STEP 12 ================================
prereg loss 0.13562666 reg_l1 82.34225 reg_l2 25.027884
loss 8.369852
cutoff 5.8015576e-7 network size 930
STEP 13 ================================
prereg loss 0.13696273 reg_l1 81.96913 reg_l2 24.891727
loss 8.333877
cutoff 5.3211443e-6 network size 929
STEP 14 ================================
prereg loss 0.14400303 reg_l1 81.57896 reg_l2 24.754387
loss 8.301899
cutoff 7.288869e-6 network size 928
STEP 15 ================================
prereg loss 0.14581019 reg_l1 81.150345 reg_l2 24.616474
loss 8.260845
cutoff 2.696208e-6 network size 927
STEP 16 ================================
prereg loss 0.1474297 reg_l1 80.700096 reg_l2 24.478455
loss 8.21744
cutoff 2.9824623e-6 network size 926
STEP 17 ================================
prereg loss 0.15001935 reg_l1 80.27263 reg_l2 24.340574
loss 8.177282
cutoff 4.511443e-6 network size 925
STEP 18 ================================
prereg loss 0.15511382 reg_l1 79.87537 reg_l2 24.203007
loss 8.142651
cutoff 1.1405646e-5 network size 924
STEP 19 ================================
prereg loss 0.15765221 reg_l1 79.45698 reg_l2 24.065779
loss 8.10335
cutoff 5.5335695e-6 network size 923
STEP 20 ================================
prereg loss 0.15952839 reg_l1 79.01819 reg_l2 23.929182
loss 8.061347
cutoff 4.1448948e-7 network size 922
STEP 21 ================================
prereg loss 0.1645715 reg_l1 78.6233 reg_l2 23.793684
loss 8.026901
cutoff 4.2723113e-6 network size 921
STEP 22 ================================
prereg loss 0.16815817 reg_l1 78.224815 reg_l2 23.65973
loss 7.9906397
cutoff 1.0006552e-6 network size 920
STEP 23 ================================
prereg loss 0.17142504 reg_l1 77.808876 reg_l2 23.527456
loss 7.9523125
cutoff 5.1164534e-7 network size 919
STEP 24 ================================
prereg loss 0.17285717 reg_l1 77.369095 reg_l2 23.396917
loss 7.9097667
cutoff 1.3074052e-5 network size 918
STEP 25 ================================
prereg loss 0.17847152 reg_l1 76.95516 reg_l2 23.268019
loss 7.8739877
cutoff 1.4945399e-6 network size 917
STEP 26 ================================
prereg loss 0.18319452 reg_l1 76.54711 reg_l2 23.140722
loss 7.837906
cutoff 4.9429946e-7 network size 916
STEP 27 ================================
prereg loss 0.1855256 reg_l1 76.126526 reg_l2 23.014992
loss 7.798178
cutoff 6.425289e-6 network size 915
STEP 28 ================================
prereg loss 0.19050792 reg_l1 75.70231 reg_l2 22.89107
loss 7.760739
cutoff 4.7287118e-5 network size 914
STEP 29 ================================
prereg loss 0.19661734 reg_l1 75.32277 reg_l2 22.7695
loss 7.728894
cutoff 4.6758214e-7 network size 913
STEP 30 ================================
prereg loss 0.1979007 reg_l1 74.944885 reg_l2 22.650555
loss 7.6923895
cutoff 5.2086834e-6 network size 912
STEP 31 ================================
prereg loss 0.19897272 reg_l1 74.56066 reg_l2 22.53396
loss 7.655039
cutoff 7.974577e-6 network size 911
STEP 32 ================================
prereg loss 0.20309922 reg_l1 74.17903 reg_l2 22.4193
loss 7.6210027
cutoff 7.847693e-7 network size 910
STEP 33 ================================
prereg loss 0.20925166 reg_l1 73.83429 reg_l2 22.30652
loss 7.592681
cutoff 8.299889e-6 network size 909
STEP 34 ================================
prereg loss 0.20956193 reg_l1 73.48932 reg_l2 22.195435
loss 7.5584936
cutoff 3.0946376e-6 network size 908
STEP 35 ================================
prereg loss 0.21285462 reg_l1 73.134964 reg_l2 22.085867
loss 7.526351
cutoff 1.0667557e-5 network size 907
STEP 36 ================================
prereg loss 0.21816649 reg_l1 72.773766 reg_l2 21.977997
loss 7.495543
cutoff 1.1636821e-6 network size 906
STEP 37 ================================
prereg loss 0.22209683 reg_l1 72.44366 reg_l2 21.87215
loss 7.4664626
cutoff 2.3685025e-6 network size 905
STEP 38 ================================
prereg loss 0.22628678 reg_l1 72.13606 reg_l2 21.768478
loss 7.4398932
cutoff 9.556534e-7 network size 904
STEP 39 ================================
prereg loss 0.23139144 reg_l1 71.81452 reg_l2 21.666813
loss 7.4128437
cutoff 1.0615331e-6 network size 903
STEP 40 ================================
prereg loss 0.2369216 reg_l1 71.479126 reg_l2 21.566978
loss 7.3848343
cutoff 2.2317363e-6 network size 902
STEP 41 ================================
prereg loss 0.2407965 reg_l1 71.164345 reg_l2 21.468874
loss 7.357231
cutoff 3.9167353e-6 network size 901
STEP 42 ================================
prereg loss 0.24471474 reg_l1 70.861275 reg_l2 21.372478
loss 7.3308425
cutoff 1.1214834e-5 network size 900
STEP 43 ================================
prereg loss 0.24872822 reg_l1 70.560234 reg_l2 21.27776
loss 7.304752
cutoff 3.0917e-7 network size 899
STEP 44 ================================
prereg loss 0.25359014 reg_l1 70.240295 reg_l2 21.184597
loss 7.27762
cutoff 1.9805593e-6 network size 898
STEP 45 ================================
prereg loss 0.25973973 reg_l1 69.95296 reg_l2 21.093088
loss 7.255036
cutoff 3.851863e-6 network size 897
STEP 46 ================================
prereg loss 0.26360977 reg_l1 69.664734 reg_l2 21.00338
loss 7.2300835
cutoff 4.3901673e-6 network size 896
STEP 47 ================================
prereg loss 0.26939175 reg_l1 69.36009 reg_l2 20.915344
loss 7.205401
cutoff 1.5271653e-6 network size 895
STEP 48 ================================
prereg loss 0.27201542 reg_l1 69.05215 reg_l2 20.82882
loss 7.1772304
cutoff 2.3730536e-7 network size 894
STEP 49 ================================
prereg loss 0.27971438 reg_l1 68.77967 reg_l2 20.743877
loss 7.157682
cutoff 2.3769098e-7 network size 893
STEP 50 ================================
prereg loss 0.2834778 reg_l1 68.52051 reg_l2 20.660357
loss 7.1355286
cutoff 3.944966e-6 network size 892
STEP 51 ================================
prereg loss 0.28918913 reg_l1 68.24194 reg_l2 20.578074
loss 7.113384
cutoff 4.321744e-6 network size 891
STEP 52 ================================
prereg loss 0.29483473 reg_l1 67.952324 reg_l2 20.497034
loss 7.090067
cutoff 1.3998006e-6 network size 890
STEP 53 ================================
prereg loss 0.3014774 reg_l1 67.68256 reg_l2 20.41764
loss 7.069734
cutoff 5.6692807e-7 network size 889
STEP 54 ================================
prereg loss 0.30711654 reg_l1 67.436165 reg_l2 20.340149
loss 7.050733
cutoff 1.5757629e-5 network size 888
STEP 55 ================================
prereg loss 0.3083299 reg_l1 67.17577 reg_l2 20.264532
loss 7.0259075
cutoff 3.8292164e-6 network size 887
STEP 56 ================================
prereg loss 0.30927488 reg_l1 66.91095 reg_l2 20.19041
loss 7.0003695
cutoff 2.919056e-6 network size 886
STEP 57 ================================
prereg loss 0.311331 reg_l1 66.66417 reg_l2 20.117556
loss 6.977748
cutoff 3.282359e-6 network size 885
STEP 58 ================================
prereg loss 0.31445676 reg_l1 66.43377 reg_l2 20.045774
loss 6.957834
cutoff 6.180271e-7 network size 884
STEP 59 ================================
prereg loss 0.3160659 reg_l1 66.19501 reg_l2 19.974752
loss 6.9355664
cutoff 2.7433853e-6 network size 883
STEP 60 ================================
prereg loss 0.31813097 reg_l1 65.94359 reg_l2 19.90458
loss 6.91249
cutoff 2.2856984e-6 network size 882
STEP 61 ================================
prereg loss 0.32210475 reg_l1 65.71331 reg_l2 19.835615
loss 6.893436
cutoff 6.0748716e-6 network size 881
STEP 62 ================================
prereg loss 0.32420737 reg_l1 65.482056 reg_l2 19.768106
loss 6.872413
cutoff 3.1581221e-6 network size 880
STEP 63 ================================
prereg loss 0.32568946 reg_l1 65.23798 reg_l2 19.701948
loss 6.849488
cutoff 1.3356039e-6 network size 879
STEP 64 ================================
prereg loss 0.32815287 reg_l1 64.99026 reg_l2 19.636732
loss 6.8271785
cutoff 7.59706e-6 network size 878
STEP 65 ================================
prereg loss 0.33200517 reg_l1 64.760315 reg_l2 19.572289
loss 6.808037
cutoff 6.4173946e-8 network size 877
STEP 66 ================================
prereg loss 0.33523202 reg_l1 64.54344 reg_l2 19.508518
loss 6.7895765
cutoff 1.3534736e-7 network size 876
STEP 67 ================================
prereg loss 0.33880952 reg_l1 64.30649 reg_l2 19.445587
loss 6.7694583
cutoff 2.9936782e-6 network size 875
STEP 68 ================================
prereg loss 0.34297213 reg_l1 64.06641 reg_l2 19.383768
loss 6.749613
cutoff 9.659561e-8 network size 874
STEP 69 ================================
prereg loss 0.34601542 reg_l1 63.837578 reg_l2 19.323288
loss 6.7297735
cutoff 1.9750041e-7 network size 873
STEP 70 ================================
prereg loss 0.34773436 reg_l1 63.622097 reg_l2 19.264107
loss 6.7099442
cutoff 5.2586984e-7 network size 872
STEP 71 ================================
prereg loss 0.35075507 reg_l1 63.396217 reg_l2 19.205927
loss 6.690377
cutoff 2.8143404e-8 network size 871
STEP 72 ================================
prereg loss 0.35415357 reg_l1 63.161133 reg_l2 19.148575
loss 6.670267
cutoff 5.691836e-7 network size 870
STEP 73 ================================
prereg loss 0.35715786 reg_l1 62.945602 reg_l2 19.09208
loss 6.651718
cutoff 6.893093e-6 network size 869
STEP 74 ================================
prereg loss 0.36040795 reg_l1 62.738594 reg_l2 19.036518
loss 6.6342673
cutoff 3.0681094e-6 network size 868
STEP 75 ================================
prereg loss 0.36338174 reg_l1 62.52203 reg_l2 18.982027
loss 6.615585
cutoff 4.453599e-6 network size 867
STEP 76 ================================
prereg loss 0.3666625 reg_l1 62.30468 reg_l2 18.928629
loss 6.597131
cutoff 1.4236139e-6 network size 866
STEP 77 ================================
prereg loss 0.3702095 reg_l1 62.101414 reg_l2 18.876436
loss 6.5803514
cutoff 1.637316e-6 network size 865
STEP 78 ================================
prereg loss 0.37376064 reg_l1 61.907883 reg_l2 18.825344
loss 6.564549
cutoff 3.1558302e-6 network size 864
STEP 79 ================================
prereg loss 0.37645993 reg_l1 61.71238 reg_l2 18.775248
loss 6.547698
cutoff 1.6525155e-7 network size 863
STEP 80 ================================
prereg loss 0.37868813 reg_l1 61.51905 reg_l2 18.725872
loss 6.5305934
cutoff 2.161978e-7 network size 862
STEP 81 ================================
prereg loss 0.3812902 reg_l1 61.343815 reg_l2 18.677094
loss 6.5156717
cutoff 9.546056e-9 network size 861
STEP 82 ================================
prereg loss 0.3843657 reg_l1 61.18324 reg_l2 18.62891
loss 6.5026894
cutoff 2.302404e-7 network size 860
STEP 83 ================================
prereg loss 0.38815024 reg_l1 61.01518 reg_l2 18.581383
loss 6.4896684
cutoff 3.123714e-7 network size 859
STEP 84 ================================
prereg loss 0.39065546 reg_l1 60.830486 reg_l2 18.534403
loss 6.4737043
cutoff 3.4885306e-6 network size 858
STEP 85 ================================
prereg loss 0.39296773 reg_l1 60.653484 reg_l2 18.488163
loss 6.4583163
cutoff 3.3748802e-7 network size 857
STEP 86 ================================
prereg loss 0.39441636 reg_l1 60.48186 reg_l2 18.442652
loss 6.4426026
cutoff 7.322524e-8 network size 856
STEP 87 ================================
prereg loss 0.39581227 reg_l1 60.307762 reg_l2 18.397629
loss 6.426589
cutoff 1.3394456e-6 network size 855
STEP 88 ================================
prereg loss 0.39709532 reg_l1 60.127266 reg_l2 18.352924
loss 6.409822
cutoff 1.1843531e-6 network size 854
STEP 89 ================================
prereg loss 0.39790624 reg_l1 59.95967 reg_l2 18.30863
loss 6.3938737
cutoff 3.3133401e-6 network size 853
STEP 90 ================================
prereg loss 0.39835736 reg_l1 59.792267 reg_l2 18.264647
loss 6.377584
cutoff 1.6037084e-6 network size 852
STEP 91 ================================
prereg loss 0.3989176 reg_l1 59.62439 reg_l2 18.221169
loss 6.3613567
cutoff 3.3978722e-7 network size 851
STEP 92 ================================
prereg loss 0.399827 reg_l1 59.456852 reg_l2 18.178143
loss 6.3455124
cutoff 1.5696423e-7 network size 850
STEP 93 ================================
prereg loss 0.4023119 reg_l1 59.291557 reg_l2 18.135532
loss 6.3314676
cutoff 2.7334318e-7 network size 849
STEP 94 ================================
prereg loss 0.40286505 reg_l1 59.1342 reg_l2 18.093336
loss 6.316285
cutoff 9.680589e-7 network size 848
STEP 95 ================================
prereg loss 0.40432027 reg_l1 58.974976 reg_l2 18.05157
loss 6.301818
cutoff 5.2786545e-6 network size 847
STEP 96 ================================
prereg loss 0.40662754 reg_l1 58.805344 reg_l2 18.010118
loss 6.2871623
cutoff 2.9691728e-7 network size 846
STEP 97 ================================
prereg loss 0.4102359 reg_l1 58.647934 reg_l2 17.968912
loss 6.275029
cutoff 4.254689e-7 network size 845
STEP 98 ================================
prereg loss 0.41304448 reg_l1 58.49626 reg_l2 17.927902
loss 6.2626705
cutoff 2.7168426e-7 network size 844
STEP 99 ================================
prereg loss 0.4137524 reg_l1 58.335705 reg_l2 17.887175
loss 6.247323
cutoff 4.973299e-8 network size 843
STEP 100 ================================
prereg loss 0.41616973 reg_l1 58.17992 reg_l2 17.846931
loss 6.234162
cutoff 1.0365329e-7 network size 842
2022-08-14T10:54:50.448
```

As planned, we are going to switch to interleaving steps.

After 100 more sparsifications in 200 steps, the run is quite smooth:

```
julia> interleaving_steps!(200)
2022-08-14T10:58:13.915
STEP 1 ================================
prereg loss 0.41870657 reg_l1 58.039562 reg_l2 17.807367
loss 6.222663
cutoff 1.9904019e-7 network size 841
STEP 2 ================================
prereg loss 0.42039454 reg_l1 57.899822 reg_l2 17.76863
loss 6.2103767
STEP 3 ================================
prereg loss 0.42189512 reg_l1 57.75913 reg_l2 17.730568
loss 6.197808
cutoff 1.4532952e-6 network size 840
STEP 4 ================================
prereg loss 0.4241423 reg_l1 57.61733 reg_l2 17.692965
loss 6.1858754
STEP 5 ================================
prereg loss 0.42684358 reg_l1 57.47861 reg_l2 17.655544
loss 6.174705
cutoff 3.7750397e-7 network size 839
STEP 6 ================================
prereg loss 0.42880875 reg_l1 57.33992 reg_l2 17.618332
loss 6.162801
STEP 7 ================================
prereg loss 0.4300568 reg_l1 57.200356 reg_l2 17.581373
loss 6.150092
cutoff 4.4352782e-7 network size 838
STEP 8 ================================
prereg loss 0.430929 reg_l1 57.055073 reg_l2 17.544746
loss 6.1364365
STEP 9 ================================
prereg loss 0.4325134 reg_l1 56.925453 reg_l2 17.508623
loss 6.1250587
cutoff 8.0562313e-7 network size 837
STEP 10 ================================
prereg loss 0.43377048 reg_l1 56.801636 reg_l2 17.473011
loss 6.1139345
STEP 11 ================================
prereg loss 0.43519655 reg_l1 56.66609 reg_l2 17.437925
loss 6.101805
cutoff 6.485061e-7 network size 836
STEP 12 ================================
prereg loss 0.43675607 reg_l1 56.527107 reg_l2 17.403255
loss 6.089467
STEP 13 ================================
prereg loss 0.43925455 reg_l1 56.39435 reg_l2 17.368862
loss 6.0786896
cutoff 1.2416713e-6 network size 835
STEP 14 ================================
prereg loss 0.44250295 reg_l1 56.260643 reg_l2 17.334475
loss 6.0685673
STEP 15 ================================
prereg loss 0.44506797 reg_l1 56.120586 reg_l2 17.299995
loss 6.0571265
cutoff 9.5312134e-7 network size 834
STEP 16 ================================
prereg loss 0.44777158 reg_l1 55.980225 reg_l2 17.265465
loss 6.045794
STEP 17 ================================
prereg loss 0.4512305 reg_l1 55.84782 reg_l2 17.23105
loss 6.0360126
cutoff 3.980793e-6 network size 833
STEP 18 ================================
prereg loss 0.45433202 reg_l1 55.724342 reg_l2 17.19702
loss 6.0267663
STEP 19 ================================
prereg loss 0.4571817 reg_l1 55.599712 reg_l2 17.163517
loss 6.017153
cutoff 3.3179313e-6 network size 832
STEP 20 ================================
prereg loss 0.45997962 reg_l1 55.470364 reg_l2 17.130642
loss 6.007016
STEP 21 ================================
prereg loss 0.46219486 reg_l1 55.346523 reg_l2 17.098368
loss 5.996847
cutoff 2.6744965e-7 network size 831
STEP 22 ================================
prereg loss 0.4639798 reg_l1 55.224804 reg_l2 17.066605
loss 5.98646
STEP 23 ================================
prereg loss 0.4665507 reg_l1 55.104053 reg_l2 17.035242
loss 5.9769564
cutoff 2.0125299e-8 network size 830
STEP 24 ================================
prereg loss 0.47004464 reg_l1 54.977108 reg_l2 17.00414
loss 5.9677553
STEP 25 ================================
prereg loss 0.47284973 reg_l1 54.86464 reg_l2 16.973267
loss 5.959314
cutoff 6.696155e-7 network size 829
STEP 26 ================================
prereg loss 0.4752563 reg_l1 54.74825 reg_l2 16.942596
loss 5.9500813
STEP 27 ================================
prereg loss 0.4775244 reg_l1 54.627342 reg_l2 16.912024
loss 5.9402585
cutoff 2.5494373e-6 network size 828
STEP 28 ================================
prereg loss 0.4796258 reg_l1 54.50942 reg_l2 16.881622
loss 5.9305677
STEP 29 ================================
prereg loss 0.48151436 reg_l1 54.394665 reg_l2 16.851397
loss 5.920981
cutoff 3.528241e-6 network size 827
STEP 30 ================================
prereg loss 0.48405498 reg_l1 54.285637 reg_l2 16.82138
loss 5.9126186
STEP 31 ================================
prereg loss 0.48699513 reg_l1 54.169037 reg_l2 16.791681
loss 5.903899
cutoff 5.115571e-7 network size 826
STEP 32 ================================
prereg loss 0.48987913 reg_l1 54.04899 reg_l2 16.762306
loss 5.8947783
STEP 33 ================================
prereg loss 0.4927927 reg_l1 53.93402 reg_l2 16.733051
loss 5.8861947
cutoff 7.4945274e-7 network size 825
STEP 34 ================================
prereg loss 0.49528003 reg_l1 53.82024 reg_l2 16.703814
loss 5.877304
STEP 35 ================================
prereg loss 0.49848405 reg_l1 53.699444 reg_l2 16.674572
loss 5.8684287
cutoff 9.11823e-7 network size 824
STEP 36 ================================
prereg loss 0.50213116 reg_l1 53.583214 reg_l2 16.645535
loss 5.8604527
STEP 37 ================================
prereg loss 0.5034312 reg_l1 53.481796 reg_l2 16.616795
loss 5.851611
cutoff 1.1824886e-7 network size 823
STEP 38 ================================
prereg loss 0.5043241 reg_l1 53.37939 reg_l2 16.588358
loss 5.842263
STEP 39 ================================
prereg loss 0.50658536 reg_l1 53.270832 reg_l2 16.560097
loss 5.8336687
cutoff 1.0153399e-6 network size 822
STEP 40 ================================
prereg loss 0.5089543 reg_l1 53.16548 reg_l2 16.531948
loss 5.8255024
STEP 41 ================================
prereg loss 0.51256645 reg_l1 53.06284 reg_l2 16.50399
loss 5.8188505
cutoff 5.064794e-7 network size 821
STEP 42 ================================
prereg loss 0.5155465 reg_l1 52.959396 reg_l2 16.476212
loss 5.8114862
STEP 43 ================================
prereg loss 0.5163566 reg_l1 52.854424 reg_l2 16.448608
loss 5.801799
cutoff 2.0852895e-7 network size 820
STEP 44 ================================
prereg loss 0.51746625 reg_l1 52.742 reg_l2 16.421206
loss 5.791666
STEP 45 ================================
prereg loss 0.5193394 reg_l1 52.637547 reg_l2 16.394085
loss 5.7830944
cutoff 2.160843e-6 network size 819
STEP 46 ================================
prereg loss 0.5205784 reg_l1 52.533367 reg_l2 16.367386
loss 5.7739153
STEP 47 ================================
prereg loss 0.522422 reg_l1 52.42379 reg_l2 16.341135
loss 5.764801
cutoff 9.979412e-7 network size 818
STEP 48 ================================
prereg loss 0.5245332 reg_l1 52.31593 reg_l2 16.315168
loss 5.7561264
STEP 49 ================================
prereg loss 0.52514887 reg_l1 52.21946 reg_l2 16.289328
loss 5.747095
cutoff 7.2104376e-7 network size 817
STEP 50 ================================
prereg loss 0.5264178 reg_l1 52.124676 reg_l2 16.263561
loss 5.7388854
STEP 51 ================================
prereg loss 0.52720296 reg_l1 52.02664 reg_l2 16.237865
loss 5.7298675
cutoff 9.884097e-7 network size 816
STEP 52 ================================
prereg loss 0.5284594 reg_l1 51.930523 reg_l2 16.212355
loss 5.721512
STEP 53 ================================
prereg loss 0.53053343 reg_l1 51.835922 reg_l2 16.187153
loss 5.7141256
cutoff 1.7263592e-6 network size 815
STEP 54 ================================
prereg loss 0.53166795 reg_l1 51.74005 reg_l2 16.162268
loss 5.705673
STEP 55 ================================
prereg loss 0.53282493 reg_l1 51.6402 reg_l2 16.137716
loss 5.696845
cutoff 1.8806895e-6 network size 814
STEP 56 ================================
prereg loss 0.5346678 reg_l1 51.541294 reg_l2 16.113445
loss 5.6887975
STEP 57 ================================
prereg loss 0.53712845 reg_l1 51.44606 reg_l2 16.089285
loss 5.6817346
cutoff 1.1028023e-6 network size 813
STEP 58 ================================
prereg loss 0.53879666 reg_l1 51.350796 reg_l2 16.065125
loss 5.673877
STEP 59 ================================
prereg loss 0.54002696 reg_l1 51.246788 reg_l2 16.041008
loss 5.664706
cutoff 1.106222e-6 network size 812
STEP 60 ================================
prereg loss 0.5415396 reg_l1 51.147495 reg_l2 16.017096
loss 5.656289
STEP 61 ================================
prereg loss 0.54302686 reg_l1 51.05846 reg_l2 15.993444
loss 5.648873
cutoff 4.3702312e-7 network size 811
STEP 62 ================================
prereg loss 0.544627 reg_l1 50.969456 reg_l2 15.970037
loss 5.641573
STEP 63 ================================
prereg loss 0.54652035 reg_l1 50.874676 reg_l2 15.946832
loss 5.633988
cutoff 2.3201064e-6 network size 810
STEP 64 ================================
prereg loss 0.5478003 reg_l1 50.78631 reg_l2 15.923812
loss 5.6264315
STEP 65 ================================
prereg loss 0.549496 reg_l1 50.71068 reg_l2 15.901028
loss 5.620564
cutoff 2.6990892e-6 network size 809
STEP 66 ================================
prereg loss 0.55254483 reg_l1 50.630264 reg_l2 15.878536
loss 5.615571
STEP 67 ================================
prereg loss 0.5560056 reg_l1 50.54166 reg_l2 15.856408
loss 5.610172
cutoff 1.3429963e-7 network size 808
STEP 68 ================================
prereg loss 0.55815655 reg_l1 50.450764 reg_l2 15.834554
loss 5.603233
STEP 69 ================================
prereg loss 0.5595185 reg_l1 50.36647 reg_l2 15.812797
loss 5.5961657
cutoff 7.7265224e-7 network size 807
STEP 70 ================================
prereg loss 0.56042826 reg_l1 50.290764 reg_l2 15.791007
loss 5.5895047
STEP 71 ================================
prereg loss 0.5606736 reg_l1 50.2037 reg_l2 15.769229
loss 5.5810437
cutoff 4.0719533e-7 network size 806
STEP 72 ================================
prereg loss 0.5612733 reg_l1 50.115936 reg_l2 15.747696
loss 5.572867
STEP 73 ================================
prereg loss 0.56169134 reg_l1 50.03358 reg_l2 15.726682
loss 5.5650496
cutoff 6.585324e-7 network size 805
STEP 74 ================================
prereg loss 0.56140363 reg_l1 49.952904 reg_l2 15.706283
loss 5.556694
STEP 75 ================================
prereg loss 0.56166923 reg_l1 49.86517 reg_l2 15.686236
loss 5.5481863
cutoff 1.2715172e-6 network size 804
STEP 76 ================================
prereg loss 0.561056 reg_l1 49.774185 reg_l2 15.666209
loss 5.5384746
STEP 77 ================================
prereg loss 0.5610637 reg_l1 49.69876 reg_l2 15.645946
loss 5.53094
cutoff 1.2760593e-6 network size 803
STEP 78 ================================
prereg loss 0.56115705 reg_l1 49.627045 reg_l2 15.625424
loss 5.523862
STEP 79 ================================
prereg loss 0.5610075 reg_l1 49.55429 reg_l2 15.604863
loss 5.5164366
cutoff 3.749501e-6 network size 802
STEP 80 ================================
prereg loss 0.5626171 reg_l1 49.470966 reg_l2 15.584386
loss 5.509714
STEP 81 ================================
prereg loss 0.5676451 reg_l1 49.386177 reg_l2 15.563985
loss 5.506263
cutoff 3.2442495e-7 network size 801
STEP 82 ================================
prereg loss 0.56792665 reg_l1 49.30597 reg_l2 15.543552
loss 5.4985237
STEP 83 ================================
prereg loss 0.5660476 reg_l1 49.226994 reg_l2 15.523124
loss 5.488747
cutoff 1.2887176e-6 network size 800
STEP 84 ================================
prereg loss 0.5684652 reg_l1 49.15388 reg_l2 15.502918
loss 5.4838533
STEP 85 ================================
prereg loss 0.56999063 reg_l1 49.07969 reg_l2 15.483261
loss 5.4779596
cutoff 4.0452505e-7 network size 799
STEP 86 ================================
prereg loss 0.5698438 reg_l1 49.00447 reg_l2 15.464272
loss 5.470291
STEP 87 ================================
prereg loss 0.5724999 reg_l1 48.929516 reg_l2 15.445733
loss 5.4654512
cutoff 1.221284e-6 network size 798
STEP 88 ================================
prereg loss 0.575754 reg_l1 48.85567 reg_l2 15.4272375
loss 5.4613214
STEP 89 ================================
prereg loss 0.574247 reg_l1 48.78154 reg_l2 15.408528
loss 5.452401
cutoff 3.680725e-7 network size 797
STEP 90 ================================
prereg loss 0.57174253 reg_l1 48.707333 reg_l2 15.389608
loss 5.442476
STEP 91 ================================
prereg loss 0.575131 reg_l1 48.63306 reg_l2 15.370843
loss 5.438437
cutoff 8.5373904e-7 network size 796
STEP 92 ================================
prereg loss 0.5749949 reg_l1 48.556675 reg_l2 15.352557
loss 5.4306626
STEP 93 ================================
prereg loss 0.57295495 reg_l1 48.480648 reg_l2 15.334794
loss 5.42102
cutoff 5.2586984e-8 network size 795
STEP 94 ================================
prereg loss 0.57477444 reg_l1 48.409286 reg_l2 15.317404
loss 5.415703
STEP 95 ================================
prereg loss 0.5777281 reg_l1 48.335663 reg_l2 15.300076
loss 5.4112945
cutoff 4.388785e-7 network size 794
STEP 96 ================================
prereg loss 0.5776274 reg_l1 48.26376 reg_l2 15.282634
loss 5.404003
STEP 97 ================================
prereg loss 0.5786184 reg_l1 48.196712 reg_l2 15.265107
loss 5.3982897
cutoff 5.993843e-7 network size 793
STEP 98 ================================
prereg loss 0.58184344 reg_l1 48.130074 reg_l2 15.247608
loss 5.3948507
STEP 99 ================================
prereg loss 0.58356 reg_l1 48.057796 reg_l2 15.230188
loss 5.38934
cutoff 4.497747e-7 network size 792
STEP 100 ================================
prereg loss 0.58421457 reg_l1 47.983807 reg_l2 15.212795
loss 5.3825955
STEP 101 ================================
prereg loss 0.58345747 reg_l1 47.91525 reg_l2 15.195324
loss 5.3749824
cutoff 9.505893e-7 network size 791
STEP 102 ================================
prereg loss 0.58281666 reg_l1 47.84599 reg_l2 15.177899
loss 5.3674154
STEP 103 ================================
prereg loss 0.58348995 reg_l1 47.77658 reg_l2 15.160652
loss 5.361148
cutoff 4.770609e-7 network size 790
STEP 104 ================================
prereg loss 0.5849276 reg_l1 47.7062 reg_l2 15.143956
loss 5.3555474
STEP 105 ================================
prereg loss 0.5851767 reg_l1 47.642673 reg_l2 15.127815
loss 5.3494444
cutoff 1.8415158e-6 network size 789
STEP 106 ================================
prereg loss 0.584682 reg_l1 47.58373 reg_l2 15.112079
loss 5.343055
STEP 107 ================================
prereg loss 0.5858668 reg_l1 47.519707 reg_l2 15.096458
loss 5.3378377
cutoff 7.857161e-7 network size 788
STEP 108 ================================
prereg loss 0.58737993 reg_l1 47.452 reg_l2 15.080551
loss 5.33258
STEP 109 ================================
prereg loss 0.5883738 reg_l1 47.39036 reg_l2 15.064432
loss 5.3274097
cutoff 2.6111957e-7 network size 787
STEP 110 ================================
prereg loss 0.58894783 reg_l1 47.32317 reg_l2 15.048536
loss 5.3212647
STEP 111 ================================
prereg loss 0.5930419 reg_l1 47.260677 reg_l2 15.033117
loss 5.31911
cutoff 1.911485e-7 network size 786
STEP 112 ================================
prereg loss 0.59479904 reg_l1 47.19893 reg_l2 15.0182
loss 5.314692
STEP 113 ================================
prereg loss 0.5958626 reg_l1 47.14147 reg_l2 15.003399
loss 5.31001
cutoff 3.750174e-7 network size 785
STEP 114 ================================
prereg loss 0.59823716 reg_l1 47.08403 reg_l2 14.988453
loss 5.30664
STEP 115 ================================
prereg loss 0.59883595 reg_l1 47.0259 reg_l2 14.973256
loss 5.3014264
cutoff 1.3590034e-7 network size 784
STEP 116 ================================
prereg loss 0.59980756 reg_l1 46.962124 reg_l2 14.9579115
loss 5.29602
STEP 117 ================================
prereg loss 0.60195565 reg_l1 46.901894 reg_l2 14.942714
loss 5.292145
cutoff 2.6732305e-6 network size 783
STEP 118 ================================
prereg loss 0.6053258 reg_l1 46.840675 reg_l2 14.927868
loss 5.2893934
STEP 119 ================================
prereg loss 0.60222524 reg_l1 46.7808 reg_l2 14.913468
loss 5.2803054
cutoff 5.8737805e-7 network size 782
STEP 120 ================================
prereg loss 0.59960645 reg_l1 46.719475 reg_l2 14.899314
loss 5.271554
STEP 121 ================================
prereg loss 0.6008214 reg_l1 46.660442 reg_l2 14.885378
loss 5.2668657
cutoff 1.5448313e-7 network size 781
STEP 122 ================================
prereg loss 0.6014592 reg_l1 46.605442 reg_l2 14.871577
loss 5.2620034
STEP 123 ================================
prereg loss 0.6028168 reg_l1 46.548565 reg_l2 14.85789
loss 5.2576733
cutoff 6.300979e-8 network size 780
STEP 124 ================================
prereg loss 0.6026314 reg_l1 46.490982 reg_l2 14.844303
loss 5.25173
STEP 125 ================================
prereg loss 0.603351 reg_l1 46.437042 reg_l2 14.830655
loss 5.2470555
cutoff 1.2741043e-7 network size 779
STEP 126 ================================
prereg loss 0.6039933 reg_l1 46.38075 reg_l2 14.816683
loss 5.2420683
STEP 127 ================================
prereg loss 0.6041866 reg_l1 46.321808 reg_l2 14.802342
loss 5.236367
cutoff 1.7849379e-7 network size 778
STEP 128 ================================
prereg loss 0.605421 reg_l1 46.26101 reg_l2 14.787817
loss 5.231522
STEP 129 ================================
prereg loss 0.60568196 reg_l1 46.202988 reg_l2 14.773298
loss 5.2259808
cutoff 9.2526534e-8 network size 777
STEP 130 ================================
prereg loss 0.60719967 reg_l1 46.14732 reg_l2 14.758891
loss 5.221932
STEP 131 ================================
prereg loss 0.6078144 reg_l1 46.089054 reg_l2 14.744695
loss 5.2167196
cutoff 3.6568963e-8 network size 776
STEP 132 ================================
prereg loss 0.61012304 reg_l1 46.030514 reg_l2 14.730949
loss 5.213175
STEP 133 ================================
prereg loss 0.6125052 reg_l1 45.978024 reg_l2 14.717697
loss 5.210308
cutoff 5.333932e-7 network size 775
STEP 134 ================================
prereg loss 0.6119639 reg_l1 45.924732 reg_l2 14.704722
loss 5.2044373
STEP 135 ================================
prereg loss 0.6107276 reg_l1 45.872887 reg_l2 14.691657
loss 5.1980166
cutoff 9.6069925e-9 network size 774
STEP 136 ================================
prereg loss 0.61023057 reg_l1 45.814255 reg_l2 14.678381
loss 5.191656
STEP 137 ================================
prereg loss 0.6108409 reg_l1 45.75514 reg_l2 14.665083
loss 5.1863546
cutoff 1.446555e-6 network size 773
STEP 138 ================================
prereg loss 0.61031413 reg_l1 45.69527 reg_l2 14.651987
loss 5.179841
STEP 139 ================================
prereg loss 0.61498356 reg_l1 45.6382 reg_l2 14.639304
loss 5.1788034
cutoff 6.2500476e-9 network size 772
STEP 140 ================================
prereg loss 0.6179225 reg_l1 45.588604 reg_l2 14.627094
loss 5.1767826
STEP 141 ================================
prereg loss 0.61466104 reg_l1 45.545395 reg_l2 14.61518
loss 5.169201
cutoff 3.0819956e-7 network size 771
STEP 142 ================================
prereg loss 0.61316794 reg_l1 45.4982 reg_l2 14.603271
loss 5.1629877
STEP 143 ================================
prereg loss 0.61310273 reg_l1 45.44661 reg_l2 14.59109
loss 5.157764
cutoff 4.3928958e-7 network size 770
STEP 144 ================================
prereg loss 0.61218643 reg_l1 45.390545 reg_l2 14.578578
loss 5.151241
STEP 145 ================================
prereg loss 0.6118953 reg_l1 45.34364 reg_l2 14.566048
loss 5.1462593
cutoff 1.9155596e-6 network size 769
STEP 146 ================================
prereg loss 0.61351335 reg_l1 45.296314 reg_l2 14.553855
loss 5.143145
STEP 147 ================================
prereg loss 0.6135176 reg_l1 45.245842 reg_l2 14.54219
loss 5.138102
cutoff 1.2581222e-6 network size 768
STEP 148 ================================
prereg loss 0.61210734 reg_l1 45.192257 reg_l2 14.5309925
loss 5.131333
STEP 149 ================================
prereg loss 0.6120812 reg_l1 45.142704 reg_l2 14.520046
loss 5.1263514
cutoff 8.40912e-8 network size 767
STEP 150 ================================
prereg loss 0.61300933 reg_l1 45.096523 reg_l2 14.509177
loss 5.122662
STEP 151 ================================
prereg loss 0.6174181 reg_l1 45.047863 reg_l2 14.498275
loss 5.122205
cutoff 1.6320118e-6 network size 766
STEP 152 ================================
prereg loss 0.6190713 reg_l1 44.997856 reg_l2 14.487277
loss 5.1188574
STEP 153 ================================
prereg loss 0.6161196 reg_l1 44.946453 reg_l2 14.475948
loss 5.1107655
cutoff 2.702982e-7 network size 765
STEP 154 ================================
prereg loss 0.6147912 reg_l1 44.89775 reg_l2 14.464283
loss 5.1045666
STEP 155 ================================
prereg loss 0.61485374 reg_l1 44.847736 reg_l2 14.452409
loss 5.0996275
cutoff 1.8761057e-8 network size 764
STEP 156 ================================
prereg loss 0.61519593 reg_l1 44.796535 reg_l2 14.440629
loss 5.0948496
STEP 157 ================================
prereg loss 0.6168647 reg_l1 44.74935 reg_l2 14.429269
loss 5.0917997
cutoff 2.91373e-7 network size 763
STEP 158 ================================
prereg loss 0.62135476 reg_l1 44.704304 reg_l2 14.418589
loss 5.091785
STEP 159 ================================
prereg loss 0.6195546 reg_l1 44.658287 reg_l2 14.408375
loss 5.0853834
cutoff 8.1202415e-7 network size 762
STEP 160 ================================
prereg loss 0.61549246 reg_l1 44.608456 reg_l2 14.398359
loss 5.076338
STEP 161 ================================
prereg loss 0.6159396 reg_l1 44.56394 reg_l2 14.38807
loss 5.0723333
cutoff 7.557901e-8 network size 761
STEP 162 ================================
prereg loss 0.6155534 reg_l1 44.51895 reg_l2 14.37729
loss 5.0674486
STEP 163 ================================
prereg loss 0.61374724 reg_l1 44.470146 reg_l2 14.366249
loss 5.060762
cutoff 9.657088e-7 network size 760
STEP 164 ================================
prereg loss 0.6146521 reg_l1 44.417976 reg_l2 14.355412
loss 5.05645
STEP 165 ================================
prereg loss 0.61770815 reg_l1 44.371708 reg_l2 14.345159
loss 5.054879
cutoff 5.969159e-7 network size 759
STEP 166 ================================
prereg loss 0.6183361 reg_l1 44.326496 reg_l2 14.335384
loss 5.050986
STEP 167 ================================
prereg loss 0.61715585 reg_l1 44.282047 reg_l2 14.325768
loss 5.045361
cutoff 8.9613604e-7 network size 758
STEP 168 ================================
prereg loss 0.61583453 reg_l1 44.233223 reg_l2 14.315976
loss 5.039157
STEP 169 ================================
prereg loss 0.61589444 reg_l1 44.183586 reg_l2 14.305942
loss 5.034253
cutoff 5.533948e-7 network size 757
STEP 170 ================================
prereg loss 0.61788857 reg_l1 44.136227 reg_l2 14.295906
loss 5.0315113
STEP 171 ================================
prereg loss 0.61828583 reg_l1 44.086468 reg_l2 14.286187
loss 5.0269327
cutoff 6.5469726e-7 network size 756
STEP 172 ================================
prereg loss 0.61779165 reg_l1 44.035328 reg_l2 14.276684
loss 5.0213246
STEP 173 ================================
prereg loss 0.6178905 reg_l1 43.987793 reg_l2 14.2672825
loss 5.0166698
cutoff 3.0197407e-7 network size 755
STEP 174 ================================
prereg loss 0.6196228 reg_l1 43.94428 reg_l2 14.258041
loss 5.0140505
STEP 175 ================================
prereg loss 0.62112343 reg_l1 43.89759 reg_l2 14.248809
loss 5.0108824
cutoff 8.5943066e-7 network size 754
STEP 176 ================================
prereg loss 0.62093574 reg_l1 43.853905 reg_l2 14.239596
loss 5.0063267
STEP 177 ================================
prereg loss 0.6204841 reg_l1 43.810852 reg_l2 14.2304325
loss 5.0015697
cutoff 6.204937e-8 network size 753
STEP 178 ================================
prereg loss 0.61995125 reg_l1 43.766293 reg_l2 14.221266
loss 4.9965806
STEP 179 ================================
prereg loss 0.62081724 reg_l1 43.722218 reg_l2 14.212093
loss 4.993039
cutoff 9.819923e-7 network size 752
STEP 180 ================================
prereg loss 0.6211881 reg_l1 43.676773 reg_l2 14.202909
loss 4.9888654
STEP 181 ================================
prereg loss 0.62213415 reg_l1 43.634655 reg_l2 14.193715
loss 4.9856
cutoff 6.570408e-7 network size 751
STEP 182 ================================
prereg loss 0.62360346 reg_l1 43.58954 reg_l2 14.184347
loss 4.9825573
STEP 183 ================================
prereg loss 0.6232638 reg_l1 43.542656 reg_l2 14.174979
loss 4.9775295
cutoff 1.2738883e-7 network size 750
STEP 184 ================================
prereg loss 0.62206745 reg_l1 43.49295 reg_l2 14.16565
loss 4.9713626
STEP 185 ================================
prereg loss 0.6212768 reg_l1 43.44976 reg_l2 14.156504
loss 4.966253
cutoff 1.2386763e-6 network size 749
STEP 186 ================================
prereg loss 0.62218094 reg_l1 43.4075 reg_l2 14.14762
loss 4.962931
STEP 187 ================================
prereg loss 0.6225456 reg_l1 43.35942 reg_l2 14.1389475
loss 4.958488
cutoff 2.0905281e-7 network size 748
STEP 188 ================================
prereg loss 0.6220414 reg_l1 43.312496 reg_l2 14.1303215
loss 4.953291
STEP 189 ================================
prereg loss 0.62264854 reg_l1 43.26773 reg_l2 14.121518
loss 4.949422
cutoff 2.7975693e-7 network size 747
STEP 190 ================================
prereg loss 0.6238395 reg_l1 43.22211 reg_l2 14.112439
loss 4.9460506
STEP 191 ================================
prereg loss 0.6249052 reg_l1 43.17144 reg_l2 14.103155
loss 4.942049
cutoff 8.377101e-7 network size 746
STEP 192 ================================
prereg loss 0.62549096 reg_l1 43.1245 reg_l2 14.094008
loss 4.937941
STEP 193 ================================
prereg loss 0.6258022 reg_l1 43.08207 reg_l2 14.085327
loss 4.934009
cutoff 3.8056896e-8 network size 745
STEP 194 ================================
prereg loss 0.62449 reg_l1 43.039482 reg_l2 14.077227
loss 4.928438
STEP 195 ================================
prereg loss 0.62399286 reg_l1 42.997597 reg_l2 14.069455
loss 4.923753
cutoff 1.4026409e-6 network size 744
STEP 196 ================================
prereg loss 0.6249026 reg_l1 42.955284 reg_l2 14.061658
loss 4.920431
STEP 197 ================================
prereg loss 0.626011 reg_l1 42.917053 reg_l2 14.053446
loss 4.9177165
cutoff 2.8485374e-7 network size 743
STEP 198 ================================
prereg loss 0.6266323 reg_l1 42.87606 reg_l2 14.044685
loss 4.9142385
STEP 199 ================================
prereg loss 0.62724113 reg_l1 42.834564 reg_l2 14.035702
loss 4.9106975
cutoff 5.968541e-7 network size 742
STEP 200 ================================
prereg loss 0.6289295 reg_l1 42.791847 reg_l2 14.026591
loss 4.9081144
2022-08-14T12:31:17.274
```

```
julia> # let's repeat this

julia> interleaving_steps!(200)
2022-08-14T12:35:02.053
STEP 1 ================================
prereg loss 0.629893 reg_l1 42.75649 reg_l2 14.017596
loss 4.905542
cutoff 3.8912185e-7 network size 741
STEP 2 ================================
prereg loss 0.6294205 reg_l1 42.71509 reg_l2 14.008799
loss 4.9009295
STEP 3 ================================
prereg loss 0.629314 reg_l1 42.675316 reg_l2 14.000335
loss 4.896846
cutoff 3.170826e-7 network size 740
STEP 4 ================================
prereg loss 0.62848526 reg_l1 42.635143 reg_l2 13.992137
loss 4.8919997
STEP 5 ================================
prereg loss 0.6281816 reg_l1 42.594208 reg_l2 13.984142
loss 4.8876023
cutoff 2.7029455e-7 network size 739
STEP 6 ================================
prereg loss 0.62914413 reg_l1 42.556034 reg_l2 13.976338
loss 4.8847475
STEP 7 ================================
prereg loss 0.63079154 reg_l1 42.515453 reg_l2 13.968627
loss 4.882337
cutoff 9.960495e-7 network size 738
STEP 8 ================================
prereg loss 0.6328965 reg_l1 42.47188 reg_l2 13.960872
loss 4.8800845
STEP 9 ================================
prereg loss 0.6316398 reg_l1 42.434635 reg_l2 13.953017
loss 4.8751035
cutoff 3.9370525e-7 network size 737
STEP 10 ================================
prereg loss 0.6310282 reg_l1 42.39704 reg_l2 13.9450865
loss 4.8707323
STEP 11 ================================
prereg loss 0.63137615 reg_l1 42.356354 reg_l2 13.937063
loss 4.8670115
cutoff 5.4269185e-7 network size 736
STEP 12 ================================
prereg loss 0.63185143 reg_l1 42.31415 reg_l2 13.929066
loss 4.863267
STEP 13 ================================
prereg loss 0.63336164 reg_l1 42.275063 reg_l2 13.921086
loss 4.860868
cutoff 1.620705e-6 network size 735
STEP 14 ================================
prereg loss 0.63583976 reg_l1 42.240376 reg_l2 13.913203
loss 4.8598776
STEP 15 ================================
prereg loss 0.6364371 reg_l1 42.205456 reg_l2 13.905381
loss 4.8569827
cutoff 7.402559e-8 network size 734
STEP 16 ================================
prereg loss 0.6355987 reg_l1 42.168503 reg_l2 13.897532
loss 4.852449
STEP 17 ================================
prereg loss 0.63618934 reg_l1 42.135624 reg_l2 13.889609
loss 4.849752
cutoff 3.8528378e-7 network size 733
STEP 18 ================================
prereg loss 0.636402 reg_l1 42.100468 reg_l2 13.881617
loss 4.846449
STEP 19 ================================
prereg loss 0.63678455 reg_l1 42.063736 reg_l2 13.873606
loss 4.8431582
cutoff 5.397669e-7 network size 732
STEP 20 ================================
prereg loss 0.6376149 reg_l1 42.024044 reg_l2 13.8657055
loss 4.840019
STEP 21 ================================
prereg loss 0.6384264 reg_l1 41.990604 reg_l2 13.857798
loss 4.8374867
cutoff 7.0824353e-7 network size 731
STEP 22 ================================
prereg loss 0.6389454 reg_l1 41.954815 reg_l2 13.849806
loss 4.8344274
STEP 23 ================================
prereg loss 0.6396634 reg_l1 41.911793 reg_l2 13.841787
loss 4.8308425
cutoff 3.009627e-7 network size 730
STEP 24 ================================
prereg loss 0.6409987 reg_l1 41.86803 reg_l2 13.83382
loss 4.827802
STEP 25 ================================
prereg loss 0.6424452 reg_l1 41.830307 reg_l2 13.826045
loss 4.8254757
cutoff 1.6050762e-7 network size 729
STEP 26 ================================
prereg loss 0.6426672 reg_l1 41.801796 reg_l2 13.818474
loss 4.822847
STEP 27 ================================
prereg loss 0.6425049 reg_l1 41.77172 reg_l2 13.81113
loss 4.819677
cutoff 1.5535988e-8 network size 728
STEP 28 ================================
prereg loss 0.6427335 reg_l1 41.7334 reg_l2 13.803917
loss 4.8160734
STEP 29 ================================
prereg loss 0.6428751 reg_l1 41.69705 reg_l2 13.7967825
loss 4.81258
cutoff 7.8001904e-8 network size 727
STEP 30 ================================
prereg loss 0.6442077 reg_l1 41.660778 reg_l2 13.789658
loss 4.8102856
STEP 31 ================================
prereg loss 0.6452386 reg_l1 41.62715 reg_l2 13.782574
loss 4.807954
cutoff 5.8418664e-7 network size 726
STEP 32 ================================
prereg loss 0.6455467 reg_l1 41.597973 reg_l2 13.775596
loss 4.8053436
STEP 33 ================================
prereg loss 0.6471414 reg_l1 41.564842 reg_l2 13.768494
loss 4.8036256
cutoff 1.8208084e-7 network size 725
STEP 34 ================================
prereg loss 0.64956254 reg_l1 41.52992 reg_l2 13.7613735
loss 4.802554
STEP 35 ================================
prereg loss 0.6511197 reg_l1 41.497417 reg_l2 13.754263
loss 4.8008614
cutoff 1.7424463e-7 network size 724
STEP 36 ================================
prereg loss 0.6522641 reg_l1 41.465595 reg_l2 13.747257
loss 4.798824
STEP 37 ================================
prereg loss 0.65361536 reg_l1 41.433376 reg_l2 13.740217
loss 4.796953
cutoff 9.508076e-7 network size 723
STEP 38 ================================
prereg loss 0.65569496 reg_l1 41.399105 reg_l2 13.733125
loss 4.7956057
STEP 39 ================================
prereg loss 0.65718675 reg_l1 41.367077 reg_l2 13.725887
loss 4.793895
cutoff 7.994604e-7 network size 722
STEP 40 ================================
prereg loss 0.6577151 reg_l1 41.333645 reg_l2 13.718498
loss 4.7910795
STEP 41 ================================
prereg loss 0.65854645 reg_l1 41.304024 reg_l2 13.711036
loss 4.788949
cutoff 3.1387356e-7 network size 721
STEP 42 ================================
prereg loss 0.66019243 reg_l1 41.26944 reg_l2 13.70356
loss 4.7871366
STEP 43 ================================
prereg loss 0.6626373 reg_l1 41.234158 reg_l2 13.696135
loss 4.786053
cutoff 7.615454e-7 network size 720
STEP 44 ================================
prereg loss 0.6661524 reg_l1 41.19756 reg_l2 13.688777
loss 4.7859087
STEP 45 ================================
prereg loss 0.6684325 reg_l1 41.16522 reg_l2 13.68147
loss 4.784954
cutoff 1.9072468e-7 network size 719
STEP 46 ================================
prereg loss 0.668791 reg_l1 41.137398 reg_l2 13.674181
loss 4.782531
STEP 47 ================================
prereg loss 0.67061925 reg_l1 41.10559 reg_l2 13.666857
loss 4.7811785
cutoff 1.1971424e-6 network size 718
STEP 48 ================================
prereg loss 0.6727792 reg_l1 41.070526 reg_l2 13.659425
loss 4.779832
STEP 49 ================================
prereg loss 0.6739851 reg_l1 41.036213 reg_l2 13.651913
loss 4.7776065
cutoff 7.259059e-7 network size 717
STEP 50 ================================
prereg loss 0.6761429 reg_l1 41.004387 reg_l2 13.644366
loss 4.7765813
STEP 51 ================================
prereg loss 0.67862797 reg_l1 40.97259 reg_l2 13.636902
loss 4.775887
cutoff 6.908085e-7 network size 716
STEP 52 ================================
prereg loss 0.6801911 reg_l1 40.933468 reg_l2 13.629756
loss 4.773538
STEP 53 ================================
prereg loss 0.682272 reg_l1 40.90253 reg_l2 13.622879
loss 4.7725253
cutoff 5.6345925e-7 network size 715
STEP 54 ================================
prereg loss 0.68384564 reg_l1 40.871864 reg_l2 13.616108
loss 4.771032
STEP 55 ================================
prereg loss 0.684415 reg_l1 40.839424 reg_l2 13.6092825
loss 4.7683573
cutoff 2.6548332e-7 network size 714
STEP 56 ================================
prereg loss 0.68550557 reg_l1 40.803345 reg_l2 13.602384
loss 4.76584
STEP 57 ================================
prereg loss 0.6864593 reg_l1 40.770805 reg_l2 13.595349
loss 4.7635403
cutoff 6.562477e-7 network size 713
STEP 58 ================================
prereg loss 0.6875613 reg_l1 40.73989 reg_l2 13.588483
loss 4.761551
STEP 59 ================================
prereg loss 0.6899959 reg_l1 40.706036 reg_l2 13.581822
loss 4.7605996
cutoff 2.6299858e-8 network size 712
STEP 60 ================================
prereg loss 0.693591 reg_l1 40.672832 reg_l2 13.575352
loss 4.7608743
STEP 61 ================================
prereg loss 0.69654614 reg_l1 40.64236 reg_l2 13.5690365
loss 4.7607822
cutoff 6.95778e-7 network size 711
STEP 62 ================================
prereg loss 0.6981194 reg_l1 40.612083 reg_l2 13.56274
loss 4.759328
STEP 63 ================================
prereg loss 0.7000773 reg_l1 40.58361 reg_l2 13.556222
loss 4.758438
cutoff 1.4300749e-6 network size 710
STEP 64 ================================
prereg loss 0.7015719 reg_l1 40.55026 reg_l2 13.549531
loss 4.756598
STEP 65 ================================
prereg loss 0.70262027 reg_l1 40.51869 reg_l2 13.542753
loss 4.754489
cutoff 1.1328302e-7 network size 709
STEP 66 ================================
prereg loss 0.7043129 reg_l1 40.488354 reg_l2 13.535987
loss 4.753148
STEP 67 ================================
prereg loss 0.70624137 reg_l1 40.45357 reg_l2 13.529371
loss 4.7515984
cutoff 1.4717807e-7 network size 708
STEP 68 ================================
prereg loss 0.7067005 reg_l1 40.41814 reg_l2 13.522918
loss 4.7485147
STEP 69 ================================
prereg loss 0.70622987 reg_l1 40.387344 reg_l2 13.516639
loss 4.744964
cutoff 2.9879777e-7 network size 707
STEP 70 ================================
prereg loss 0.7053729 reg_l1 40.356174 reg_l2 13.510353
loss 4.74099
STEP 71 ================================
prereg loss 0.7063833 reg_l1 40.320972 reg_l2 13.503783
loss 4.7384806
cutoff 2.260058e-7 network size 706
STEP 72 ================================
prereg loss 0.7076588 reg_l1 40.281166 reg_l2 13.496982
loss 4.7357755
STEP 73 ================================
prereg loss 0.7095579 reg_l1 40.24845 reg_l2 13.490312
loss 4.734403
cutoff 1.9773097e-8 network size 705
STEP 74 ================================
prereg loss 0.71151334 reg_l1 40.219448 reg_l2 13.484037
loss 4.7334585
STEP 75 ================================
prereg loss 0.71080714 reg_l1 40.18663 reg_l2 13.478165
loss 4.7294703
cutoff 2.0994048e-7 network size 704
STEP 76 ================================
prereg loss 0.7099199 reg_l1 40.153828 reg_l2 13.472511
loss 4.7253027
STEP 77 ================================
prereg loss 0.7116016 reg_l1 40.11984 reg_l2 13.466776
loss 4.7235856
cutoff 1.829103e-7 network size 703
STEP 78 ================================
prereg loss 0.7130873 reg_l1 40.091198 reg_l2 13.460726
loss 4.722207
STEP 79 ================================
prereg loss 0.7153505 reg_l1 40.060898 reg_l2 13.454249
loss 4.7214403
cutoff 4.5289198e-8 network size 702
STEP 80 ================================
prereg loss 0.7183473 reg_l1 40.02845 reg_l2 13.447666
loss 4.7211924
STEP 81 ================================
prereg loss 0.72086823 reg_l1 39.99886 reg_l2 13.441279
loss 4.720754
cutoff 7.837116e-8 network size 701
STEP 82 ================================
prereg loss 0.72183913 reg_l1 39.970543 reg_l2 13.435259
loss 4.7188935
STEP 83 ================================
prereg loss 0.7231434 reg_l1 39.941452 reg_l2 13.4294615
loss 4.7172885
cutoff 1.2515084e-6 network size 700
STEP 84 ================================
prereg loss 0.72438776 reg_l1 39.91314 reg_l2 13.423754
loss 4.7157016
STEP 85 ================================
prereg loss 0.72636896 reg_l1 39.885017 reg_l2 13.418046
loss 4.714871
cutoff 1.33914e-8 network size 699
STEP 86 ================================
prereg loss 0.72820973 reg_l1 39.85745 reg_l2 13.412313
loss 4.713955
STEP 87 ================================
prereg loss 0.7313335 reg_l1 39.82953 reg_l2 13.406664
loss 4.7142863
cutoff 4.9815026e-7 network size 698
STEP 88 ================================
prereg loss 0.73205477 reg_l1 39.799126 reg_l2 13.401167
loss 4.7119675
STEP 89 ================================
prereg loss 0.73123145 reg_l1 39.776115 reg_l2 13.395745
loss 4.708843
cutoff 4.3122782e-8 network size 697
STEP 90 ================================
prereg loss 0.73229194 reg_l1 39.750217 reg_l2 13.390276
loss 4.7073135
STEP 91 ================================
prereg loss 0.73382074 reg_l1 39.722424 reg_l2 13.384617
loss 4.7060633
cutoff 1.2000237e-6 network size 696
STEP 92 ================================
prereg loss 0.7373612 reg_l1 39.691765 reg_l2 13.378769
loss 4.7065377
STEP 93 ================================
prereg loss 0.74308205 reg_l1 39.66071 reg_l2 13.372962
loss 4.709153
cutoff 4.501435e-7 network size 695
STEP 94 ================================
prereg loss 0.74440503 reg_l1 39.634396 reg_l2 13.367217
loss 4.7078447
STEP 95 ================================
prereg loss 0.7431626 reg_l1 39.606236 reg_l2 13.361598
loss 4.703786
cutoff 6.5574386e-7 network size 694
STEP 96 ================================
prereg loss 0.7417174 reg_l1 39.57977 reg_l2 13.356105
loss 4.6996946
STEP 97 ================================
prereg loss 0.74046016 reg_l1 39.557423 reg_l2 13.350522
loss 4.6962023
cutoff 1.7611455e-7 network size 693
STEP 98 ================================
prereg loss 0.7407217 reg_l1 39.53248 reg_l2 13.344793
loss 4.6939697
STEP 99 ================================
prereg loss 0.74063635 reg_l1 39.5051 reg_l2 13.33887
loss 4.6911464
cutoff 1.2889723e-7 network size 692
STEP 100 ================================
prereg loss 0.7394225 reg_l1 39.476444 reg_l2 13.332894
loss 4.687067
STEP 101 ================================
prereg loss 0.738973 reg_l1 39.450035 reg_l2 13.326977
loss 4.6839767
cutoff 1.0519725e-6 network size 691
STEP 102 ================================
prereg loss 0.73990554 reg_l1 39.426655 reg_l2 13.321336
loss 4.682571
STEP 103 ================================
prereg loss 0.7399548 reg_l1 39.403225 reg_l2 13.315969
loss 4.6802773
cutoff 1.11351255e-7 network size 690
STEP 104 ================================
prereg loss 0.7379484 reg_l1 39.37726 reg_l2 13.310739
loss 4.6756744
STEP 105 ================================
prereg loss 0.73683757 reg_l1 39.353443 reg_l2 13.30558
loss 4.672182
cutoff 1.4931138e-6 network size 689
STEP 106 ================================
prereg loss 0.7377162 reg_l1 39.330204 reg_l2 13.300517
loss 4.6707363
STEP 107 ================================
prereg loss 0.738833 reg_l1 39.305893 reg_l2 13.2956295
loss 4.6694226
cutoff 3.9432052e-8 network size 688
STEP 108 ================================
prereg loss 0.74159247 reg_l1 39.281113 reg_l2 13.29093
loss 4.669704
STEP 109 ================================
prereg loss 0.74219984 reg_l1 39.257107 reg_l2 13.286357
loss 4.6679106
cutoff 3.769892e-7 network size 687
STEP 110 ================================
prereg loss 0.7422323 reg_l1 39.232147 reg_l2 13.281814
loss 4.665447
STEP 111 ================================
prereg loss 0.7438581 reg_l1 39.206505 reg_l2 13.2771635
loss 4.664509
cutoff 1.5821479e-6 network size 686
STEP 112 ================================
prereg loss 0.7457684 reg_l1 39.182186 reg_l2 13.272343
loss 4.663987
STEP 113 ================================
prereg loss 0.7490512 reg_l1 39.159542 reg_l2 13.267358
loss 4.6650057
cutoff 1.5426485e-6 network size 685
STEP 114 ================================
prereg loss 0.75025594 reg_l1 39.132046 reg_l2 13.262275
loss 4.6634607
STEP 115 ================================
prereg loss 0.75080657 reg_l1 39.10472 reg_l2 13.257324
loss 4.6612787
cutoff 3.1164564e-7 network size 684
STEP 116 ================================
prereg loss 0.7497765 reg_l1 39.0796 reg_l2 13.252684
loss 4.657737
STEP 117 ================================
prereg loss 0.7467117 reg_l1 39.059437 reg_l2 13.248379
loss 4.652655
cutoff 5.025504e-8 network size 683
STEP 118 ================================
prereg loss 0.74699736 reg_l1 39.03788 reg_l2 13.244306
loss 4.6507854
STEP 119 ================================
prereg loss 0.7452305 reg_l1 39.012413 reg_l2 13.240223
loss 4.646472
cutoff 1.3045883e-7 network size 682
STEP 120 ================================
prereg loss 0.7448854 reg_l1 38.99044 reg_l2 13.235999
loss 4.6439295
STEP 121 ================================
prereg loss 0.74606913 reg_l1 38.970375 reg_l2 13.231498
loss 4.6431065
cutoff 9.417272e-8 network size 681
STEP 122 ================================
prereg loss 0.745124 reg_l1 38.946213 reg_l2 13.226831
loss 4.639745
STEP 123 ================================
prereg loss 0.7430031 reg_l1 38.91665 reg_l2 13.222082
loss 4.634668
cutoff 7.998242e-7 network size 680
STEP 124 ================================
prereg loss 0.740886 reg_l1 38.89051 reg_l2 13.21745
loss 4.629937
STEP 125 ================================
prereg loss 0.74079853 reg_l1 38.86914 reg_l2 13.212846
loss 4.6277127
cutoff 7.99555e-8 network size 679
STEP 126 ================================
prereg loss 0.7408102 reg_l1 38.848446 reg_l2 13.208247
loss 4.6256547
STEP 127 ================================
prereg loss 0.741372 reg_l1 38.824696 reg_l2 13.2036915
loss 4.623842
cutoff 2.7562783e-7 network size 678
STEP 128 ================================
prereg loss 0.740817 reg_l1 38.798706 reg_l2 13.199257
loss 4.6206875
STEP 129 ================================
prereg loss 0.7394774 reg_l1 38.78118 reg_l2 13.194933
loss 4.6175957
cutoff 8.259303e-8 network size 677
STEP 130 ================================
prereg loss 0.7392047 reg_l1 38.761524 reg_l2 13.190591
loss 4.6153574
STEP 131 ================================
prereg loss 0.73977923 reg_l1 38.738636 reg_l2 13.186186
loss 4.6136427
cutoff 8.9969035e-8 network size 676
STEP 132 ================================
prereg loss 0.74021107 reg_l1 38.714756 reg_l2 13.181727
loss 4.6116867
STEP 133 ================================
prereg loss 0.73934275 reg_l1 38.693993 reg_l2 13.177351
loss 4.608742
cutoff 4.8879156e-7 network size 675
STEP 134 ================================
prereg loss 0.7389791 reg_l1 38.66841 reg_l2 13.173206
loss 4.60582
STEP 135 ================================
prereg loss 0.7393522 reg_l1 38.64772 reg_l2 13.169408
loss 4.604124
cutoff 1.3658428e-7 network size 674
STEP 136 ================================
prereg loss 0.7385037 reg_l1 38.626526 reg_l2 13.16591
loss 4.601156
STEP 137 ================================
prereg loss 0.73711514 reg_l1 38.606037 reg_l2 13.162547
loss 4.597719
cutoff 1.8658943e-7 network size 673
STEP 138 ================================
prereg loss 0.7351604 reg_l1 38.581284 reg_l2 13.159096
loss 4.593289
STEP 139 ================================
prereg loss 0.7353316 reg_l1 38.554924 reg_l2 13.155498
loss 4.590824
cutoff 5.438794e-7 network size 672
STEP 140 ================================
prereg loss 0.7378179 reg_l1 38.530506 reg_l2 13.151805
loss 4.5908685
STEP 141 ================================
prereg loss 0.7389911 reg_l1 38.511555 reg_l2 13.148088
loss 4.5901465
cutoff 6.1956234e-7 network size 671
STEP 142 ================================
prereg loss 0.7395695 reg_l1 38.48983 reg_l2 13.144309
loss 4.5885525
STEP 143 ================================
prereg loss 0.7383829 reg_l1 38.46339 reg_l2 13.140599
loss 4.584722
cutoff 5.416223e-8 network size 670
STEP 144 ================================
prereg loss 0.73744607 reg_l1 38.43936 reg_l2 13.137026
loss 4.5813823
STEP 145 ================================
prereg loss 0.73589987 reg_l1 38.420185 reg_l2 13.133663
loss 4.5779185
cutoff 1.283137e-6 network size 669
STEP 146 ================================
prereg loss 0.73450994 reg_l1 38.40157 reg_l2 13.130405
loss 4.574667
STEP 147 ================================
prereg loss 0.7342156 reg_l1 38.37706 reg_l2 13.126967
loss 4.571922
cutoff 1.2941382e-7 network size 668
STEP 148 ================================
prereg loss 0.7346599 reg_l1 38.352592 reg_l2 13.123228
loss 4.569919
STEP 149 ================================
prereg loss 0.73387617 reg_l1 38.327595 reg_l2 13.119302
loss 4.5666356
cutoff 6.4240885e-7 network size 667
STEP 150 ================================
prereg loss 0.7351527 reg_l1 38.304714 reg_l2 13.115395
loss 4.565624
STEP 151 ================================
prereg loss 0.7388169 reg_l1 38.28233 reg_l2 13.111759
loss 4.56705
cutoff 3.1462696e-7 network size 666
STEP 152 ================================
prereg loss 0.73743093 reg_l1 38.255157 reg_l2 13.10858
loss 4.562947
STEP 153 ================================
prereg loss 0.7331168 reg_l1 38.23545 reg_l2 13.105872
loss 4.556662
cutoff 1.2039891e-8 network size 665
STEP 154 ================================
prereg loss 0.73170376 reg_l1 38.220978 reg_l2 13.103421
loss 4.5538015
STEP 155 ================================
prereg loss 0.7284027 reg_l1 38.202225 reg_l2 13.100948
loss 4.5486255
cutoff 1.8514402e-7 network size 664
STEP 156 ================================
prereg loss 0.72864723 reg_l1 38.17756 reg_l2 13.098085
loss 4.546403
STEP 157 ================================
prereg loss 0.73315066 reg_l1 38.15638 reg_l2 13.094774
loss 4.5487885
cutoff 2.30149e-7 network size 663
STEP 158 ================================
prereg loss 0.7360757 reg_l1 38.13251 reg_l2 13.091229
loss 4.549327
STEP 159 ================================
prereg loss 0.733608 reg_l1 38.105034 reg_l2 13.087835
loss 4.5441113
cutoff 8.3198756e-8 network size 662
STEP 160 ================================
prereg loss 0.7329209 reg_l1 38.08052 reg_l2 13.08476
loss 4.5409727
STEP 161 ================================
prereg loss 0.73344505 reg_l1 38.059685 reg_l2 13.081984
loss 4.5394135
cutoff 5.099355e-8 network size 661
STEP 162 ================================
prereg loss 0.73226935 reg_l1 38.040318 reg_l2 13.0793495
loss 4.536301
STEP 163 ================================
prereg loss 0.730123 reg_l1 38.02118 reg_l2 13.076763
loss 4.532241
cutoff 7.1873365e-7 network size 660
STEP 164 ================================
prereg loss 0.7292828 reg_l1 37.998684 reg_l2 13.074073
loss 4.529151
STEP 165 ================================
prereg loss 0.72998846 reg_l1 37.979362 reg_l2 13.071234
loss 4.5279245
cutoff 2.5890768e-7 network size 659
STEP 166 ================================
prereg loss 0.7301996 reg_l1 37.959866 reg_l2 13.068307
loss 4.526186
STEP 167 ================================
prereg loss 0.7286751 reg_l1 37.938343 reg_l2 13.0653925
loss 4.5225096
cutoff 2.8275826e-7 network size 658
STEP 168 ================================
prereg loss 0.72792006 reg_l1 37.918392 reg_l2 13.062622
loss 4.519759
STEP 169 ================================
prereg loss 0.72845495 reg_l1 37.899555 reg_l2 13.060124
loss 4.5184107
cutoff 9.88759e-7 network size 657
STEP 170 ================================
prereg loss 0.72836494 reg_l1 37.882275 reg_l2 13.057932
loss 4.5165925
STEP 171 ================================
prereg loss 0.72765535 reg_l1 37.864964 reg_l2 13.055974
loss 4.5141516
cutoff 2.0075822e-7 network size 656
STEP 172 ================================
prereg loss 0.7261208 reg_l1 37.848953 reg_l2 13.054138
loss 4.5110164
STEP 173 ================================
prereg loss 0.72538656 reg_l1 37.830025 reg_l2 13.05228
loss 4.508389
cutoff 6.3135667e-7 network size 655
STEP 174 ================================
prereg loss 0.7266448 reg_l1 37.81574 reg_l2 13.050266
loss 4.508219
STEP 175 ================================
prereg loss 0.7281516 reg_l1 37.799255 reg_l2 13.048037
loss 4.508077
cutoff 6.037553e-7 network size 654
STEP 176 ================================
prereg loss 0.7295924 reg_l1 37.780464 reg_l2 13.045579
loss 4.507639
STEP 177 ================================
prereg loss 0.7278275 reg_l1 37.764095 reg_l2 13.042852
loss 4.504237
cutoff 1.903827e-7 network size 653
STEP 178 ================================
prereg loss 0.7264376 reg_l1 37.74678 reg_l2 13.039962
loss 4.501116
STEP 179 ================================
prereg loss 0.7258842 reg_l1 37.727943 reg_l2 13.0370655
loss 4.4986787
cutoff 3.3252445e-7 network size 652
STEP 180 ================================
prereg loss 0.7264385 reg_l1 37.708675 reg_l2 13.034295
loss 4.497306
STEP 181 ================================
prereg loss 0.7291433 reg_l1 37.690556 reg_l2 13.031826
loss 4.498199
cutoff 2.4192173e-7 network size 651
STEP 182 ================================
prereg loss 0.7290254 reg_l1 37.672024 reg_l2 13.02972
loss 4.4962277
STEP 183 ================================
prereg loss 0.7267438 reg_l1 37.654747 reg_l2 13.02799
loss 4.4922185
cutoff 3.8269354e-7 network size 650
STEP 184 ================================
prereg loss 0.72628856 reg_l1 37.634422 reg_l2 13.026456
loss 4.489731
STEP 185 ================================
prereg loss 0.72652644 reg_l1 37.616116 reg_l2 13.024842
loss 4.488138
cutoff 3.9392762e-7 network size 649
STEP 186 ================================
prereg loss 0.7270384 reg_l1 37.60108 reg_l2 13.022952
loss 4.4871464
STEP 187 ================================
prereg loss 0.7285 reg_l1 37.584373 reg_l2 13.020675
loss 4.4869375
cutoff 8.8184606e-8 network size 648
STEP 188 ================================
prereg loss 0.7273288 reg_l1 37.56392 reg_l2 13.018165
loss 4.483721
STEP 189 ================================
prereg loss 0.72570056 reg_l1 37.545517 reg_l2 13.015745
loss 4.4802523
cutoff 4.441972e-9 network size 647
STEP 190 ================================
prereg loss 0.7251583 reg_l1 37.529007 reg_l2 13.0135975
loss 4.4780593
STEP 191 ================================
prereg loss 0.72564375 reg_l1 37.515255 reg_l2 13.011801
loss 4.4771695
cutoff 9.0294634e-8 network size 646
STEP 192 ================================
prereg loss 0.7267966 reg_l1 37.49993 reg_l2 13.01039
loss 4.4767895
STEP 193 ================================
prereg loss 0.72795266 reg_l1 37.48587 reg_l2 13.009187
loss 4.4765396
cutoff 6.6274333e-7 network size 645
STEP 194 ================================
prereg loss 0.72650045 reg_l1 37.469795 reg_l2 13.007986
loss 4.4734797
STEP 195 ================================
prereg loss 0.7252721 reg_l1 37.457012 reg_l2 13.006582
loss 4.4709735
cutoff 2.549641e-7 network size 644
STEP 196 ================================
prereg loss 0.72518355 reg_l1 37.44134 reg_l2 13.004822
loss 4.469318
STEP 197 ================================
prereg loss 0.72569287 reg_l1 37.425667 reg_l2 13.002771
loss 4.46826
cutoff 7.4875777e-7 network size 643
STEP 198 ================================
prereg loss 0.7253646 reg_l1 37.407978 reg_l2 13.000592
loss 4.466162
STEP 199 ================================
prereg loss 0.7248992 reg_l1 37.38891 reg_l2 12.998639
loss 4.46379
cutoff 1.2744204e-7 network size 642
STEP 200 ================================
prereg loss 0.7273236 reg_l1 37.374184 reg_l2 12.997209
loss 4.464742
2022-08-14T13:55:23.609

julia> serialize("cf-s2-642-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-642-parameters-opt.ser", opt)
```

Still going quite smooth. Let's start doing some checkpointing early,
the first serialization is above.

Also, the number of parameters is now small enough that the effective regularization
is no longer excessive, the preregularization loss has started to go down.

Repeating 100 sparsifications in 200 steps:

```
julia> interleaving_steps!(200)
2022-08-14T14:00:03.003
STEP 1 ================================
prereg loss 0.72920847 reg_l1 37.357548 reg_l2 12.996368
loss 4.464963
cutoff 1.1025986e-7 network size 641
STEP 2 ================================
prereg loss 0.7270659 reg_l1 37.345333 reg_l2 12.995934
loss 4.4615993
STEP 3 ================================
prereg loss 0.72532505 reg_l1 37.332256 reg_l2 12.995398
loss 4.4585505
cutoff 4.860194e-7 network size 640
STEP 4 ================================
prereg loss 0.72486985 reg_l1 37.318726 reg_l2 12.994315
loss 4.4567423
STEP 5 ================================
prereg loss 0.7234509 reg_l1 37.30443 reg_l2 12.992419
loss 4.4538937
cutoff 3.8015069e-6 network size 639
STEP 6 ================================
prereg loss 0.7221816 reg_l1 37.29196 reg_l2 12.99002
loss 4.451378
STEP 7 ================================
prereg loss 0.724004 reg_l1 37.274467 reg_l2 12.987697
loss 4.451451
cutoff 1.3460522e-7 network size 638
STEP 8 ================================
prereg loss 0.72771215 reg_l1 37.25659 reg_l2 12.98605
loss 4.453371
STEP 9 ================================
prereg loss 0.7277344 reg_l1 37.242825 reg_l2 12.98525
loss 4.452017
cutoff 7.9448364e-7 network size 637
STEP 10 ================================
prereg loss 0.72230047 reg_l1 37.234283 reg_l2 12.985154
loss 4.445729
STEP 11 ================================
prereg loss 0.71943086 reg_l1 37.22551 reg_l2 12.985127
loss 4.441982
cutoff 8.5751157e-7 network size 636
STEP 12 ================================
prereg loss 0.72039884 reg_l1 37.214256 reg_l2 12.984497
loss 4.4418244
STEP 13 ================================
prereg loss 0.7163587 reg_l1 37.20081 reg_l2 12.983091
loss 4.43644
cutoff 4.476169e-8 network size 635
STEP 14 ================================
prereg loss 0.71341705 reg_l1 37.18568 reg_l2 12.98121
loss 4.431985
STEP 15 ================================
prereg loss 0.7192974 reg_l1 37.169147 reg_l2 12.979186
loss 4.4362125
cutoff 6.253406e-7 network size 634
STEP 16 ================================
prereg loss 0.7203763 reg_l1 37.155148 reg_l2 12.977489
loss 4.435891
STEP 17 ================================
prereg loss 0.71558136 reg_l1 37.139915 reg_l2 12.97635
loss 4.429573
cutoff 2.769786e-6 network size 633
STEP 18 ================================
prereg loss 0.71246165 reg_l1 37.128685 reg_l2 12.97567
loss 4.42533
STEP 19 ================================
prereg loss 0.71166867 reg_l1 37.11535 reg_l2 12.975112
loss 4.4232035
cutoff 2.6753787e-7 network size 632
STEP 20 ================================
prereg loss 0.7106356 reg_l1 37.101242 reg_l2 12.974466
loss 4.4207597
STEP 21 ================================
prereg loss 0.7098033 reg_l1 37.086834 reg_l2 12.973528
loss 4.4184866
cutoff 5.7983925e-7 network size 631
STEP 22 ================================
prereg loss 0.7090552 reg_l1 37.073555 reg_l2 12.972242
loss 4.4164104
STEP 23 ================================
prereg loss 0.70850414 reg_l1 37.061733 reg_l2 12.970661
loss 4.4146776
cutoff 1.6666672e-7 network size 630
STEP 24 ================================
prereg loss 0.708497 reg_l1 37.04987 reg_l2 12.968954
loss 4.413484
STEP 25 ================================
prereg loss 0.7083417 reg_l1 37.037292 reg_l2 12.967321
loss 4.412071
cutoff 5.598522e-7 network size 629
STEP 26 ================================
prereg loss 0.7078627 reg_l1 37.025505 reg_l2 12.965873
loss 4.4104133
STEP 27 ================================
prereg loss 0.7067216 reg_l1 37.010548 reg_l2 12.9647665
loss 4.4077764
cutoff 2.8946494e-7 network size 628
STEP 28 ================================
prereg loss 0.7054917 reg_l1 36.999454 reg_l2 12.963909
loss 4.405437
STEP 29 ================================
prereg loss 0.7052801 reg_l1 36.98962 reg_l2 12.963144
loss 4.404242
cutoff 9.924406e-9 network size 627
STEP 30 ================================
prereg loss 0.7055714 reg_l1 36.976788 reg_l2 12.962219
loss 4.40325
STEP 31 ================================
prereg loss 0.70616055 reg_l1 36.962505 reg_l2 12.961046
loss 4.4024115
cutoff 2.7930582e-9 network size 626
STEP 32 ================================
prereg loss 0.70584583 reg_l1 36.948174 reg_l2 12.959737
loss 4.4006634
STEP 33 ================================
prereg loss 0.7035455 reg_l1 36.93615 reg_l2 12.958383
loss 4.3971605
cutoff 8.701272e-8 network size 625
STEP 34 ================================
prereg loss 0.7037921 reg_l1 36.925846 reg_l2 12.956849
loss 4.3963766
STEP 35 ================================
prereg loss 0.704167 reg_l1 36.911076 reg_l2 12.954974
loss 4.3952746
cutoff 4.415051e-8 network size 624
STEP 36 ================================
prereg loss 0.7032635 reg_l1 36.89168 reg_l2 12.952945
loss 4.3924317
STEP 37 ================================
prereg loss 0.7008394 reg_l1 36.87245 reg_l2 12.951073
loss 4.3880844
cutoff 1.1535576e-8 network size 623
STEP 38 ================================
prereg loss 0.70003265 reg_l1 36.856358 reg_l2 12.949444
loss 4.3856683
STEP 39 ================================
prereg loss 0.70029575 reg_l1 36.83724 reg_l2 12.948072
loss 4.38402
cutoff 2.8793966e-7 network size 622
STEP 40 ================================
prereg loss 0.7027429 reg_l1 36.818424 reg_l2 12.946717
loss 4.3845854
STEP 41 ================================
prereg loss 0.70401627 reg_l1 36.803215 reg_l2 12.94538
loss 4.384338
cutoff 5.020047e-7 network size 621
STEP 42 ================================
prereg loss 0.70002794 reg_l1 36.787598 reg_l2 12.944012
loss 4.378788
STEP 43 ================================
prereg loss 0.6983027 reg_l1 36.771244 reg_l2 12.942596
loss 4.3754272
cutoff 6.424998e-7 network size 620
STEP 44 ================================
prereg loss 0.69778466 reg_l1 36.755882 reg_l2 12.941149
loss 4.373373
STEP 45 ================================
prereg loss 0.69604725 reg_l1 36.739197 reg_l2 12.939715
loss 4.369967
cutoff 5.8659907e-8 network size 619
STEP 46 ================================
prereg loss 0.6933513 reg_l1 36.72377 reg_l2 12.93844
loss 4.3657284
STEP 47 ================================
prereg loss 0.6923352 reg_l1 36.710537 reg_l2 12.937476
loss 4.363389
cutoff 2.2474524e-7 network size 618
STEP 48 ================================
prereg loss 0.69275236 reg_l1 36.697235 reg_l2 12.936748
loss 4.362476
STEP 49 ================================
prereg loss 0.68916494 reg_l1 36.684868 reg_l2 12.936178
loss 4.3576517
cutoff 4.2644524e-7 network size 617
STEP 50 ================================
prereg loss 0.68858856 reg_l1 36.67467 reg_l2 12.935462
loss 4.3560557
STEP 51 ================================
prereg loss 0.6896809 reg_l1 36.659298 reg_l2 12.934327
loss 4.355611
cutoff 4.6445348e-7 network size 616
STEP 52 ================================
prereg loss 0.6870267 reg_l1 36.64052 reg_l2 12.932861
loss 4.3510785
STEP 53 ================================
prereg loss 0.6881494 reg_l1 36.622524 reg_l2 12.931419
loss 4.350402
cutoff 1.0471122e-6 network size 615
STEP 54 ================================
prereg loss 0.6906417 reg_l1 36.605675 reg_l2 12.930354
loss 4.351209
STEP 55 ================================
prereg loss 0.6858424 reg_l1 36.59152 reg_l2 12.929684
loss 4.344994
cutoff 1.1262982e-6 network size 614
STEP 56 ================================
prereg loss 0.68221813 reg_l1 36.574764 reg_l2 12.929006
loss 4.3396945
STEP 57 ================================
prereg loss 0.6814889 reg_l1 36.557255 reg_l2 12.928048
loss 4.3372145
cutoff 2.5724148e-7 network size 613
STEP 58 ================================
prereg loss 0.68074226 reg_l1 36.542656 reg_l2 12.926719
loss 4.3350077
STEP 59 ================================
prereg loss 0.68082327 reg_l1 36.52518 reg_l2 12.925272
loss 4.333341
cutoff 9.3998096e-8 network size 612
STEP 60 ================================
prereg loss 0.6823042 reg_l1 36.505283 reg_l2 12.923943
loss 4.332833
STEP 61 ================================
prereg loss 0.68065935 reg_l1 36.491566 reg_l2 12.922861
loss 4.329816
cutoff 3.1418222e-6 network size 611
STEP 62 ================================
prereg loss 0.68015003 reg_l1 36.47848 reg_l2 12.92207
loss 4.327998
STEP 63 ================================
prereg loss 0.67997533 reg_l1 36.46152 reg_l2 12.921329
loss 4.3261275
cutoff 1.9938125e-7 network size 610
STEP 64 ================================
prereg loss 0.6772771 reg_l1 36.44406 reg_l2 12.920406
loss 4.321683
STEP 65 ================================
prereg loss 0.678201 reg_l1 36.42824 reg_l2 12.919217
loss 4.3210254
cutoff 3.197766e-6 network size 609
STEP 66 ================================
prereg loss 0.6817383 reg_l1 36.40898 reg_l2 12.917908
loss 4.3226366
STEP 67 ================================
prereg loss 0.6806679 reg_l1 36.389645 reg_l2 12.916637
loss 4.3196325
cutoff 4.9785285e-8 network size 608
STEP 68 ================================
prereg loss 0.6758617 reg_l1 36.374462 reg_l2 12.915551
loss 4.313308
STEP 69 ================================
prereg loss 0.6732228 reg_l1 36.3617 reg_l2 12.914618
loss 4.309393
cutoff 2.4519977e-7 network size 607
STEP 70 ================================
prereg loss 0.67298543 reg_l1 36.34559 reg_l2 12.913797
loss 4.307544
STEP 71 ================================
prereg loss 0.67200637 reg_l1 36.330727 reg_l2 12.913033
loss 4.305079
cutoff 6.8388715e-8 network size 606
STEP 72 ================================
prereg loss 0.6702066 reg_l1 36.319405 reg_l2 12.9122095
loss 4.3021474
STEP 73 ================================
prereg loss 0.6705464 reg_l1 36.309513 reg_l2 12.911337
loss 4.301498
cutoff 3.640298e-7 network size 605
STEP 74 ================================
prereg loss 0.6719991 reg_l1 36.293404 reg_l2 12.91037
loss 4.3013396
STEP 75 ================================
prereg loss 0.6723364 reg_l1 36.27874 reg_l2 12.909381
loss 4.3002105
cutoff 3.3507604e-7 network size 604
STEP 76 ================================
prereg loss 0.6704213 reg_l1 36.262943 reg_l2 12.908433
loss 4.2967157
STEP 77 ================================
prereg loss 0.6683731 reg_l1 36.251183 reg_l2 12.907495
loss 4.2934914
cutoff 1.5780643e-7 network size 603
STEP 78 ================================
prereg loss 0.66660655 reg_l1 36.23654 reg_l2 12.906447
loss 4.290261
STEP 79 ================================
prereg loss 0.6667301 reg_l1 36.219368 reg_l2 12.905278
loss 4.2886667
cutoff 1.6535814e-6 network size 602
STEP 80 ================================
prereg loss 0.66666764 reg_l1 36.20305 reg_l2 12.904055
loss 4.2869725
STEP 81 ================================
prereg loss 0.6647407 reg_l1 36.187454 reg_l2 12.902957
loss 4.2834864
cutoff 3.0595402e-7 network size 601
STEP 82 ================================
prereg loss 0.6636497 reg_l1 36.173817 reg_l2 12.901947
loss 4.2810316
STEP 83 ================================
prereg loss 0.6662663 reg_l1 36.157578 reg_l2 12.90115
loss 4.282024
cutoff 7.041672e-7 network size 600
STEP 84 ================================
prereg loss 0.6686353 reg_l1 36.141407 reg_l2 12.900642
loss 4.282776
STEP 85 ================================
prereg loss 0.6645625 reg_l1 36.12749 reg_l2 12.900368
loss 4.277312
cutoff 7.918843e-7 network size 599
STEP 86 ================================
prereg loss 0.6598313 reg_l1 36.11523 reg_l2 12.900059
loss 4.2713547
STEP 87 ================================
prereg loss 0.6602418 reg_l1 36.103504 reg_l2 12.899373
loss 4.270592
cutoff 7.504423e-8 network size 598
STEP 88 ================================
prereg loss 0.6598362 reg_l1 36.088 reg_l2 12.898257
loss 4.268636
STEP 89 ================================
prereg loss 0.658217 reg_l1 36.073647 reg_l2 12.896954
loss 4.2655816
cutoff 3.3694232e-7 network size 597
STEP 90 ================================
prereg loss 0.65839726 reg_l1 36.062878 reg_l2 12.895828
loss 4.264685
STEP 91 ================================
prereg loss 0.6599403 reg_l1 36.04661 reg_l2 12.895118
loss 4.2646017
cutoff 1.1107768e-6 network size 596
STEP 92 ================================
prereg loss 0.6583699 reg_l1 36.026474 reg_l2 12.894823
loss 4.2610173
STEP 93 ================================
prereg loss 0.6562557 reg_l1 36.01173 reg_l2 12.89485
loss 4.257429
cutoff 6.2324034e-8 network size 595
STEP 94 ================================
prereg loss 0.6534792 reg_l1 36.0004 reg_l2 12.894968
loss 4.2535195
STEP 95 ================================
prereg loss 0.6526446 reg_l1 35.98697 reg_l2 12.894966
loss 4.2513413
cutoff 1.9278286e-7 network size 594
STEP 96 ================================
prereg loss 0.65194917 reg_l1 35.97264 reg_l2 12.894684
loss 4.249213
STEP 97 ================================
prereg loss 0.65029997 reg_l1 35.95884 reg_l2 12.8941555
loss 4.246184
cutoff 3.6369602e-7 network size 593
STEP 98 ================================
prereg loss 0.6495263 reg_l1 35.943024 reg_l2 12.8933935
loss 4.243829
STEP 99 ================================
prereg loss 0.65061796 reg_l1 35.92681 reg_l2 12.892574
loss 4.243299
cutoff 3.4099003e-7 network size 592
STEP 100 ================================
prereg loss 0.65313804 reg_l1 35.912266 reg_l2 12.891857
loss 4.2443647
STEP 101 ================================
prereg loss 0.6514987 reg_l1 35.901512 reg_l2 12.891419
loss 4.24165
cutoff 1.2792315e-6 network size 591
STEP 102 ================================
prereg loss 0.64811975 reg_l1 35.890053 reg_l2 12.891295
loss 4.237125
STEP 103 ================================
prereg loss 0.64727587 reg_l1 35.877243 reg_l2 12.891405
loss 4.235
cutoff 1.4808902e-6 network size 590
STEP 104 ================================
prereg loss 0.6475192 reg_l1 35.8619 reg_l2 12.891519
loss 4.2337093
STEP 105 ================================
prereg loss 0.6442258 reg_l1 35.851257 reg_l2 12.891474
loss 4.2293515
cutoff 5.7538045e-7 network size 589
STEP 106 ================================
prereg loss 0.64200735 reg_l1 35.84058 reg_l2 12.891303
loss 4.2260656
STEP 107 ================================
prereg loss 0.64386433 reg_l1 35.824413 reg_l2 12.891016
loss 4.2263055
cutoff 1.3764875e-6 network size 588
STEP 108 ================================
prereg loss 0.6467403 reg_l1 35.81119 reg_l2 12.890745
loss 4.2278595
STEP 109 ================================
prereg loss 0.6446256 reg_l1 35.80097 reg_l2 12.890694
loss 4.2247224
cutoff 7.9564416e-8 network size 587
STEP 110 ================================
prereg loss 0.6390917 reg_l1 35.79338 reg_l2 12.890833
loss 4.2184296
STEP 111 ================================
prereg loss 0.63915646 reg_l1 35.78495 reg_l2 12.89094
loss 4.2176514
cutoff 7.288618e-7 network size 586
STEP 112 ================================
prereg loss 0.63879377 reg_l1 35.771393 reg_l2 12.890828
loss 4.2159333
STEP 113 ================================
prereg loss 0.63647074 reg_l1 35.757507 reg_l2 12.890505
loss 4.2122216
cutoff 5.9430022e-8 network size 585
STEP 114 ================================
prereg loss 0.6375537 reg_l1 35.743977 reg_l2 12.8900385
loss 4.2119513
STEP 115 ================================
prereg loss 0.6413574 reg_l1 35.730263 reg_l2 12.88958
loss 4.214384
cutoff 3.5463745e-7 network size 584
STEP 116 ================================
prereg loss 0.6382197 reg_l1 35.71619 reg_l2 12.889211
loss 4.209839
STEP 117 ================================
prereg loss 0.6341654 reg_l1 35.703556 reg_l2 12.888924
loss 4.204521
cutoff 4.001522e-7 network size 583
STEP 118 ================================
prereg loss 0.634193 reg_l1 35.69528 reg_l2 12.888795
loss 4.203721
STEP 119 ================================
prereg loss 0.63346887 reg_l1 35.685066 reg_l2 12.888925
loss 4.201976
cutoff 1.1753764e-6 network size 582
STEP 120 ================================
prereg loss 0.6313731 reg_l1 35.673218 reg_l2 12.889334
loss 4.1986947
STEP 121 ================================
prereg loss 0.6342398 reg_l1 35.66329 reg_l2 12.889807
loss 4.2005687
cutoff 1.6808917e-7 network size 581
STEP 122 ================================
prereg loss 0.6357813 reg_l1 35.653954 reg_l2 12.890087
loss 4.2011766
STEP 123 ================================
prereg loss 0.6290064 reg_l1 35.646633 reg_l2 12.890113
loss 4.19367
cutoff 1.0983167e-6 network size 580
STEP 124 ================================
prereg loss 0.62832433 reg_l1 35.640015 reg_l2 12.889881
loss 4.192326
STEP 125 ================================
prereg loss 0.6275451 reg_l1 35.62872 reg_l2 12.889443
loss 4.1904173
cutoff 3.3182732e-7 network size 579
STEP 126 ================================
prereg loss 0.62590855 reg_l1 35.613983 reg_l2 12.889022
loss 4.187307
STEP 127 ================================
prereg loss 0.6314328 reg_l1 35.5997 reg_l2 12.888771
loss 4.191403
cutoff 2.9074545e-7 network size 578
STEP 128 ================================
prereg loss 0.6293315 reg_l1 35.58938 reg_l2 12.888813
loss 4.188269
STEP 129 ================================
prereg loss 0.62379044 reg_l1 35.580456 reg_l2 12.888944
loss 4.181836
cutoff 1.5432306e-8 network size 577
STEP 130 ================================
prereg loss 0.62452734 reg_l1 35.569798 reg_l2 12.88894
loss 4.181507
STEP 131 ================================
prereg loss 0.6215985 reg_l1 35.556404 reg_l2 12.888809
loss 4.177239
cutoff 2.0294556e-7 network size 576
STEP 132 ================================
prereg loss 0.62238157 reg_l1 35.54111 reg_l2 12.888682
loss 4.1764927
STEP 133 ================================
prereg loss 0.6250935 reg_l1 35.53044 reg_l2 12.888699
loss 4.178138
cutoff 1.6897875e-6 network size 575
STEP 134 ================================
prereg loss 0.6207222 reg_l1 35.52331 reg_l2 12.888888
loss 4.1730533
STEP 135 ================================
prereg loss 0.6187579 reg_l1 35.51408 reg_l2 12.889034
loss 4.170166
cutoff 5.1484676e-8 network size 574
STEP 136 ================================
prereg loss 0.61897683 reg_l1 35.503357 reg_l2 12.888873
loss 4.1693125
STEP 137 ================================
prereg loss 0.61745024 reg_l1 35.489594 reg_l2 12.8884735
loss 4.1664095
cutoff 1.0743788e-6 network size 573
STEP 138 ================================
prereg loss 0.6171644 reg_l1 35.480324 reg_l2 12.888019
loss 4.165197
STEP 139 ================================
prereg loss 0.61794454 reg_l1 35.469563 reg_l2 12.88783
loss 4.164901
cutoff 8.9606874e-7 network size 572
STEP 140 ================================
prereg loss 0.616536 reg_l1 35.458748 reg_l2 12.887983
loss 4.1624107
STEP 141 ================================
prereg loss 0.61381257 reg_l1 35.45112 reg_l2 12.888328
loss 4.1589246
cutoff 9.248106e-8 network size 571
STEP 142 ================================
prereg loss 0.6134199 reg_l1 35.44407 reg_l2 12.888538
loss 4.157827
STEP 143 ================================
prereg loss 0.61227703 reg_l1 35.434658 reg_l2 12.888545
loss 4.1557426
cutoff 1.3687895e-7 network size 570
STEP 144 ================================
prereg loss 0.6110738 reg_l1 35.4218 reg_l2 12.888449
loss 4.1532536
STEP 145 ================================
prereg loss 0.61040056 reg_l1 35.41114 reg_l2 12.888397
loss 4.1515145
cutoff 3.9615406e-7 network size 569
STEP 146 ================================
prereg loss 0.609701 reg_l1 35.400288 reg_l2 12.888479
loss 4.1497297
STEP 147 ================================
prereg loss 0.60942334 reg_l1 35.389355 reg_l2 12.888552
loss 4.148359
cutoff 6.370137e-7 network size 568
STEP 148 ================================
prereg loss 0.60942906 reg_l1 35.378113 reg_l2 12.8885145
loss 4.14724
STEP 149 ================================
prereg loss 0.60955876 reg_l1 35.367947 reg_l2 12.88833
loss 4.1463532
cutoff 4.991307e-9 network size 567
STEP 150 ================================
prereg loss 0.6071679 reg_l1 35.35814 reg_l2 12.888075
loss 4.142982
STEP 151 ================================
prereg loss 0.6060042 reg_l1 35.34765 reg_l2 12.887789
loss 4.140769
cutoff 3.531295e-7 network size 566
STEP 152 ================================
prereg loss 0.6054132 reg_l1 35.337685 reg_l2 12.887664
loss 4.1391816
STEP 153 ================================
prereg loss 0.6044801 reg_l1 35.327164 reg_l2 12.887828
loss 4.1371965
cutoff 2.8076778e-7 network size 565
STEP 154 ================================
prereg loss 0.60525984 reg_l1 35.317997 reg_l2 12.888132
loss 4.1370597
STEP 155 ================================
prereg loss 0.606289 reg_l1 35.30697 reg_l2 12.888315
loss 4.136986
cutoff 6.1699575e-7 network size 564
STEP 156 ================================
prereg loss 0.60562676 reg_l1 35.296486 reg_l2 12.888244
loss 4.1352754
STEP 157 ================================
prereg loss 0.6034546 reg_l1 35.289112 reg_l2 12.887996
loss 4.132366
cutoff 8.745701e-9 network size 563
STEP 158 ================================
prereg loss 0.60253793 reg_l1 35.277855 reg_l2 12.887695
loss 4.1303234
STEP 159 ================================
prereg loss 0.6019845 reg_l1 35.266495 reg_l2 12.887563
loss 4.128634
cutoff 3.3460674e-7 network size 562
STEP 160 ================================
prereg loss 0.6014026 reg_l1 35.258614 reg_l2 12.887829
loss 4.127264
STEP 161 ================================
prereg loss 0.5996378 reg_l1 35.25201 reg_l2 12.888415
loss 4.124839
cutoff 1.3468962e-6 network size 561
STEP 162 ================================
prereg loss 0.5973868 reg_l1 35.2445 reg_l2 12.889075
loss 4.1218367
STEP 163 ================================
prereg loss 0.59725416 reg_l1 35.23386 reg_l2 12.889369
loss 4.1206403
cutoff 7.9875326e-7 network size 560
STEP 164 ================================
prereg loss 0.6009907 reg_l1 35.220413 reg_l2 12.889167
loss 4.123032
STEP 165 ================================
prereg loss 0.6036067 reg_l1 35.20963 reg_l2 12.888711
loss 4.12457
cutoff 1.0347139e-7 network size 559
STEP 166 ================================
prereg loss 0.5988929 reg_l1 35.200813 reg_l2 12.888288
loss 4.118974
STEP 167 ================================
prereg loss 0.59656715 reg_l1 35.19349 reg_l2 12.888043
loss 4.1159163
cutoff 3.6063284e-8 network size 558
STEP 168 ================================
prereg loss 0.59888786 reg_l1 35.185787 reg_l2 12.888099
loss 4.1174664
STEP 169 ================================
prereg loss 0.59722185 reg_l1 35.178253 reg_l2 12.888406
loss 4.1150475
cutoff 2.4783367e-6 network size 557
STEP 170 ================================
prereg loss 0.59355825 reg_l1 35.168514 reg_l2 12.888865
loss 4.1104097
STEP 171 ================================
prereg loss 0.5923969 reg_l1 35.1565 reg_l2 12.88938
loss 4.108047
cutoff 1.7082202e-6 network size 556
STEP 172 ================================
prereg loss 0.59353495 reg_l1 35.143345 reg_l2 12.889822
loss 4.107869
STEP 173 ================================
prereg loss 0.59489506 reg_l1 35.13306 reg_l2 12.890069
loss 4.108201
cutoff 2.4719338e-7 network size 555
STEP 174 ================================
prereg loss 0.5929947 reg_l1 35.12456 reg_l2 12.890063
loss 4.1054506
STEP 175 ================================
prereg loss 0.59056634 reg_l1 35.11282 reg_l2 12.889803
loss 4.101848
cutoff 8.8890374e-8 network size 554
STEP 176 ================================
prereg loss 0.58945334 reg_l1 35.10019 reg_l2 12.889318
loss 4.0994725
STEP 177 ================================
prereg loss 0.5883837 reg_l1 35.089417 reg_l2 12.888905
loss 4.0973253
cutoff 9.142932e-7 network size 553
STEP 178 ================================
prereg loss 0.5880844 reg_l1 35.082466 reg_l2 12.888808
loss 4.096331
STEP 179 ================================
prereg loss 0.5906769 reg_l1 35.072388 reg_l2 12.88909
loss 4.0979156
cutoff 8.213392e-7 network size 552
STEP 180 ================================
prereg loss 0.5898491 reg_l1 35.06047 reg_l2 12.889636
loss 4.0958962
STEP 181 ================================
prereg loss 0.58599913 reg_l1 35.05284 reg_l2 12.890112
loss 4.0912833
cutoff 3.2732714e-8 network size 551
STEP 182 ================================
prereg loss 0.5858822 reg_l1 35.046642 reg_l2 12.890281
loss 4.0905466
STEP 183 ================================
prereg loss 0.5850036 reg_l1 35.03671 reg_l2 12.890084
loss 4.0886745
cutoff 5.6136923e-7 network size 550
STEP 184 ================================
prereg loss 0.5829097 reg_l1 35.02357 reg_l2 12.889746
loss 4.085267
STEP 185 ================================
prereg loss 0.58351445 reg_l1 35.01383 reg_l2 12.889573
loss 4.084897
cutoff 4.789923e-6 network size 549
STEP 186 ================================
prereg loss 0.58712476 reg_l1 35.00439 reg_l2 12.889745
loss 4.087564
STEP 187 ================================
prereg loss 0.58422476 reg_l1 34.99413 reg_l2 12.8902
loss 4.0836377
cutoff 1.0441763e-6 network size 548
STEP 188 ================================
prereg loss 0.580225 reg_l1 34.98392 reg_l2 12.890655
loss 4.078617
STEP 189 ================================
prereg loss 0.57959455 reg_l1 34.974483 reg_l2 12.890818
loss 4.077043
cutoff 2.1685992e-8 network size 547
STEP 190 ================================
prereg loss 0.5784157 reg_l1 34.96627 reg_l2 12.890685
loss 4.0750427
STEP 191 ================================
prereg loss 0.5797611 reg_l1 34.95672 reg_l2 12.890495
loss 4.0754333
cutoff 7.1870454e-7 network size 546
STEP 192 ================================
prereg loss 0.58231235 reg_l1 34.945564 reg_l2 12.8905325
loss 4.076869
STEP 193 ================================
prereg loss 0.5774827 reg_l1 34.937225 reg_l2 12.890823
loss 4.071205
cutoff 1.3598765e-7 network size 545
STEP 194 ================================
prereg loss 0.5766692 reg_l1 34.932144 reg_l2 12.891211
loss 4.0698833
STEP 195 ================================
prereg loss 0.57712245 reg_l1 34.921978 reg_l2 12.891461
loss 4.06932
cutoff 1.5201285e-6 network size 544
STEP 196 ================================
prereg loss 0.57494694 reg_l1 34.909954 reg_l2 12.891512
loss 4.0659423
STEP 197 ================================
prereg loss 0.57485646 reg_l1 34.901432 reg_l2 12.891489
loss 4.0649996
cutoff 3.401874e-7 network size 543
STEP 198 ================================
prereg loss 0.5773734 reg_l1 34.892166 reg_l2 12.891477
loss 4.06659
STEP 199 ================================
prereg loss 0.5759828 reg_l1 34.880474 reg_l2 12.891576
loss 4.06403
cutoff 2.430541e-6 network size 542
STEP 200 ================================
prereg loss 0.57375556 reg_l1 34.871998 reg_l2 12.891889
loss 4.0609555
2022-08-14T15:08:32.119

julia> serialize("cf-s2-542-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-542-parameters-opt.ser", opt)
```

Another 100 sparsifications in 200 steps (continues to behave well):

```
julia> # let's do another 100 sparsifications in 200 steps

julia> interleaving_steps!(200)
2022-08-14T15:17:47.207
STEP 1 ================================
prereg loss 0.5709342 reg_l1 34.86489 reg_l2 12.892341
loss 4.057423
cutoff 2.1661981e-7 network size 541
STEP 2 ================================
prereg loss 0.57135355 reg_l1 34.858604 reg_l2 12.892663
loss 4.0572143
STEP 3 ================================
prereg loss 0.57067347 reg_l1 34.849144 reg_l2 12.892678
loss 4.055588
cutoff 1.8858327e-7 network size 540
STEP 4 ================================
prereg loss 0.5692943 reg_l1 34.83814 reg_l2 12.892439
loss 4.053108
STEP 5 ================================
prereg loss 0.5704674 reg_l1 34.828197 reg_l2 12.892145
loss 4.053287
cutoff 3.9827137e-7 network size 539
STEP 6 ================================
prereg loss 0.5727826 reg_l1 34.816048 reg_l2 12.89201
loss 4.0543876
STEP 7 ================================
prereg loss 0.57201606 reg_l1 34.80664 reg_l2 12.892159
loss 4.05268
cutoff 1.0246913e-6 network size 538
STEP 8 ================================
prereg loss 0.56760895 reg_l1 34.80075 reg_l2 12.892577
loss 4.047684
STEP 9 ================================
prereg loss 0.5665004 reg_l1 34.795265 reg_l2 12.893
loss 4.046027
cutoff 1.2475757e-6 network size 537
STEP 10 ================================
prereg loss 0.5671405 reg_l1 34.786583 reg_l2 12.893153
loss 4.045799
STEP 11 ================================
prereg loss 0.56500244 reg_l1 34.775692 reg_l2 12.892985
loss 4.042572
cutoff 1.3847457e-6 network size 536
STEP 12 ================================
prereg loss 0.5654326 reg_l1 34.764503 reg_l2 12.892738
loss 4.041883
STEP 13 ================================
prereg loss 0.56769174 reg_l1 34.753933 reg_l2 12.892814
loss 4.043085
cutoff 5.288239e-7 network size 535
STEP 14 ================================
prereg loss 0.5668101 reg_l1 34.744797 reg_l2 12.89333
loss 4.04129
STEP 15 ================================
prereg loss 0.56341547 reg_l1 34.737564 reg_l2 12.894099
loss 4.037172
cutoff 7.4648415e-7 network size 534
STEP 16 ================================
prereg loss 0.564892 reg_l1 34.729206 reg_l2 12.894692
loss 4.0378127
STEP 17 ================================
prereg loss 0.5637125 reg_l1 34.721443 reg_l2 12.894731
loss 4.0358567
cutoff 3.6513302e-7 network size 533
STEP 18 ================================
prereg loss 0.5614272 reg_l1 34.711655 reg_l2 12.894297
loss 4.032593
STEP 19 ================================
prereg loss 0.5636096 reg_l1 34.701412 reg_l2 12.8938055
loss 4.0337505
cutoff 1.2865348e-7 network size 532
STEP 20 ================================
prereg loss 0.5618051 reg_l1 34.69127 reg_l2 12.893671
loss 4.030932
STEP 21 ================================
prereg loss 0.5581441 reg_l1 34.683857 reg_l2 12.893992
loss 4.02653
cutoff 6.463739e-8 network size 531
STEP 22 ================================
prereg loss 0.5574139 reg_l1 34.67765 reg_l2 12.894565
loss 4.025179
STEP 23 ================================
prereg loss 0.55690664 reg_l1 34.66967 reg_l2 12.895228
loss 4.023874
cutoff 9.1613765e-7 network size 530
STEP 24 ================================
prereg loss 0.55662394 reg_l1 34.65909 reg_l2 12.895797
loss 4.0225325
STEP 25 ================================
prereg loss 0.56012994 reg_l1 34.64977 reg_l2 12.896161
loss 4.025107
cutoff 2.0275547e-7 network size 529
STEP 26 ================================
prereg loss 0.55829203 reg_l1 34.640816 reg_l2 12.896426
loss 4.0223737
STEP 27 ================================
prereg loss 0.5545404 reg_l1 34.6312 reg_l2 12.896519
loss 4.01766
cutoff 4.3940963e-7 network size 528
STEP 28 ================================
prereg loss 0.55364585 reg_l1 34.621746 reg_l2 12.896364
loss 4.0158205
STEP 29 ================================
prereg loss 0.55332416 reg_l1 34.61106 reg_l2 12.896009
loss 4.01443
cutoff 4.3888576e-8 network size 527
STEP 30 ================================
prereg loss 0.5526476 reg_l1 34.599663 reg_l2 12.89568
loss 4.0126143
STEP 31 ================================
prereg loss 0.5539394 reg_l1 34.588978 reg_l2 12.895704
loss 4.0128374
cutoff 5.399397e-7 network size 526
STEP 32 ================================
prereg loss 0.55385697 reg_l1 34.580944 reg_l2 12.896198
loss 4.0119514
STEP 33 ================================
prereg loss 0.55146843 reg_l1 34.57629 reg_l2 12.896912
loss 4.0090976
cutoff 6.808059e-7 network size 525
STEP 34 ================================
prereg loss 0.54956895 reg_l1 34.569237 reg_l2 12.897427
loss 4.0064926
STEP 35 ================================
prereg loss 0.5485905 reg_l1 34.559986 reg_l2 12.897534
loss 4.004589
cutoff 1.3475565e-6 network size 524
STEP 36 ================================
prereg loss 0.5484868 reg_l1 34.548084 reg_l2 12.89736
loss 4.0032954
STEP 37 ================================
prereg loss 0.54833627 reg_l1 34.539436 reg_l2 12.897212
loss 4.00228
cutoff 1.5338669e-6 network size 523
STEP 38 ================================
prereg loss 0.54653907 reg_l1 34.533802 reg_l2 12.89732
loss 3.9999194
STEP 39 ================================
prereg loss 0.54560953 reg_l1 34.525024 reg_l2 12.897559
loss 3.998112
cutoff 1.5744372e-6 network size 522
STEP 40 ================================
prereg loss 0.5442318 reg_l1 34.51436 reg_l2 12.8978
loss 3.9956675
STEP 41 ================================
prereg loss 0.5445897 reg_l1 34.505737 reg_l2 12.898013
loss 3.9951634
cutoff 4.6507193e-7 network size 521
STEP 42 ================================
prereg loss 0.54553705 reg_l1 34.49719 reg_l2 12.898235
loss 3.995256
STEP 43 ================================
prereg loss 0.5452257 reg_l1 34.4876 reg_l2 12.89849
loss 3.9939854
cutoff 6.135306e-7 network size 520
STEP 44 ================================
prereg loss 0.5434893 reg_l1 34.478508 reg_l2 12.8988
loss 3.9913402
STEP 45 ================================
prereg loss 0.54243565 reg_l1 34.470493 reg_l2 12.899157
loss 3.989485
cutoff 5.767142e-7 network size 519
STEP 46 ================================
prereg loss 0.5411046 reg_l1 34.463024 reg_l2 12.899514
loss 3.987407
STEP 47 ================================
prereg loss 0.54073524 reg_l1 34.455063 reg_l2 12.899634
loss 3.9862416
cutoff 1.3314057e-6 network size 518
STEP 48 ================================
prereg loss 0.54117286 reg_l1 34.444874 reg_l2 12.899494
loss 3.98566
STEP 49 ================================
prereg loss 0.5402946 reg_l1 34.437515 reg_l2 12.8993025
loss 3.9840462
cutoff 1.9805375e-6 network size 517
STEP 50 ================================
prereg loss 0.53827095 reg_l1 34.431614 reg_l2 12.899255
loss 3.9814324
STEP 51 ================================
prereg loss 0.53729355 reg_l1 34.423508 reg_l2 12.899432
loss 3.9796443
cutoff 2.5901245e-6 network size 516
STEP 52 ================================
prereg loss 0.5371735 reg_l1 34.41692 reg_l2 12.899711
loss 3.9788656
STEP 53 ================================
prereg loss 0.5374797 reg_l1 34.408382 reg_l2 12.899992
loss 3.978318
cutoff 7.953622e-8 network size 515
STEP 54 ================================
prereg loss 0.53631943 reg_l1 34.39771 reg_l2 12.900248
loss 3.9760904
STEP 55 ================================
prereg loss 0.53576654 reg_l1 34.387383 reg_l2 12.90041
loss 3.974505
cutoff 1.2624397e-6 network size 514
STEP 56 ================================
prereg loss 0.53513354 reg_l1 34.378223 reg_l2 12.900592
loss 3.972956
STEP 57 ================================
prereg loss 0.53450197 reg_l1 34.370785 reg_l2 12.900809
loss 3.9715805
cutoff 1.3647659e-6 network size 513
STEP 58 ================================
prereg loss 0.53416294 reg_l1 34.363194 reg_l2 12.900997
loss 3.9704823
STEP 59 ================================
prereg loss 0.5336935 reg_l1 34.352566 reg_l2 12.901101
loss 3.9689503
cutoff 8.386669e-7 network size 512
STEP 60 ================================
prereg loss 0.53297293 reg_l1 34.34281 reg_l2 12.901156
loss 3.9672542
STEP 61 ================================
prereg loss 0.5324579 reg_l1 34.334007 reg_l2 12.901329
loss 3.9658587
cutoff 8.204079e-7 network size 511
STEP 62 ================================
prereg loss 0.5318141 reg_l1 34.325924 reg_l2 12.901677
loss 3.9644065
STEP 63 ================================
prereg loss 0.53115755 reg_l1 34.319668 reg_l2 12.902031
loss 3.9631243
cutoff 2.423876e-7 network size 510
STEP 64 ================================
prereg loss 0.5305154 reg_l1 34.311565 reg_l2 12.902195
loss 3.961672
STEP 65 ================================
prereg loss 0.5297861 reg_l1 34.30305 reg_l2 12.902141
loss 3.9600914
cutoff 8.1179314e-7 network size 509
STEP 66 ================================
prereg loss 0.5292785 reg_l1 34.293884 reg_l2 12.901925
loss 3.958667
STEP 67 ================================
prereg loss 0.5291803 reg_l1 34.287037 reg_l2 12.901721
loss 3.957884
cutoff 2.415356e-6 network size 508
STEP 68 ================================
prereg loss 0.5291413 reg_l1 34.275455 reg_l2 12.901696
loss 3.956687
STEP 69 ================================
prereg loss 0.5285415 reg_l1 34.264397 reg_l2 12.901941
loss 3.9549813
cutoff 3.1660602e-7 network size 507
STEP 70 ================================
prereg loss 0.52796364 reg_l1 34.255665 reg_l2 12.902377
loss 3.95353
STEP 71 ================================
prereg loss 0.526782 reg_l1 34.247787 reg_l2 12.902843
loss 3.9515607
cutoff 2.6113048e-7 network size 506
STEP 72 ================================
prereg loss 0.5258304 reg_l1 34.23819 reg_l2 12.903119
loss 3.9496493
STEP 73 ================================
prereg loss 0.52505255 reg_l1 34.23131 reg_l2 12.903122
loss 3.9481838
cutoff 1.4224934e-6 network size 505
STEP 74 ================================
prereg loss 0.5244107 reg_l1 34.223614 reg_l2 12.902917
loss 3.946772
STEP 75 ================================
prereg loss 0.5245451 reg_l1 34.214325 reg_l2 12.902611
loss 3.9459777
cutoff 6.139253e-7 network size 504
STEP 76 ================================
prereg loss 0.5249226 reg_l1 34.20535 reg_l2 12.902425
loss 3.9454575
STEP 77 ================================
prereg loss 0.5247295 reg_l1 34.196465 reg_l2 12.902577
loss 3.944376
cutoff 1.2583769e-8 network size 503
STEP 78 ================================
prereg loss 0.5227237 reg_l1 34.188725 reg_l2 12.903111
loss 3.9415963
STEP 79 ================================
prereg loss 0.52186304 reg_l1 34.178123 reg_l2 12.903738
loss 3.9396753
cutoff 4.5205525e-7 network size 502
STEP 80 ================================
prereg loss 0.52185416 reg_l1 34.169525 reg_l2 12.904056
loss 3.9388068
STEP 81 ================================
prereg loss 0.52048844 reg_l1 34.16258 reg_l2 12.903862
loss 3.9367464
cutoff 4.3171895e-8 network size 501
STEP 82 ================================
prereg loss 0.5205605 reg_l1 34.155304 reg_l2 12.903332
loss 3.936091
STEP 83 ================================
prereg loss 0.5213915 reg_l1 34.14415 reg_l2 12.902821
loss 3.9358068
cutoff 8.0879545e-8 network size 500
STEP 84 ================================
prereg loss 0.52198863 reg_l1 34.13049 reg_l2 12.902643
loss 3.9350376
STEP 85 ================================
prereg loss 0.520299 reg_l1 34.124123 reg_l2 12.902928
loss 3.9327114
cutoff 4.8788206e-7 network size 499
STEP 86 ================================
prereg loss 0.5187254 reg_l1 34.121098 reg_l2 12.903568
loss 3.9308352
STEP 87 ================================
prereg loss 0.5185782 reg_l1 34.114098 reg_l2 12.904227
loss 3.929988
cutoff 2.1965898e-6 network size 498
STEP 88 ================================
prereg loss 0.5178922 reg_l1 34.104916 reg_l2 12.904478
loss 3.9283838
STEP 89 ================================
prereg loss 0.51619744 reg_l1 34.0957 reg_l2 12.904187
loss 3.9257674
cutoff 1.9376198e-6 network size 497
STEP 90 ================================
prereg loss 0.51756525 reg_l1 34.08864 reg_l2 12.903621
loss 3.926429
STEP 91 ================================
prereg loss 0.51784146 reg_l1 34.07903 reg_l2 12.903193
loss 3.9257445
cutoff 1.3955232e-6 network size 496
STEP 92 ================================
prereg loss 0.51712483 reg_l1 34.068283 reg_l2 12.903071
loss 3.9239533
STEP 93 ================================
prereg loss 0.5169001 reg_l1 34.059376 reg_l2 12.903172
loss 3.9228377
cutoff 9.43357e-7 network size 495
STEP 94 ================================
prereg loss 0.51612175 reg_l1 34.05194 reg_l2 12.903343
loss 3.9213157
STEP 95 ================================
prereg loss 0.51474893 reg_l1 34.044422 reg_l2 12.903431
loss 3.9191914
cutoff 2.0211828e-6 network size 494
STEP 96 ================================
prereg loss 0.5138191 reg_l1 34.0367 reg_l2 12.903399
loss 3.917489
STEP 97 ================================
prereg loss 0.5136276 reg_l1 34.02917 reg_l2 12.903399
loss 3.9165447
cutoff 8.8532033e-7 network size 493
STEP 98 ================================
prereg loss 0.51307034 reg_l1 34.02328 reg_l2 12.903598
loss 3.9153986
STEP 99 ================================
prereg loss 0.5126193 reg_l1 34.01569 reg_l2 12.903978
loss 3.9141884
cutoff 9.3907875e-7 network size 492
STEP 100 ================================
prereg loss 0.51213694 reg_l1 34.00767 reg_l2 12.904247
loss 3.912904
STEP 101 ================================
prereg loss 0.5112905 reg_l1 34.00084 reg_l2 12.904155
loss 3.9113746
cutoff 1.346245e-6 network size 491
STEP 102 ================================
prereg loss 0.51078254 reg_l1 33.993355 reg_l2 12.903776
loss 3.910118
STEP 103 ================================
prereg loss 0.51071405 reg_l1 33.983814 reg_l2 12.903244
loss 3.9090955
cutoff 1.4587495e-6 network size 490
STEP 104 ================================
prereg loss 0.51106626 reg_l1 33.973507 reg_l2 12.902855
loss 3.908417
STEP 105 ================================
prereg loss 0.51124215 reg_l1 33.96459 reg_l2 12.902803
loss 3.907701
cutoff 5.92845e-7 network size 489
STEP 106 ================================
prereg loss 0.5102144 reg_l1 33.957664 reg_l2 12.903022
loss 3.9059808
STEP 107 ================================
prereg loss 0.5087166 reg_l1 33.94886 reg_l2 12.903294
loss 3.9036026
cutoff 7.021853e-7 network size 488
STEP 108 ================================
prereg loss 0.50794125 reg_l1 33.941036 reg_l2 12.903385
loss 3.902045
STEP 109 ================================
prereg loss 0.5076417 reg_l1 33.93296 reg_l2 12.903198
loss 3.9009376
cutoff 1.6104605e-7 network size 487
STEP 110 ================================
prereg loss 0.50659627 reg_l1 33.926388 reg_l2 12.9028845
loss 3.8992352
STEP 111 ================================
prereg loss 0.50636184 reg_l1 33.91946 reg_l2 12.902515
loss 3.8983078
cutoff 9.4057395e-7 network size 486
STEP 112 ================================
prereg loss 0.5070803 reg_l1 33.909172 reg_l2 12.902217
loss 3.8979976
STEP 113 ================================
prereg loss 0.50799024 reg_l1 33.90067 reg_l2 12.902107
loss 3.898057
cutoff 1.0883523e-6 network size 485
STEP 114 ================================
prereg loss 0.50725955 reg_l1 33.891224 reg_l2 12.902316
loss 3.896382
STEP 115 ================================
prereg loss 0.5063283 reg_l1 33.884033 reg_l2 12.902799
loss 3.8947318
cutoff 4.886024e-7 network size 484
STEP 116 ================================
prereg loss 0.5048581 reg_l1 33.878334 reg_l2 12.903393
loss 3.8926914
STEP 117 ================================
prereg loss 0.5045228 reg_l1 33.873615 reg_l2 12.903705
loss 3.8918843
cutoff 2.0660082e-8 network size 483
STEP 118 ================================
prereg loss 0.50470483 reg_l1 33.86475 reg_l2 12.903492
loss 3.89118
STEP 119 ================================
prereg loss 0.5037788 reg_l1 33.85377 reg_l2 12.902879
loss 3.8891559
cutoff 1.1922002e-6 network size 482
STEP 120 ================================
prereg loss 0.50312084 reg_l1 33.845097 reg_l2 12.90212
loss 3.8876307
STEP 121 ================================
prereg loss 0.50412273 reg_l1 33.839844 reg_l2 12.901559
loss 3.888107
cutoff 4.4238254e-7 network size 481
STEP 122 ================================
prereg loss 0.50362945 reg_l1 33.833843 reg_l2 12.901401
loss 3.887014
STEP 123 ================================
prereg loss 0.50261813 reg_l1 33.82572 reg_l2 12.901565
loss 3.8851902
cutoff 2.4655724e-6 network size 480
STEP 124 ================================
prereg loss 0.50228107 reg_l1 33.81754 reg_l2 12.901884
loss 3.884035
STEP 125 ================================
prereg loss 0.50182414 reg_l1 33.812138 reg_l2 12.902198
loss 3.883038
cutoff 6.856171e-7 network size 479
STEP 126 ================================
prereg loss 0.5011761 reg_l1 33.805832 reg_l2 12.902332
loss 3.8817594
STEP 127 ================================
prereg loss 0.501092 reg_l1 33.79706 reg_l2 12.902258
loss 3.8807979
cutoff 1.323273e-6 network size 478
STEP 128 ================================
prereg loss 0.49950728 reg_l1 33.78615 reg_l2 12.902007
loss 3.8781223
STEP 129 ================================
prereg loss 0.5003968 reg_l1 33.778248 reg_l2 12.901766
loss 3.8782215
cutoff 9.5234464e-7 network size 477
STEP 130 ================================
prereg loss 0.50163615 reg_l1 33.771812 reg_l2 12.901741
loss 3.8788176
STEP 131 ================================
prereg loss 0.49950743 reg_l1 33.7619 reg_l2 12.901876
loss 3.8756976
cutoff 9.886335e-7 network size 476
STEP 132 ================================
prereg loss 0.49841428 reg_l1 33.756313 reg_l2 12.901953
loss 3.8740456
STEP 133 ================================
prereg loss 0.4976339 reg_l1 33.751255 reg_l2 12.901872
loss 3.8727596
cutoff 1.2107002e-6 network size 475
STEP 134 ================================
prereg loss 0.4970996 reg_l1 33.745247 reg_l2 12.90159
loss 3.8716245
STEP 135 ================================
prereg loss 0.497436 reg_l1 33.73671 reg_l2 12.901267
loss 3.871107
cutoff 5.8824116e-7 network size 474
STEP 136 ================================
prereg loss 0.4987481 reg_l1 33.729782 reg_l2 12.901118
loss 3.8717263
STEP 137 ================================
prereg loss 0.49683544 reg_l1 33.725536 reg_l2 12.901205
loss 3.869389
cutoff 5.6558565e-7 network size 473
STEP 138 ================================
prereg loss 0.49619564 reg_l1 33.722435 reg_l2 12.9012575
loss 3.8684392
STEP 139 ================================
prereg loss 0.49653596 reg_l1 33.715397 reg_l2 12.901006
loss 3.8680758
cutoff 1.430526e-6 network size 472
STEP 140 ================================
prereg loss 0.49670413 reg_l1 33.708267 reg_l2 12.900472
loss 3.8675308
STEP 141 ================================
prereg loss 0.49959812 reg_l1 33.70011 reg_l2 12.899949
loss 3.869609
cutoff 8.412644e-7 network size 471
STEP 142 ================================
prereg loss 0.5018375 reg_l1 33.69138 reg_l2 12.899782
loss 3.8709755
STEP 143 ================================
prereg loss 0.4963512 reg_l1 33.683037 reg_l2 12.90011
loss 3.864655
cutoff 1.0877557e-6 network size 470
STEP 144 ================================
prereg loss 0.49500227 reg_l1 33.679184 reg_l2 12.900757
loss 3.8629208
STEP 145 ================================
prereg loss 0.4968784 reg_l1 33.67609 reg_l2 12.901331
loss 3.8644874
cutoff 2.1960295e-6 network size 469
STEP 146 ================================
prereg loss 0.49408224 reg_l1 33.670948 reg_l2 12.90151
loss 3.861177
STEP 147 ================================
prereg loss 0.4924912 reg_l1 33.663727 reg_l2 12.901266
loss 3.858864
cutoff 5.883221e-7 network size 468
STEP 148 ================================
prereg loss 0.49748862 reg_l1 33.65398 reg_l2 12.900764
loss 3.862887
STEP 149 ================================
prereg loss 0.49740565 reg_l1 33.646763 reg_l2 12.900291
loss 3.862082
cutoff 1.4805755e-6 network size 467
STEP 150 ================================
prereg loss 0.4920775 reg_l1 33.63864 reg_l2 12.899942
loss 3.8559418
STEP 151 ================================
prereg loss 0.49140066 reg_l1 33.635986 reg_l2 12.899584
loss 3.8549993
cutoff 2.8994414e-6 network size 466
STEP 152 ================================
prereg loss 0.49163085 reg_l1 33.630627 reg_l2 12.89913
loss 3.8546934
STEP 153 ================================
prereg loss 0.49034044 reg_l1 33.620968 reg_l2 12.898734
loss 3.8524373
cutoff 5.0858216e-7 network size 465
STEP 154 ================================
prereg loss 0.4912667 reg_l1 33.61261 reg_l2 12.898582
loss 3.8525279
STEP 155 ================================
prereg loss 0.49111232 reg_l1 33.60637 reg_l2 12.898819
loss 3.8517492
cutoff 2.7471106e-7 network size 464
STEP 156 ================================
prereg loss 0.4885315 reg_l1 33.601128 reg_l2 12.899287
loss 3.8486445
STEP 157 ================================
prereg loss 0.4874137 reg_l1 33.59527 reg_l2 12.899595
loss 3.8469405
cutoff 9.425239e-7 network size 463
STEP 158 ================================
prereg loss 0.48804244 reg_l1 33.58888 reg_l2 12.89936
loss 3.8469303
STEP 159 ================================
prereg loss 0.4879218 reg_l1 33.58258 reg_l2 12.898499
loss 3.8461797
cutoff 5.481925e-7 network size 462
STEP 160 ================================
prereg loss 0.4901117 reg_l1 33.576397 reg_l2 12.89727
loss 3.8477516
STEP 161 ================================
prereg loss 0.4922104 reg_l1 33.570976 reg_l2 12.896291
loss 3.849308
cutoff 9.963969e-8 network size 461
STEP 162 ================================
prereg loss 0.48989427 reg_l1 33.56485 reg_l2 12.895987
loss 3.8463793
STEP 163 ================================
prereg loss 0.48701406 reg_l1 33.558857 reg_l2 12.8963375
loss 3.8428998
cutoff 1.531298e-7 network size 460
STEP 164 ================================
prereg loss 0.48590037 reg_l1 33.552776 reg_l2 12.896942
loss 3.8411782
STEP 165 ================================
prereg loss 0.48555824 reg_l1 33.547752 reg_l2 12.897359
loss 3.8403335
cutoff 1.0240938e-6 network size 459
STEP 166 ================================
prereg loss 0.48725283 reg_l1 33.54107 reg_l2 12.897328
loss 3.8413596
STEP 167 ================================
prereg loss 0.49003989 reg_l1 33.53175 reg_l2 12.896825
loss 3.8432148
cutoff 8.141342e-9 network size 458
STEP 168 ================================
prereg loss 0.48869693 reg_l1 33.522873 reg_l2 12.896185
loss 3.8409843
STEP 169 ================================
prereg loss 0.48607114 reg_l1 33.51868 reg_l2 12.895663
loss 3.8379393
cutoff 3.9974802e-6 network size 457
STEP 170 ================================
prereg loss 0.48587927 reg_l1 33.515163 reg_l2 12.895339
loss 3.8373957
STEP 171 ================================
prereg loss 0.4860857 reg_l1 33.5098 reg_l2 12.895203
loss 3.8370657
cutoff 1.3803656e-6 network size 456
STEP 172 ================================
prereg loss 0.4861042 reg_l1 33.501724 reg_l2 12.895193
loss 3.8362768
STEP 173 ================================
prereg loss 0.48612747 reg_l1 33.49376 reg_l2 12.895289
loss 3.8355033
cutoff 3.0750334e-7 network size 455
STEP 174 ================================
prereg loss 0.4859051 reg_l1 33.489468 reg_l2 12.8954735
loss 3.834852
STEP 175 ================================
prereg loss 0.4857185 reg_l1 33.48412 reg_l2 12.895741
loss 3.8341305
cutoff 8.0004247e-7 network size 454
STEP 176 ================================
prereg loss 0.48401487 reg_l1 33.480278 reg_l2 12.896026
loss 3.8320427
STEP 177 ================================
prereg loss 0.4819482 reg_l1 33.475433 reg_l2 12.896115
loss 3.8294916
cutoff 1.6956328e-6 network size 453
STEP 178 ================================
prereg loss 0.48290935 reg_l1 33.46782 reg_l2 12.895756
loss 3.8296914
STEP 179 ================================
prereg loss 0.48360172 reg_l1 33.459583 reg_l2 12.894908
loss 3.8295603
cutoff 1.4783036e-6 network size 452
STEP 180 ================================
prereg loss 0.48510134 reg_l1 33.451347 reg_l2 12.893955
loss 3.830236
STEP 181 ================================
prereg loss 0.4876376 reg_l1 33.44408 reg_l2 12.893354
loss 3.8320456
cutoff 3.9145707e-6 network size 451
STEP 182 ================================
prereg loss 0.48604527 reg_l1 33.43959 reg_l2 12.893371
loss 3.8300045
STEP 183 ================================
prereg loss 0.48205715 reg_l1 33.434303 reg_l2 12.893884
loss 3.8254874
cutoff 2.4288602e-6 network size 450
STEP 184 ================================
prereg loss 0.48214218 reg_l1 33.430023 reg_l2 12.894361
loss 3.8251445
STEP 185 ================================
prereg loss 0.48178536 reg_l1 33.423847 reg_l2 12.894277
loss 3.82417
cutoff 3.3536708e-8 network size 449
STEP 186 ================================
prereg loss 0.48112866 reg_l1 33.41618 reg_l2 12.893631
loss 3.8227468
STEP 187 ================================
prereg loss 0.48331785 reg_l1 33.408756 reg_l2 12.892818
loss 3.8241935
cutoff 7.3113006e-7 network size 448
STEP 188 ================================
prereg loss 0.48329175 reg_l1 33.401424 reg_l2 12.892361
loss 3.8234344
STEP 189 ================================
prereg loss 0.47975776 reg_l1 33.396084 reg_l2 12.892495
loss 3.8193662
cutoff 9.584364e-7 network size 447
STEP 190 ================================
prereg loss 0.47852364 reg_l1 33.39228 reg_l2 12.892949
loss 3.817752
STEP 191 ================================
prereg loss 0.47826102 reg_l1 33.38778 reg_l2 12.893265
loss 3.817039
cutoff 1.257863e-6 network size 446
STEP 192 ================================
prereg loss 0.4791767 reg_l1 33.38088 reg_l2 12.893133
loss 3.8172646
STEP 193 ================================
prereg loss 0.48053575 reg_l1 33.371967 reg_l2 12.892586
loss 3.8177326
cutoff 3.2146454e-6 network size 445
STEP 194 ================================
prereg loss 0.47896087 reg_l1 33.366673 reg_l2 12.891885
loss 3.815628
STEP 195 ================================
prereg loss 0.4783055 reg_l1 33.360725 reg_l2 12.891291
loss 3.8143783
cutoff 2.0046718e-6 network size 444
STEP 196 ================================
prereg loss 0.47802836 reg_l1 33.353546 reg_l2 12.891062
loss 3.8133829
STEP 197 ================================
prereg loss 0.4774616 reg_l1 33.34602 reg_l2 12.891165
loss 3.8120637
cutoff 1.0585081e-6 network size 443
STEP 198 ================================
prereg loss 0.47724557 reg_l1 33.340584 reg_l2 12.891398
loss 3.811304
STEP 199 ================================
prereg loss 0.47694597 reg_l1 33.3335 reg_l2 12.891573
loss 3.8102958
cutoff 7.764829e-7 network size 442
STEP 200 ================================
prereg loss 0.47648475 reg_l1 33.326374 reg_l2 12.891574
loss 3.8091223
2022-08-14T16:14:06.612

julia> serialize("cf-s2-442-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-442-parameters-opt.ser", opt)
```

Once more:

```
julia> # once more

julia> interleaving_steps!(200)
2022-08-14T16:17:16.208
STEP 1 ================================
prereg loss 0.47633934 reg_l1 33.322086 reg_l2 12.89135
loss 3.808548
cutoff 1.2872463e-6 network size 441
STEP 2 ================================
prereg loss 0.47584313 reg_l1 33.318153 reg_l2 12.89103
loss 3.8076587
STEP 3 ================================
prereg loss 0.4756856 reg_l1 33.3115 reg_l2 12.890741
loss 3.8068357
cutoff 4.5117486e-7 network size 440
STEP 4 ================================
prereg loss 0.47600368 reg_l1 33.304943 reg_l2 12.890527
loss 3.806498
STEP 5 ================================
prereg loss 0.4755759 reg_l1 33.299034 reg_l2 12.890262
loss 3.8054793
cutoff 1.3830904e-6 network size 439
STEP 6 ================================
prereg loss 0.47470802 reg_l1 33.29313 reg_l2 12.889872
loss 3.8040211
STEP 7 ================================
prereg loss 0.47543693 reg_l1 33.28826 reg_l2 12.889503
loss 3.804263
cutoff 2.748864e-6 network size 438
STEP 8 ================================
prereg loss 0.47607866 reg_l1 33.280144 reg_l2 12.889346
loss 3.8040931
STEP 9 ================================
prereg loss 0.47564307 reg_l1 33.27371 reg_l2 12.889438
loss 3.803014
cutoff 2.2180684e-6 network size 437
STEP 10 ================================
prereg loss 0.47470278 reg_l1 33.26891 reg_l2 12.889561
loss 3.8015938
STEP 11 ================================
prereg loss 0.47446644 reg_l1 33.264824 reg_l2 12.88955
loss 3.800949
cutoff 2.2119207e-7 network size 436
STEP 12 ================================
prereg loss 0.47395766 reg_l1 33.259754 reg_l2 12.889322
loss 3.799933
STEP 13 ================================
prereg loss 0.47308588 reg_l1 33.253998 reg_l2 12.888877
loss 3.7984858
cutoff 2.057528e-6 network size 435
STEP 14 ================================
prereg loss 0.47311172 reg_l1 33.247406 reg_l2 12.8883705
loss 3.7978523
STEP 15 ================================
prereg loss 0.47646356 reg_l1 33.240074 reg_l2 12.888024
loss 3.800471
cutoff 1.123718e-6 network size 434
STEP 16 ================================
prereg loss 0.47486135 reg_l1 33.233948 reg_l2 12.888042
loss 3.7982562
STEP 17 ================================
prereg loss 0.47200763 reg_l1 33.230064 reg_l2 12.888188
loss 3.795014
cutoff 3.342444e-6 network size 433
STEP 18 ================================
prereg loss 0.471605 reg_l1 33.224766 reg_l2 12.888219
loss 3.7940817
STEP 19 ================================
prereg loss 0.4715132 reg_l1 33.21578 reg_l2 12.888065
loss 3.793091
cutoff 9.995647e-7 network size 432
STEP 20 ================================
prereg loss 0.47111812 reg_l1 33.208454 reg_l2 12.887721
loss 3.7919636
STEP 21 ================================
prereg loss 0.47145674 reg_l1 33.202797 reg_l2 12.887293
loss 3.7917366
cutoff 2.88328e-7 network size 431
STEP 22 ================================
prereg loss 0.47244886 reg_l1 33.196487 reg_l2 12.886996
loss 3.7920976
STEP 23 ================================
prereg loss 0.4717649 reg_l1 33.19072 reg_l2 12.887006
loss 3.7908368
cutoff 2.070301e-7 network size 430
STEP 24 ================================
prereg loss 0.47065818 reg_l1 33.18514 reg_l2 12.887272
loss 3.7891722
STEP 25 ================================
prereg loss 0.47020936 reg_l1 33.179935 reg_l2 12.887589
loss 3.788203
cutoff 6.537084e-7 network size 429
STEP 26 ================================
prereg loss 0.47029603 reg_l1 33.173943 reg_l2 12.887703
loss 3.7876902
STEP 27 ================================
prereg loss 0.47041532 reg_l1 33.16752 reg_l2 12.887448
loss 3.7871673
cutoff 2.0381726e-6 network size 428
STEP 28 ================================
prereg loss 0.47010162 reg_l1 33.160767 reg_l2 12.886856
loss 3.7861784
STEP 29 ================================
prereg loss 0.47010309 reg_l1 33.153683 reg_l2 12.886153
loss 3.7854714
cutoff 9.805444e-7 network size 427
STEP 30 ================================
prereg loss 0.47092018 reg_l1 33.1474 reg_l2 12.885669
loss 3.78566
STEP 31 ================================
prereg loss 0.47092056 reg_l1 33.140686 reg_l2 12.885658
loss 3.784989
cutoff 9.475625e-7 network size 426
STEP 32 ================================
prereg loss 0.46990967 reg_l1 33.135414 reg_l2 12.886136
loss 3.783451
STEP 33 ================================
prereg loss 0.46879098 reg_l1 33.132305 reg_l2 12.886838
loss 3.7820215
cutoff 2.6737216e-6 network size 425
STEP 34 ================================
prereg loss 0.46823794 reg_l1 33.128635 reg_l2 12.887322
loss 3.7811015
STEP 35 ================================
prereg loss 0.46789128 reg_l1 33.12282 reg_l2 12.887298
loss 3.7801735
cutoff 3.3491233e-8 network size 424
STEP 36 ================================
prereg loss 0.46778694 reg_l1 33.11547 reg_l2 12.886704
loss 3.779334
STEP 37 ================================
prereg loss 0.46947217 reg_l1 33.107647 reg_l2 12.885864
loss 3.780237
cutoff 1.3803281e-6 network size 423
STEP 38 ================================
prereg loss 0.47008017 reg_l1 33.10108 reg_l2 12.885225
loss 3.7801883
STEP 39 ================================
prereg loss 0.46859843 reg_l1 33.093643 reg_l2 12.88505
loss 3.7779627
cutoff 5.906186e-7 network size 422
STEP 40 ================================
prereg loss 0.46773362 reg_l1 33.08782 reg_l2 12.885338
loss 3.7765155
STEP 41 ================================
prereg loss 0.46760044 reg_l1 33.084946 reg_l2 12.885786
loss 3.776095
cutoff 2.369081e-6 network size 421
STEP 42 ================================
prereg loss 0.46681187 reg_l1 33.079784 reg_l2 12.886061
loss 3.7747903
STEP 43 ================================
prereg loss 0.46958154 reg_l1 33.071106 reg_l2 12.885999
loss 3.7766922
cutoff 5.14181e-7 network size 420
STEP 44 ================================
prereg loss 0.47077596 reg_l1 33.06351 reg_l2 12.885746
loss 3.7771273
STEP 45 ================================
prereg loss 0.4668761 reg_l1 33.05756 reg_l2 12.88553
loss 3.7726321
cutoff 1.2830933e-6 network size 419
STEP 46 ================================
prereg loss 0.4655393 reg_l1 33.052574 reg_l2 12.885392
loss 3.7707968
STEP 47 ================================
prereg loss 0.46587354 reg_l1 33.04537 reg_l2 12.885278
loss 3.7704103
cutoff 6.451446e-7 network size 418
STEP 48 ================================
prereg loss 0.46559024 reg_l1 33.03861 reg_l2 12.885166
loss 3.7694511
STEP 49 ================================
prereg loss 0.46693984 reg_l1 33.033844 reg_l2 12.885181
loss 3.7703245
cutoff 3.2474054e-7 network size 417
STEP 50 ================================
prereg loss 0.46635038 reg_l1 33.02794 reg_l2 12.885372
loss 3.7691443
STEP 51 ================================
prereg loss 0.46490556 reg_l1 33.02204 reg_l2 12.885561
loss 3.7671096
cutoff 2.38535e-7 network size 416
STEP 52 ================================
prereg loss 0.46446192 reg_l1 33.015812 reg_l2 12.885606
loss 3.7660432
STEP 53 ================================
prereg loss 0.46423617 reg_l1 33.010036 reg_l2 12.885503
loss 3.76524
cutoff 1.499211e-8 network size 415
STEP 54 ================================
prereg loss 0.46428794 reg_l1 33.00224 reg_l2 12.885352
loss 3.764512
STEP 55 ================================
prereg loss 0.465127 reg_l1 32.99405 reg_l2 12.885181
loss 3.7645319
cutoff 1.96146e-6 network size 414
STEP 56 ================================
prereg loss 0.46585774 reg_l1 32.987015 reg_l2 12.8850565
loss 3.7645593
STEP 57 ================================
prereg loss 0.46507815 reg_l1 32.98229 reg_l2 12.8851385
loss 3.763307
cutoff 1.0324584e-8 network size 413
STEP 58 ================================
prereg loss 0.4634791 reg_l1 32.97662 reg_l2 12.885464
loss 3.761141
STEP 59 ================================
prereg loss 0.46337745 reg_l1 32.971764 reg_l2 12.885822
loss 3.7605538
cutoff 1.2260598e-6 network size 412
STEP 60 ================================
prereg loss 0.46319175 reg_l1 32.96584 reg_l2 12.88588
loss 3.7597756
STEP 61 ================================
prereg loss 0.46416798 reg_l1 32.96035 reg_l2 12.8856325
loss 3.7602031
cutoff 1.3380486e-8 network size 411
STEP 62 ================================
prereg loss 0.46499288 reg_l1 32.953712 reg_l2 12.885295
loss 3.760364
STEP 63 ================================
prereg loss 0.46296522 reg_l1 32.947643 reg_l2 12.885148
loss 3.7577295
cutoff 1.1849552e-6 network size 410
STEP 64 ================================
prereg loss 0.46214285 reg_l1 32.941883 reg_l2 12.885166
loss 3.7563312
STEP 65 ================================
prereg loss 0.46220338 reg_l1 32.936756 reg_l2 12.885135
loss 3.755879
cutoff 7.086783e-7 network size 409
STEP 66 ================================
prereg loss 0.46185952 reg_l1 32.928364 reg_l2 12.884976
loss 3.754696
STEP 67 ================================
prereg loss 0.46344855 reg_l1 32.922897 reg_l2 12.884813
loss 3.7557383
cutoff 4.8444053e-7 network size 408
STEP 68 ================================
prereg loss 0.4642811 reg_l1 32.91663 reg_l2 12.884882
loss 3.755944
STEP 69 ================================
prereg loss 0.46133754 reg_l1 32.911564 reg_l2 12.885221
loss 3.752494
cutoff 6.319967e-7 network size 407
STEP 70 ================================
prereg loss 0.46050033 reg_l1 32.907352 reg_l2 12.885589
loss 3.7512355
STEP 71 ================================
prereg loss 0.46053347 reg_l1 32.90183 reg_l2 12.885677
loss 3.7507162
cutoff 5.3252734e-7 network size 406
STEP 72 ================================
prereg loss 0.46031997 reg_l1 32.895782 reg_l2 12.88542
loss 3.7498982
STEP 73 ================================
prereg loss 0.46216223 reg_l1 32.8894 reg_l2 12.885024
loss 3.7511024
cutoff 1.428496e-6 network size 405
STEP 74 ================================
prereg loss 0.46174017 reg_l1 32.882156 reg_l2 12.884814
loss 3.749956
STEP 75 ================================
prereg loss 0.46004868 reg_l1 32.87687 reg_l2 12.884933
loss 3.7477357
cutoff 1.9655272e-6 network size 404
STEP 76 ================================
prereg loss 0.4593227 reg_l1 32.870663 reg_l2 12.885248
loss 3.746389
STEP 77 ================================
prereg loss 0.45944178 reg_l1 32.8654 reg_l2 12.8855
loss 3.7459817
cutoff 1.8030623e-6 network size 403
STEP 78 ================================
prereg loss 0.459383 reg_l1 32.85863 reg_l2 12.885556
loss 3.7452462
STEP 79 ================================
prereg loss 0.45932043 reg_l1 32.85024 reg_l2 12.885566
loss 3.7443442
cutoff 1.17634045e-7 network size 402
STEP 80 ================================
prereg loss 0.45849764 reg_l1 32.842983 reg_l2 12.885684
loss 3.742796
STEP 81 ================================
prereg loss 0.4583801 reg_l1 32.83806 reg_l2 12.885849
loss 3.742186
cutoff 4.833855e-7 network size 401
STEP 82 ================================
prereg loss 0.45862383 reg_l1 32.83129 reg_l2 12.886045
loss 3.741753
STEP 83 ================================
prereg loss 0.4590902 reg_l1 32.825356 reg_l2 12.8863
loss 3.7416258
cutoff 7.2062903e-7 network size 400
STEP 84 ================================
prereg loss 0.4576767 reg_l1 32.8194 reg_l2 12.886681
loss 3.7396169
STEP 85 ================================
prereg loss 0.45719078 reg_l1 32.815006 reg_l2 12.886988
loss 3.7386913
cutoff 9.413165e-6 network size 399
STEP 86 ================================
prereg loss 0.45716915 reg_l1 32.81122 reg_l2 12.887147
loss 3.738291
STEP 87 ================================
prereg loss 0.4566424 reg_l1 32.804466 reg_l2 12.887074
loss 3.7370892
cutoff 1.1211523e-6 network size 398
STEP 88 ================================
prereg loss 0.4569547 reg_l1 32.796726 reg_l2 12.886927
loss 3.7366273
STEP 89 ================================
prereg loss 0.45653632 reg_l1 32.78952 reg_l2 12.887031
loss 3.7354884
cutoff 1.3709205e-6 network size 397
STEP 90 ================================
prereg loss 0.4556089 reg_l1 32.784935 reg_l2 12.88741
loss 3.7341025
STEP 91 ================================
prereg loss 0.45515826 reg_l1 32.781303 reg_l2 12.887923
loss 3.7332885
cutoff 5.098831e-6 network size 396
STEP 92 ================================
prereg loss 0.45483592 reg_l1 32.777176 reg_l2 12.888352
loss 3.7325535
STEP 93 ================================
prereg loss 0.4543761 reg_l1 32.76991 reg_l2 12.8885975
loss 3.731367
cutoff 1.413573e-6 network size 395
STEP 94 ================================
prereg loss 0.45440966 reg_l1 32.76224 reg_l2 12.888823
loss 3.7306337
STEP 95 ================================
prereg loss 0.45425752 reg_l1 32.75759 reg_l2 12.88919
loss 3.7300167
cutoff 5.4363045e-7 network size 394
STEP 96 ================================
prereg loss 0.4539136 reg_l1 32.75169 reg_l2 12.889645
loss 3.7290828
STEP 97 ================================
prereg loss 0.45342684 reg_l1 32.746994 reg_l2 12.890096
loss 3.7281263
cutoff 4.3341734e-6 network size 393
STEP 98 ================================
prereg loss 0.45267335 reg_l1 32.7402 reg_l2 12.890433
loss 3.7266934
STEP 99 ================================
prereg loss 0.4523454 reg_l1 32.733322 reg_l2 12.890684
loss 3.7256777
cutoff 2.555309e-6 network size 392
STEP 100 ================================
prereg loss 0.4525281 reg_l1 32.72855 reg_l2 12.89099
loss 3.725383
STEP 101 ================================
prereg loss 0.45219946 reg_l1 32.72373 reg_l2 12.891489
loss 3.7245724
cutoff 2.8911745e-6 network size 391
STEP 102 ================================
prereg loss 0.45167238 reg_l1 32.718876 reg_l2 12.892092
loss 3.7235599
STEP 103 ================================
prereg loss 0.4509613 reg_l1 32.713886 reg_l2 12.89263
loss 3.7223501
cutoff 2.1313845e-6 network size 390
STEP 104 ================================
prereg loss 0.45052877 reg_l1 32.70686 reg_l2 12.892999
loss 3.7212148
STEP 105 ================================
prereg loss 0.45039892 reg_l1 32.703262 reg_l2 12.893326
loss 3.7207253
cutoff 4.5411252e-6 network size 389
STEP 106 ================================
prereg loss 0.4499592 reg_l1 32.699715 reg_l2 12.893864
loss 3.719931
STEP 107 ================================
prereg loss 0.44969177 reg_l1 32.69352 reg_l2 12.894583
loss 3.7190437
cutoff 3.0834053e-7 network size 388
STEP 108 ================================
prereg loss 0.44909203 reg_l1 32.687088 reg_l2 12.895418
loss 3.717801
STEP 109 ================================
prereg loss 0.44845876 reg_l1 32.682816 reg_l2 12.896177
loss 3.7167404
cutoff 1.2846176e-6 network size 387
STEP 110 ================================
prereg loss 0.44810283 reg_l1 32.677902 reg_l2 12.89672
loss 3.7158933
STEP 111 ================================
prereg loss 0.44856724 reg_l1 32.671272 reg_l2 12.897034
loss 3.7156944
cutoff 1.550492e-6 network size 386
STEP 112 ================================
prereg loss 0.44807702 reg_l1 32.66468 reg_l2 12.897289
loss 3.714545
STEP 113 ================================
prereg loss 0.44728386 reg_l1 32.660755 reg_l2 12.897699
loss 3.7133594
cutoff 7.2496186e-7 network size 385
STEP 114 ================================
prereg loss 0.44700652 reg_l1 32.65581 reg_l2 12.898327
loss 3.7125876
STEP 115 ================================
prereg loss 0.44655347 reg_l1 32.65045 reg_l2 12.899182
loss 3.7115986
cutoff 3.611989e-6 network size 384
STEP 116 ================================
prereg loss 0.44650716 reg_l1 32.64467 reg_l2 12.900211
loss 3.7109742
STEP 117 ================================
prereg loss 0.44650376 reg_l1 32.63965 reg_l2 12.90126
loss 3.7104688
cutoff 3.3953038e-6 network size 383
STEP 118 ================================
prereg loss 0.44605505 reg_l1 32.636196 reg_l2 12.902158
loss 3.7096748
STEP 119 ================================
prereg loss 0.44528463 reg_l1 32.63238 reg_l2 12.902815
loss 3.7085228
cutoff 9.2860137e-7 network size 382
STEP 120 ================================
prereg loss 0.44480392 reg_l1 32.627846 reg_l2 12.903213
loss 3.7075887
STEP 121 ================================
prereg loss 0.44437551 reg_l1 32.62365 reg_l2 12.903403
loss 3.7067406
cutoff 2.7212263e-6 network size 381
STEP 122 ================================
prereg loss 0.44425997 reg_l1 32.6175 reg_l2 12.903578
loss 3.7060099
STEP 123 ================================
prereg loss 0.44433695 reg_l1 32.61181 reg_l2 12.9039345
loss 3.7055178
cutoff 4.9569935e-6 network size 380
STEP 124 ================================
prereg loss 0.4439372 reg_l1 32.60511 reg_l2 12.90448
loss 3.7044485
STEP 125 ================================
prereg loss 0.44305423 reg_l1 32.600456 reg_l2 12.905247
loss 3.7031
cutoff 5.264519e-7 network size 379
STEP 126 ================================
prereg loss 0.4426319 reg_l1 32.596256 reg_l2 12.906105
loss 3.7022576
STEP 127 ================================
prereg loss 0.44246358 reg_l1 32.59088 reg_l2 12.906847
loss 3.701552
cutoff 1.8935243e-6 network size 378
STEP 128 ================================
prereg loss 0.44194627 reg_l1 32.585033 reg_l2 12.907438
loss 3.7004497
STEP 129 ================================
prereg loss 0.44138348 reg_l1 32.58052 reg_l2 12.907878
loss 3.6994357
cutoff 3.4725963e-7 network size 377
STEP 130 ================================
prereg loss 0.44119355 reg_l1 32.57563 reg_l2 12.908307
loss 3.6987567
STEP 131 ================================
prereg loss 0.44081417 reg_l1 32.57148 reg_l2 12.908826
loss 3.6979623
cutoff 1.1527081e-6 network size 376
STEP 132 ================================
prereg loss 0.4402084 reg_l1 32.5661 reg_l2 12.909488
loss 3.6968186
STEP 133 ================================
prereg loss 0.4396043 reg_l1 32.563286 reg_l2 12.91024
loss 3.6959329
cutoff 3.0785318e-6 network size 375
STEP 134 ================================
prereg loss 0.43957743 reg_l1 32.558826 reg_l2 12.910972
loss 3.69546
STEP 135 ================================
prereg loss 0.43945345 reg_l1 32.55391 reg_l2 12.911583
loss 3.6948442
cutoff 1.4817651e-6 network size 374
STEP 136 ================================
prereg loss 0.43911645 reg_l1 32.548016 reg_l2 12.911996
loss 3.693918
STEP 137 ================================
prereg loss 0.43868053 reg_l1 32.54262 reg_l2 12.912243
loss 3.6929426
cutoff 5.648482e-6 network size 373
STEP 138 ================================
prereg loss 0.43861377 reg_l1 32.53775 reg_l2 12.912481
loss 3.692389
STEP 139 ================================
prereg loss 0.43853313 reg_l1 32.532852 reg_l2 12.9129095
loss 3.6918182
cutoff 2.845634e-6 network size 372
STEP 140 ================================
prereg loss 0.43803933 reg_l1 32.528675 reg_l2 12.913632
loss 3.6909068
STEP 141 ================================
prereg loss 0.43732765 reg_l1 32.52562 reg_l2 12.914599
loss 3.6898897
cutoff 4.037167e-6 network size 371
STEP 142 ================================
prereg loss 0.4367248 reg_l1 32.52268 reg_l2 12.915653
loss 3.6889927
STEP 143 ================================
prereg loss 0.43638968 reg_l1 32.517807 reg_l2 12.916562
loss 3.6881704
cutoff 1.165572e-6 network size 370
STEP 144 ================================
prereg loss 0.43629757 reg_l1 32.511757 reg_l2 12.917234
loss 3.6874733
STEP 145 ================================
prereg loss 0.4362756 reg_l1 32.507023 reg_l2 12.917672
loss 3.6869779
cutoff 5.1503594e-7 network size 369
STEP 146 ================================
prereg loss 0.43618104 reg_l1 32.502132 reg_l2 12.917987
loss 3.6863945
STEP 147 ================================
prereg loss 0.43582323 reg_l1 32.497437 reg_l2 12.918298
loss 3.685567
cutoff 2.248079e-6 network size 368
STEP 148 ================================
prereg loss 0.4353641 reg_l1 32.491222 reg_l2 12.918785
loss 3.6844864
STEP 149 ================================
prereg loss 0.43497726 reg_l1 32.48581 reg_l2 12.919535
loss 3.6835582
cutoff 7.2005423e-7 network size 367
STEP 150 ================================
prereg loss 0.43484235 reg_l1 32.481342 reg_l2 12.920417
loss 3.6829767
STEP 151 ================================
prereg loss 0.43484926 reg_l1 32.477085 reg_l2 12.921242
loss 3.6825578
cutoff 9.7890734e-8 network size 366
STEP 152 ================================
prereg loss 0.43428332 reg_l1 32.472076 reg_l2 12.92193
loss 3.681491
STEP 153 ================================
prereg loss 0.43385646 reg_l1 32.466976 reg_l2 12.922402
loss 3.6805542
cutoff 1.9410072e-6 network size 365
STEP 154 ================================
prereg loss 0.4335935 reg_l1 32.46105 reg_l2 12.922807
loss 3.6796985
STEP 155 ================================
prereg loss 0.43360472 reg_l1 32.45624 reg_l2 12.923246
loss 3.679229
cutoff 2.723158e-6 network size 364
STEP 156 ================================
prereg loss 0.43312487 reg_l1 32.45211 reg_l2 12.923787
loss 3.678336
STEP 157 ================================
prereg loss 0.432661 reg_l1 32.4471 reg_l2 12.924346
loss 3.6773713
cutoff 2.5842003e-6 network size 363
STEP 158 ================================
prereg loss 0.43212646 reg_l1 32.44275 reg_l2 12.92487
loss 3.6764014
STEP 159 ================================
prereg loss 0.4318148 reg_l1 32.436794 reg_l2 12.925406
loss 3.6754942
cutoff 2.8170507e-7 network size 362
STEP 160 ================================
prereg loss 0.4316284 reg_l1 32.43055 reg_l2 12.926089
loss 3.6746836
STEP 161 ================================
prereg loss 0.4312488 reg_l1 32.427467 reg_l2 12.926983
loss 3.6739957
cutoff 2.205406e-6 network size 361
STEP 162 ================================
prereg loss 0.43065935 reg_l1 32.424026 reg_l2 12.92798
loss 3.673062
STEP 163 ================================
prereg loss 0.43016297 reg_l1 32.419224 reg_l2 12.928889
loss 3.6720853
cutoff 3.3429715e-6 network size 360
STEP 164 ================================
prereg loss 0.43021074 reg_l1 32.41374 reg_l2 12.929521
loss 3.6715846
STEP 165 ================================
prereg loss 0.43021762 reg_l1 32.40919 reg_l2 12.929786
loss 3.6711369
cutoff 1.0851545e-7 network size 359
STEP 166 ================================
prereg loss 0.42953733 reg_l1 32.4045 reg_l2 12.929874
loss 3.6699872
STEP 167 ================================
prereg loss 0.42945716 reg_l1 32.39811 reg_l2 12.930021
loss 3.6692681
cutoff 1.2701628e-6 network size 358
STEP 168 ================================
prereg loss 0.42941624 reg_l1 32.391247 reg_l2 12.930484
loss 3.668541
STEP 169 ================================
prereg loss 0.428921 reg_l1 32.386887 reg_l2 12.931331
loss 3.6676097
cutoff 6.8010413e-6 network size 357
STEP 170 ================================
prereg loss 0.42828628 reg_l1 32.383358 reg_l2 12.932403
loss 3.6666222
STEP 171 ================================
prereg loss 0.42798442 reg_l1 32.377884 reg_l2 12.933411
loss 3.665773
cutoff 9.2956543e-7 network size 356
STEP 172 ================================
prereg loss 0.42779827 reg_l1 32.370907 reg_l2 12.934097
loss 3.664889
STEP 173 ================================
prereg loss 0.4276672 reg_l1 32.365337 reg_l2 12.934371
loss 3.664201
cutoff 3.4191107e-6 network size 355
STEP 174 ================================
prereg loss 0.42709538 reg_l1 32.36056 reg_l2 12.934404
loss 3.6631515
STEP 175 ================================
prereg loss 0.4270352 reg_l1 32.356075 reg_l2 12.934412
loss 3.662643
cutoff 1.2216624e-6 network size 354
STEP 176 ================================
prereg loss 0.42705274 reg_l1 32.3499 reg_l2 12.934673
loss 3.6620426
STEP 177 ================================
prereg loss 0.4267861 reg_l1 32.34529 reg_l2 12.935314
loss 3.6613154
cutoff 5.3393633e-6 network size 353
STEP 178 ================================
prereg loss 0.4263089 reg_l1 32.339905 reg_l2 12.936216
loss 3.6602993
STEP 179 ================================
prereg loss 0.42604554 reg_l1 32.334778 reg_l2 12.937126
loss 3.6595235
cutoff 1.7052107e-6 network size 352
STEP 180 ================================
prereg loss 0.4255196 reg_l1 32.33015 reg_l2 12.937876
loss 3.6585345
STEP 181 ================================
prereg loss 0.42537075 reg_l1 32.325947 reg_l2 12.938304
loss 3.6579654
cutoff 2.0919688e-6 network size 351
STEP 182 ================================
prereg loss 0.42527518 reg_l1 32.32047 reg_l2 12.938493
loss 3.657322
STEP 183 ================================
prereg loss 0.42486906 reg_l1 32.314644 reg_l2 12.938631
loss 3.6563334
cutoff 8.516603e-6 network size 350
STEP 184 ================================
prereg loss 0.42465857 reg_l1 32.308285 reg_l2 12.938879
loss 3.655487
STEP 185 ================================
prereg loss 0.42445096 reg_l1 32.301525 reg_l2 12.939335
loss 3.6546035
cutoff 4.490852e-6 network size 349
STEP 186 ================================
prereg loss 0.42399198 reg_l1 32.296417 reg_l2 12.939982
loss 3.6536336
STEP 187 ================================
prereg loss 0.42357963 reg_l1 32.291546 reg_l2 12.940723
loss 3.6527343
cutoff 6.2312465e-6 network size 348
STEP 188 ================================
prereg loss 0.4233835 reg_l1 32.285507 reg_l2 12.941397
loss 3.6519341
STEP 189 ================================
prereg loss 0.42383996 reg_l1 32.279766 reg_l2 12.941933
loss 3.6518166
cutoff 2.7113128e-7 network size 347
STEP 190 ================================
prereg loss 0.42322633 reg_l1 32.275024 reg_l2 12.942347
loss 3.650729
STEP 191 ================================
prereg loss 0.42268398 reg_l1 32.26992 reg_l2 12.942674
loss 3.649676
cutoff 6.3200932e-6 network size 346
STEP 192 ================================
prereg loss 0.42227742 reg_l1 32.26493 reg_l2 12.943009
loss 3.6487706
STEP 193 ================================
prereg loss 0.42218298 reg_l1 32.26026 reg_l2 12.943446
loss 3.6482093
cutoff 3.2301396e-6 network size 345
STEP 194 ================================
prereg loss 0.42217505 reg_l1 32.2547 reg_l2 12.943988
loss 3.647645
STEP 195 ================================
prereg loss 0.4217703 reg_l1 32.24908 reg_l2 12.944585
loss 3.6466784
cutoff 5.91619e-7 network size 344
STEP 196 ================================
prereg loss 0.4213035 reg_l1 32.24299 reg_l2 12.945178
loss 3.6456025
STEP 197 ================================
prereg loss 0.42111617 reg_l1 32.23711 reg_l2 12.945683
loss 3.6448271
cutoff 6.507671e-7 network size 343
STEP 198 ================================
prereg loss 0.4213852 reg_l1 32.232384 reg_l2 12.946088
loss 3.6446238
STEP 199 ================================
prereg loss 0.42105356 reg_l1 32.226364 reg_l2 12.946408
loss 3.64369
cutoff 1.1372313e-6 network size 342
STEP 200 ================================
prereg loss 0.42068616 reg_l1 32.220795 reg_l2 12.946659
loss 3.6427658
2022-08-14T17:01:48.552

julia> serialize("cf-s2-342-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-342-parameters-opt.ser", opt)
```

I decided to do that once more; it turned out that that was too aggressive, but I gut lucky:

```
julia> # once more

julia> interleaving_steps!(200)
2022-08-14T17:06:23.553
STEP 1 ================================
prereg loss 0.42040235 reg_l1 32.215794 reg_l2 12.946935
loss 3.6419816
cutoff 2.5608097e-6 network size 341
STEP 2 ================================
prereg loss 0.42029914 reg_l1 32.210457 reg_l2 12.947301
loss 3.6413448
STEP 3 ================================
prereg loss 0.420124 reg_l1 32.20461 reg_l2 12.947785
loss 3.640585
cutoff 5.814094e-6 network size 340
STEP 4 ================================
prereg loss 0.41976815 reg_l1 32.197384 reg_l2 12.948332
loss 3.6395066
STEP 5 ================================
prereg loss 0.41953197 reg_l1 32.19186 reg_l2 12.948862
loss 3.6387181
cutoff 1.5291153e-7 network size 339
STEP 6 ================================
prereg loss 0.41980004 reg_l1 32.1862 reg_l2 12.949341
loss 3.63842
STEP 7 ================================
prereg loss 0.41923752 reg_l1 32.180656 reg_l2 12.949781
loss 3.6373034
cutoff 6.5071654e-6 network size 338
STEP 8 ================================
prereg loss 0.41879985 reg_l1 32.1735 reg_l2 12.950204
loss 3.63615
STEP 9 ================================
prereg loss 0.41854256 reg_l1 32.167233 reg_l2 12.950581
loss 3.6352658
cutoff 5.5970304e-7 network size 337
STEP 10 ================================
prereg loss 0.41849616 reg_l1 32.162827 reg_l2 12.950985
loss 3.6347787
STEP 11 ================================
prereg loss 0.4183912 reg_l1 32.158154 reg_l2 12.951454
loss 3.6342065
cutoff 3.029174e-6 network size 336
STEP 12 ================================
prereg loss 0.41804713 reg_l1 32.152805 reg_l2 12.95199
loss 3.6333277
STEP 13 ================================
prereg loss 0.41772047 reg_l1 32.146904 reg_l2 12.952484
loss 3.632411
cutoff 1.3139219e-5 network size 335
STEP 14 ================================
prereg loss 0.41748396 reg_l1 32.14058 reg_l2 12.952896
loss 3.631542
STEP 15 ================================
prereg loss 0.4174772 reg_l1 32.13485 reg_l2 12.953203
loss 3.6309621
cutoff 8.0666746e-7 network size 334
STEP 16 ================================
prereg loss 0.4172053 reg_l1 32.12825 reg_l2 12.9534855
loss 3.6300304
STEP 17 ================================
prereg loss 0.41709796 reg_l1 32.12366 reg_l2 12.953711
loss 3.6294641
cutoff 1.2368078e-5 network size 333
STEP 18 ================================
prereg loss 0.4169249 reg_l1 32.1177 reg_l2 12.953942
loss 3.6286948
STEP 19 ================================
prereg loss 0.4166718 reg_l1 32.11089 reg_l2 12.954198
loss 3.6277606
cutoff 6.706163e-6 network size 332
STEP 20 ================================
prereg loss 0.41631514 reg_l1 32.104156 reg_l2 12.954526
loss 3.6267307
STEP 21 ================================
prereg loss 0.41584814 reg_l1 32.0985 reg_l2 12.954983
loss 3.625698
cutoff 8.584961e-6 network size 331
STEP 22 ================================
prereg loss 0.4155954 reg_l1 32.093067 reg_l2 12.955531
loss 3.6249022
STEP 23 ================================
prereg loss 0.4155869 reg_l1 32.086765 reg_l2 12.956109
loss 3.6242635
cutoff 6.350795e-6 network size 330
STEP 24 ================================
prereg loss 0.4158368 reg_l1 32.08086 reg_l2 12.956612
loss 3.6239228
STEP 25 ================================
prereg loss 0.41516706 reg_l1 32.075542 reg_l2 12.957013
loss 3.6227214
cutoff 8.325107e-6 network size 329
STEP 26 ================================
prereg loss 0.41477442 reg_l1 32.069668 reg_l2 12.957245
loss 3.6217413
STEP 27 ================================
prereg loss 0.41450027 reg_l1 32.063004 reg_l2 12.957362
loss 3.6208007
cutoff 3.6430356e-7 network size 328
STEP 28 ================================
prereg loss 0.41442382 reg_l1 32.056465 reg_l2 12.95748
loss 3.6200705
STEP 29 ================================
prereg loss 0.41460305 reg_l1 32.049885 reg_l2 12.957664
loss 3.6195915
cutoff 5.0437593e-6 network size 327
STEP 30 ================================
prereg loss 0.41435203 reg_l1 32.04415 reg_l2 12.957971
loss 3.618767
STEP 31 ================================
prereg loss 0.4140298 reg_l1 32.039165 reg_l2 12.9584055
loss 3.6179464
cutoff 1.6081904e-6 network size 326
STEP 32 ================================
prereg loss 0.4137942 reg_l1 32.03278 reg_l2 12.958858
loss 3.6170723
STEP 33 ================================
prereg loss 0.4136141 reg_l1 32.02649 reg_l2 12.959253
loss 3.616263
cutoff 3.276762e-6 network size 325
STEP 34 ================================
prereg loss 0.4135658 reg_l1 32.020008 reg_l2 12.959559
loss 3.6155667
STEP 35 ================================
prereg loss 0.41353977 reg_l1 32.014736 reg_l2 12.959749
loss 3.6150136
cutoff 1.5569087e-5 network size 324
STEP 36 ================================
prereg loss 0.41302872 reg_l1 32.00926 reg_l2 12.959935
loss 3.6139545
STEP 37 ================================
prereg loss 0.41263998 reg_l1 32.003345 reg_l2 12.9601965
loss 3.6129746
cutoff 1.7759439e-6 network size 323
STEP 38 ================================
prereg loss 0.41239506 reg_l1 31.99676 reg_l2 12.960555
loss 3.612071
STEP 39 ================================
prereg loss 0.41235554 reg_l1 31.989647 reg_l2 12.960986
loss 3.6113205
cutoff 1.2143597e-5 network size 322
STEP 40 ================================
prereg loss 0.41252342 reg_l1 31.985052 reg_l2 12.961379
loss 3.6110287
STEP 41 ================================
prereg loss 0.4123176 reg_l1 31.98041 reg_l2 12.961708
loss 3.6103585
cutoff 4.9401624e-6 network size 321
STEP 42 ================================
prereg loss 0.41197145 reg_l1 31.974869 reg_l2 12.96195
loss 3.6094584
STEP 43 ================================
prereg loss 0.411516 reg_l1 31.968637 reg_l2 12.962114
loss 3.6083798
cutoff 6.1345672e-6 network size 320
STEP 44 ================================
prereg loss 0.4112943 reg_l1 31.962105 reg_l2 12.962256
loss 3.6075048
STEP 45 ================================
prereg loss 0.41146904 reg_l1 31.956543 reg_l2 12.96244
loss 3.6071234
cutoff 1.7373019e-5 network size 319
STEP 46 ================================
prereg loss 0.41141903 reg_l1 31.951792 reg_l2 12.962744
loss 3.6065984
STEP 47 ================================
prereg loss 0.4108544 reg_l1 31.945341 reg_l2 12.963163
loss 3.6053884
cutoff 1.2130884e-5 network size 318
STEP 48 ================================
prereg loss 0.41063216 reg_l1 31.93836 reg_l2 12.96357
loss 3.604468
STEP 49 ================================
prereg loss 0.4106007 reg_l1 31.933012 reg_l2 12.963926
loss 3.6039019
cutoff 1.1089942e-6 network size 317
STEP 50 ================================
prereg loss 0.41070214 reg_l1 31.928265 reg_l2 12.964192
loss 3.6035287
STEP 51 ================================
prereg loss 0.41077542 reg_l1 31.922907 reg_l2 12.964372
loss 3.6030662
cutoff 1.3356097e-5 network size 316
STEP 52 ================================
prereg loss 0.41017258 reg_l1 31.916815 reg_l2 12.964536
loss 3.6018543
STEP 53 ================================
prereg loss 0.4097426 reg_l1 31.911722 reg_l2 12.964724
loss 3.600915
cutoff 1.3101657e-5 network size 315
STEP 54 ================================
prereg loss 0.4095426 reg_l1 31.907776 reg_l2 12.964955
loss 3.60032
STEP 55 ================================
prereg loss 0.40936446 reg_l1 31.903372 reg_l2 12.965258
loss 3.5997016
cutoff 1.3890036e-5 network size 314
STEP 56 ================================
prereg loss 0.40900612 reg_l1 31.898554 reg_l2 12.965622
loss 3.5988615
STEP 57 ================================
prereg loss 0.40887845 reg_l1 31.894823 reg_l2 12.965928
loss 3.598361
cutoff 6.9905946e-6 network size 313
STEP 58 ================================
prereg loss 0.40865612 reg_l1 31.891018 reg_l2 12.966183
loss 3.597758
STEP 59 ================================
prereg loss 0.40851656 reg_l1 31.886713 reg_l2 12.96632
loss 3.597188
cutoff 1.5713325e-5 network size 312
STEP 60 ================================
prereg loss 0.40838537 reg_l1 31.881432 reg_l2 12.966369
loss 3.5965285
STEP 61 ================================
prereg loss 0.40822795 reg_l1 31.876013 reg_l2 12.966391
loss 3.5958292
cutoff 8.908952e-6 network size 311
STEP 62 ================================
prereg loss 0.40769392 reg_l1 31.871017 reg_l2 12.966483
loss 3.5947957
STEP 63 ================================
prereg loss 0.40739688 reg_l1 31.865685 reg_l2 12.966626
loss 3.5939653
cutoff 1.1733817e-5 network size 310
STEP 64 ================================
prereg loss 0.4077118 reg_l1 31.859968 reg_l2 12.966757
loss 3.5937085
STEP 65 ================================
prereg loss 0.4078071 reg_l1 31.85607 reg_l2 12.966887
loss 3.593414
cutoff 1.510384e-5 network size 309
STEP 66 ================================
prereg loss 0.4073841 reg_l1 31.853338 reg_l2 12.966959
loss 3.5927181
STEP 67 ================================
prereg loss 0.40673438 reg_l1 31.849133 reg_l2 12.966956
loss 3.5916479
cutoff 1.3664377e-5 network size 308
STEP 68 ================================
prereg loss 0.4064373 reg_l1 31.844458 reg_l2 12.966875
loss 3.5908833
STEP 69 ================================
prereg loss 0.4063928 reg_l1 31.83988 reg_l2 12.966814
loss 3.590381
cutoff 9.988951e-6 network size 307
STEP 70 ================================
prereg loss 0.40642604 reg_l1 31.836485 reg_l2 12.966797
loss 3.5900745
STEP 71 ================================
prereg loss 0.40608844 reg_l1 31.831999 reg_l2 12.966798
loss 3.5892882
cutoff 1.148817e-5 network size 306
STEP 72 ================================
prereg loss 0.40565193 reg_l1 31.826374 reg_l2 12.966771
loss 3.5882893
STEP 73 ================================
prereg loss 0.4054663 reg_l1 31.822914 reg_l2 12.966661
loss 3.5877578
cutoff 1.1424934e-5 network size 305
STEP 74 ================================
prereg loss 0.40558705 reg_l1 31.818518 reg_l2 12.96645
loss 3.5874388
STEP 75 ================================
prereg loss 0.40520978 reg_l1 31.812826 reg_l2 12.966167
loss 3.5864925
cutoff 8.21655e-6 network size 304
STEP 76 ================================
prereg loss 0.40472016 reg_l1 31.807264 reg_l2 12.965851
loss 3.5854466
STEP 77 ================================
prereg loss 0.40462366 reg_l1 31.803055 reg_l2 12.965515
loss 3.5849292
cutoff 9.755837e-6 network size 303
STEP 78 ================================
prereg loss 0.40434253 reg_l1 31.798544 reg_l2 12.965243
loss 3.584197
STEP 79 ================================
prereg loss 0.404073 reg_l1 31.793226 reg_l2 12.965036
loss 3.5833957
cutoff 8.914711e-6 network size 302
STEP 80 ================================
prereg loss 0.4038651 reg_l1 31.788708 reg_l2 12.964831
loss 3.582736
STEP 81 ================================
prereg loss 0.40367708 reg_l1 31.784311 reg_l2 12.96458
loss 3.5821083
cutoff 1.8842547e-7 network size 301
STEP 82 ================================
prereg loss 0.4035704 reg_l1 31.779646 reg_l2 12.964231
loss 3.581535
STEP 83 ================================
prereg loss 0.4030087 reg_l1 31.775091 reg_l2 12.963821
loss 3.5805178
cutoff 1.409006e-6 network size 300
STEP 84 ================================
prereg loss 0.4027461 reg_l1 31.769344 reg_l2 12.963344
loss 3.5796807
STEP 85 ================================
prereg loss 0.40263322 reg_l1 31.764307 reg_l2 12.962854
loss 3.579064
cutoff 2.1629776e-6 network size 299
STEP 86 ================================
prereg loss 0.40253758 reg_l1 31.75938 reg_l2 12.9624405
loss 3.5784757
STEP 87 ================================
prereg loss 0.40215477 reg_l1 31.75463 reg_l2 12.962137
loss 3.5776176
cutoff 5.7663347e-6 network size 298
STEP 88 ================================
prereg loss 0.40186194 reg_l1 31.749647 reg_l2 12.961887
loss 3.5768266
STEP 89 ================================
prereg loss 0.4017437 reg_l1 31.745432 reg_l2 12.96162
loss 3.5762868
cutoff 2.435316e-5 network size 297
STEP 90 ================================
prereg loss 0.40146035 reg_l1 31.740847 reg_l2 12.96131
loss 3.575545
STEP 91 ================================
prereg loss 0.40126282 reg_l1 31.735922 reg_l2 12.960918
loss 3.574855
cutoff 3.284622e-5 network size 296
STEP 92 ================================
prereg loss 0.4010716 reg_l1 31.730152 reg_l2 12.960437
loss 3.574087
STEP 93 ================================
prereg loss 0.4012592 reg_l1 31.724052 reg_l2 12.959883
loss 3.5736644
cutoff 3.0761294e-6 network size 295
STEP 94 ================================
prereg loss 0.4006689 reg_l1 31.718811 reg_l2 12.959357
loss 3.57255
STEP 95 ================================
prereg loss 0.40023458 reg_l1 31.714405 reg_l2 12.958892
loss 3.5716753
cutoff 1.9430649e-5 network size 294
STEP 96 ================================
prereg loss 0.40000916 reg_l1 31.709137 reg_l2 12.958445
loss 3.5709229
STEP 97 ================================
prereg loss 0.40016606 reg_l1 31.70404 reg_l2 12.958008
loss 3.5705702
cutoff 7.022263e-7 network size 293
STEP 98 ================================
prereg loss 0.3999884 reg_l1 31.698166 reg_l2 12.957585
loss 3.5698051
STEP 99 ================================
prereg loss 0.39948115 reg_l1 31.69346 reg_l2 12.957192
loss 3.5688272
cutoff 4.737005e-6 network size 292
STEP 100 ================================
prereg loss 0.3991373 reg_l1 31.688831 reg_l2 12.956825
loss 3.5680203
STEP 101 ================================
prereg loss 0.39892203 reg_l1 31.6846 reg_l2 12.956447
loss 3.567382
cutoff 1.6502207e-5 network size 291
STEP 102 ================================
prereg loss 0.39883977 reg_l1 31.67984 reg_l2 12.956059
loss 3.5668237
STEP 103 ================================
prereg loss 0.39879736 reg_l1 31.674154 reg_l2 12.955605
loss 3.5662127
cutoff 6.3004227e-6 network size 290
STEP 104 ================================
prereg loss 0.39855993 reg_l1 31.668377 reg_l2 12.9551
loss 3.5653977
STEP 105 ================================
prereg loss 0.3981822 reg_l1 31.663723 reg_l2 12.954537
loss 3.5645545
cutoff 9.560383e-6 network size 289
STEP 106 ================================
prereg loss 0.39787102 reg_l1 31.658587 reg_l2 12.953928
loss 3.5637298
STEP 107 ================================
prereg loss 0.39767718 reg_l1 31.653542 reg_l2 12.953326
loss 3.5630314
cutoff 2.596328e-5 network size 288
STEP 108 ================================
prereg loss 0.397586 reg_l1 31.647888 reg_l2 12.952734
loss 3.562375
STEP 109 ================================
prereg loss 0.3975585 reg_l1 31.642189 reg_l2 12.952169
loss 3.5617774
cutoff 2.6893395e-5 network size 287
STEP 110 ================================
prereg loss 0.3972129 reg_l1 31.63724 reg_l2 12.951652
loss 3.560937
STEP 111 ================================
prereg loss 0.3969479 reg_l1 31.63235 reg_l2 12.951134
loss 3.560183
cutoff 1.1097451e-5 network size 286
STEP 112 ================================
prereg loss 0.39676905 reg_l1 31.627533 reg_l2 12.950633
loss 3.5595224
STEP 113 ================================
prereg loss 0.39663315 reg_l1 31.62346 reg_l2 12.950152
loss 3.5589793
cutoff 4.3408523e-5 network size 285
STEP 114 ================================
prereg loss 0.39647081 reg_l1 31.619097 reg_l2 12.949671
loss 3.5583806
STEP 115 ================================
prereg loss 0.39620152 reg_l1 31.614056 reg_l2 12.949163
loss 3.5576072
cutoff 3.035217e-5 network size 284
STEP 116 ================================
prereg loss 0.39588222 reg_l1 31.608541 reg_l2 12.948618
loss 3.5567362
STEP 117 ================================
prereg loss 0.3955943 reg_l1 31.603085 reg_l2 12.948026
loss 3.555903
cutoff 1.6985949e-5 network size 283
STEP 118 ================================
prereg loss 0.395242 reg_l1 31.598265 reg_l2 12.947446
loss 3.5550685
STEP 119 ================================
prereg loss 0.39500546 reg_l1 31.593481 reg_l2 12.946867
loss 3.5543537
cutoff 1.3469018e-5 network size 282
STEP 120 ================================
prereg loss 0.39522272 reg_l1 31.588438 reg_l2 12.94627
loss 3.5540664
STEP 121 ================================
prereg loss 0.3949225 reg_l1 31.584656 reg_l2 12.945702
loss 3.553388
cutoff 4.8855793e-5 network size 281
STEP 122 ================================
prereg loss 0.39420125 reg_l1 31.580355 reg_l2 12.945127
loss 3.5522368
STEP 123 ================================
prereg loss 0.3939031 reg_l1 31.57552 reg_l2 12.944513
loss 3.551455
cutoff 1.7013444e-6 network size 280
STEP 124 ================================
prereg loss 0.39374834 reg_l1 31.570303 reg_l2 12.943864
loss 3.5507786
STEP 125 ================================
prereg loss 0.39367792 reg_l1 31.565548 reg_l2 12.943193
loss 3.550233
cutoff 4.6731457e-5 network size 279
STEP 126 ================================
prereg loss 0.3935276 reg_l1 31.561123 reg_l2 12.942507
loss 3.54964
STEP 127 ================================
prereg loss 0.39293486 reg_l1 31.556765 reg_l2 12.941832
loss 3.5486114
cutoff 3.0668176e-5 network size 278
STEP 128 ================================
prereg loss 0.3925924 reg_l1 31.551472 reg_l2 12.941155
loss 3.5477397
STEP 129 ================================
prereg loss 0.3924556 reg_l1 31.546083 reg_l2 12.940464
loss 3.547064
cutoff 3.119839e-5 network size 277
STEP 130 ================================
prereg loss 0.39275092 reg_l1 31.54221 reg_l2 12.939763
loss 3.546972
STEP 131 ================================
prereg loss 0.39238083 reg_l1 31.537947 reg_l2 12.939077
loss 3.5461755
cutoff 5.003657e-5 network size 276
STEP 132 ================================
prereg loss 0.39163956 reg_l1 31.533106 reg_l2 12.9384165
loss 3.54495
STEP 133 ================================
prereg loss 0.3912862 reg_l1 31.52843 reg_l2 12.937742
loss 3.5441294
cutoff 6.439285e-5 network size 275
STEP 134 ================================
prereg loss 0.39104664 reg_l1 31.5236 reg_l2 12.937067
loss 3.5434065
STEP 135 ================================
prereg loss 0.39095706 reg_l1 31.518188 reg_l2 12.936394
loss 3.542776
cutoff 3.8848084e-5 network size 274
STEP 136 ================================
prereg loss 0.39089754 reg_l1 31.51339 reg_l2 12.935701
loss 3.5422366
STEP 137 ================================
prereg loss 0.390772 reg_l1 31.508629 reg_l2 12.934978
loss 3.541635
cutoff 1.2698249e-5 network size 273
STEP 138 ================================
prereg loss 0.39021635 reg_l1 31.503767 reg_l2 12.934272
loss 3.5405931
STEP 139 ================================
prereg loss 0.38987947 reg_l1 31.498703 reg_l2 12.933541
loss 3.5397499
cutoff 2.8420716e-5 network size 272
STEP 140 ================================
prereg loss 0.38966238 reg_l1 31.493603 reg_l2 12.932786
loss 3.539023
STEP 141 ================================
prereg loss 0.38951033 reg_l1 31.488607 reg_l2 12.932014
loss 3.538371
cutoff 7.216751e-5 network size 271
STEP 142 ================================
prereg loss 0.38952053 reg_l1 31.484043 reg_l2 12.931227
loss 3.5379248
STEP 143 ================================
prereg loss 0.3892251 reg_l1 31.478758 reg_l2 12.930477
loss 3.5371008
cutoff 6.4241496e-5 network size 270
STEP 144 ================================
prereg loss 0.38878512 reg_l1 31.4736 reg_l2 12.929762
loss 3.5361452
STEP 145 ================================
prereg loss 0.38855514 reg_l1 31.468807 reg_l2 12.929054
loss 3.535436
cutoff 7.004819e-5 network size 269
STEP 146 ================================
prereg loss 0.38843864 reg_l1 31.463947 reg_l2 12.92839
loss 3.5348334
STEP 147 ================================
prereg loss 0.38845924 reg_l1 31.458485 reg_l2 12.927731
loss 3.5343077
cutoff 4.701018e-5 network size 268
STEP 148 ================================
prereg loss 0.38793772 reg_l1 31.453295 reg_l2 12.927114
loss 3.5332673
STEP 149 ================================
prereg loss 0.38756448 reg_l1 31.44823 reg_l2 12.926483
loss 3.5323875
cutoff 7.76659e-5 network size 267
STEP 150 ================================
prereg loss 0.3873487 reg_l1 31.44345 reg_l2 12.925782
loss 3.5316937
STEP 151 ================================
prereg loss 0.3872513 reg_l1 31.438303 reg_l2 12.925019
loss 3.5310817
cutoff 0.00010334357 network size 266
STEP 152 ================================
prereg loss 0.3873822 reg_l1 31.432848 reg_l2 12.924197
loss 3.530667
STEP 153 ================================
prereg loss 0.38684136 reg_l1 31.427183 reg_l2 12.923409
loss 3.5295596
cutoff 7.800324e-5 network size 265
STEP 154 ================================
prereg loss 0.38648936 reg_l1 31.421831 reg_l2 12.922634
loss 3.5286725
STEP 155 ================================
prereg loss 0.38624042 reg_l1 31.416475 reg_l2 12.921904
loss 3.527888
cutoff 8.940726e-5 network size 264
STEP 156 ================================
prereg loss 0.38608137 reg_l1 31.41094 reg_l2 12.921193
loss 3.5271754
STEP 157 ================================
prereg loss 0.38577557 reg_l1 31.405663 reg_l2 12.920552
loss 3.526342
cutoff 8.6355874e-5 network size 263
STEP 158 ================================
prereg loss 0.38555372 reg_l1 31.400331 reg_l2 12.919875
loss 3.525587
STEP 159 ================================
prereg loss 0.38549367 reg_l1 31.394636 reg_l2 12.919154
loss 3.5249574
cutoff 8.7014116e-5 network size 262
STEP 160 ================================
prereg loss 0.38506725 reg_l1 31.389797 reg_l2 12.918423
loss 3.524047
STEP 161 ================================
prereg loss 0.3847933 reg_l1 31.38472 reg_l2 12.917677
loss 3.5232654
cutoff 1.8374376e-5 network size 261
STEP 162 ================================
prereg loss 0.38462877 reg_l1 31.379267 reg_l2 12.916915
loss 3.5225556
STEP 163 ================================
prereg loss 0.38459086 reg_l1 31.373793 reg_l2 12.916152
loss 3.5219703
cutoff 9.1375274e-5 network size 260
STEP 164 ================================
prereg loss 0.38417506 reg_l1 31.368265 reg_l2 12.915425
loss 3.5210016
STEP 165 ================================
prereg loss 0.38391334 reg_l1 31.36276 reg_l2 12.914689
loss 3.5201893
cutoff 0.0001168363 network size 259
STEP 166 ================================
prereg loss 0.3837439 reg_l1 31.357649 reg_l2 12.913933
loss 3.5195088
STEP 167 ================================
prereg loss 0.3837693 reg_l1 31.352413 reg_l2 12.913147
loss 3.5190105
cutoff 0.00019937176 network size 258
STEP 168 ================================
prereg loss 0.38341033 reg_l1 31.346762 reg_l2 12.9123745
loss 3.5180864
STEP 169 ================================
prereg loss 0.3831101 reg_l1 31.341345 reg_l2 12.911622
loss 3.5172446
cutoff 0.00012476604 network size 257
STEP 170 ================================
prereg loss 0.3830502 reg_l1 31.335669 reg_l2 12.91088
loss 3.516617
STEP 171 ================================
prereg loss 0.38330215 reg_l1 31.330128 reg_l2 12.910124
loss 3.516315
cutoff 0.00060221925 network size 256
STEP 172 ================================
prereg loss 0.38287807 reg_l1 31.324224 reg_l2 12.909363
loss 3.5153005
STEP 173 ================================
prereg loss 0.38244233 reg_l1 31.319048 reg_l2 12.908605
loss 3.514347
cutoff 0.0008635807 network size 255
STEP 174 ================================
prereg loss 0.38230082 reg_l1 31.312946 reg_l2 12.907863
loss 3.5135956
STEP 175 ================================
prereg loss 0.3822236 reg_l1 31.307692 reg_l2 12.907152
loss 3.5129929
cutoff 0.00077183964 network size 254
STEP 176 ================================
prereg loss 0.3823142 reg_l1 31.301569 reg_l2 12.906429
loss 3.5124712
STEP 177 ================================
prereg loss 0.38233978 reg_l1 31.296316 reg_l2 12.905676
loss 3.5119715
cutoff 0.000934268 network size 253
STEP 178 ================================
prereg loss 0.38178182 reg_l1 31.290356 reg_l2 12.90489
loss 3.5108175
STEP 179 ================================
prereg loss 0.38147762 reg_l1 31.285301 reg_l2 12.904072
loss 3.5100079
cutoff 0.0013686108 network size 252
STEP 180 ================================
prereg loss 0.3813491 reg_l1 31.278702 reg_l2 12.90324
loss 3.5092194
STEP 181 ================================
prereg loss 0.38141385 reg_l1 31.273256 reg_l2 12.902414
loss 3.5087397
cutoff 0.001396518 network size 251
STEP 182 ================================
prereg loss 0.38163516 reg_l1 31.266375 reg_l2 12.901596
loss 3.5082726
STEP 183 ================================
prereg loss 0.3811363 reg_l1 31.261122 reg_l2 12.900836
loss 3.5072484
cutoff 0.0015101596 network size 250
STEP 184 ================================
prereg loss 0.38078535 reg_l1 31.254454 reg_l2 12.900096
loss 3.5062308
STEP 185 ================================
prereg loss 0.38059667 reg_l1 31.249336 reg_l2 12.899364
loss 3.5055304
cutoff 0.0017781398 network size 249
STEP 186 ================================
prereg loss 0.38055575 reg_l1 31.242332 reg_l2 12.8986
loss 3.5047889
STEP 187 ================================
prereg loss 0.38086775 reg_l1 31.23698 reg_l2 12.897812
loss 3.5045657
cutoff 0.0019629253 network size 248
STEP 188 ================================
prereg loss 0.38042268 reg_l1 31.229841 reg_l2 12.897047
loss 3.5034068
STEP 189 ================================
prereg loss 0.37993187 reg_l1 31.224787 reg_l2 12.8963175
loss 3.5024107
cutoff 0.002589985 network size 247
STEP 190 ================================
prereg loss 0.3807586 reg_l1 31.21703 reg_l2 12.895579
loss 3.5024614
STEP 191 ================================
prereg loss 0.3804114 reg_l1 31.211199 reg_l2 12.89458
loss 3.5015314
cutoff 0.002667012 network size 246
STEP 192 ================================
prereg loss 0.38242403 reg_l1 31.202662 reg_l2 12.893446
loss 3.5026903
STEP 193 ================================
prereg loss 0.38224372 reg_l1 31.197058 reg_l2 12.892237
loss 3.5019495
cutoff 0.002725531 network size 245
STEP 194 ================================
prereg loss 0.38339198 reg_l1 31.189339 reg_l2 12.891213
loss 3.502326
STEP 195 ================================
prereg loss 0.38147995 reg_l1 31.18492 reg_l2 12.8907
loss 3.4999719
cutoff 0.0030133254 network size 244
STEP 196 ================================
prereg loss 0.38047686 reg_l1 31.177227 reg_l2 12.89059
loss 3.4981997
STEP 197 ================================
prereg loss 0.38011554 reg_l1 31.172035 reg_l2 12.890614
loss 3.497319
cutoff 0.0026265273 network size 243
STEP 198 ================================
prereg loss 0.38118827 reg_l1 31.163963 reg_l2 12.890365
loss 3.4975848
STEP 199 ================================
prereg loss 0.38074932 reg_l1 31.158894 reg_l2 12.889641
loss 3.4966385
cutoff 0.0031116838 network size 242
STEP 200 ================================
prereg loss 0.3808388 reg_l1 31.150715 reg_l2 12.888363
loss 3.4959104
2022-08-14T17:42:29.603

julia> serialize("cf-s2-242-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-242-parameters-opt.ser", opt)
```

I have been lucky. In a better instrumented setup we would start increasing
steps:sparsifications ratio after 100 steps.

OK, let's do 50 steps without sparsification, and then do more fine-grained runs
while gradually increasing steps:sparsifications ratio.

**The intention is to backtrack on serious degradations.**

```
julia> steps!(50)
2022-08-14T17:45:31.752
STEP 1 ================================
prereg loss 0.38001207 reg_l1 31.145226 reg_l2 12.886416
loss 3.4945347
STEP 2 ================================
prereg loss 0.37918085 reg_l1 31.13933 reg_l2 12.884296
loss 3.493114
STEP 3 ================================
prereg loss 0.37948677 reg_l1 31.13347 reg_l2 12.882505
loss 3.4928339
STEP 4 ================================
prereg loss 0.38074198 reg_l1 31.128153 reg_l2 12.881312
loss 3.4935575
STEP 5 ================================
prereg loss 0.38018608 reg_l1 31.123692 reg_l2 12.880718
loss 3.4925554
STEP 6 ================================
prereg loss 0.37878302 reg_l1 31.119585 reg_l2 12.880434
loss 3.4907415
STEP 7 ================================
prereg loss 0.37851748 reg_l1 31.114958 reg_l2 12.880086
loss 3.4900131
STEP 8 ================================
prereg loss 0.3781055 reg_l1 31.109465 reg_l2 12.879384
loss 3.4890518
STEP 9 ================================
prereg loss 0.37815586 reg_l1 31.10329 reg_l2 12.878259
loss 3.488485
STEP 10 ================================
prereg loss 0.3791153 reg_l1 31.096937 reg_l2 12.876833
loss 3.488809
STEP 11 ================================
prereg loss 0.37854967 reg_l1 31.091059 reg_l2 12.875411
loss 3.4876554
STEP 12 ================================
prereg loss 0.37793642 reg_l1 31.085602 reg_l2 12.874188
loss 3.4864967
STEP 13 ================================
prereg loss 0.37759385 reg_l1 31.080503 reg_l2 12.873322
loss 3.485644
STEP 14 ================================
prereg loss 0.37728667 reg_l1 31.07575 reg_l2 12.8727455
loss 3.4848619
STEP 15 ================================
prereg loss 0.37706256 reg_l1 31.071108 reg_l2 12.87226
loss 3.4841733
STEP 16 ================================
prereg loss 0.37702498 reg_l1 31.06642 reg_l2 12.871632
loss 3.483667
STEP 17 ================================
prereg loss 0.37713453 reg_l1 31.061419 reg_l2 12.870665
loss 3.4832764
STEP 18 ================================
prereg loss 0.3766587 reg_l1 31.056168 reg_l2 12.869435
loss 3.4822755
STEP 19 ================================
prereg loss 0.3763605 reg_l1 31.050411 reg_l2 12.868055
loss 3.4814017
STEP 20 ================================
prereg loss 0.37621272 reg_l1 31.04435 reg_l2 12.866708
loss 3.4806476
STEP 21 ================================
prereg loss 0.3763246 reg_l1 31.038351 reg_l2 12.86556
loss 3.4801598
STEP 22 ================================
prereg loss 0.3762855 reg_l1 31.032753 reg_l2 12.864629
loss 3.4795609
STEP 23 ================================
prereg loss 0.3759062 reg_l1 31.027544 reg_l2 12.863857
loss 3.4786606
STEP 24 ================================
prereg loss 0.37559775 reg_l1 31.022465 reg_l2 12.863092
loss 3.4778442
STEP 25 ================================
prereg loss 0.37569952 reg_l1 31.01719 reg_l2 12.862217
loss 3.4774187
STEP 26 ================================
prereg loss 0.37581128 reg_l1 31.011728 reg_l2 12.861235
loss 3.4769843
STEP 27 ================================
prereg loss 0.375498 reg_l1 31.006163 reg_l2 12.86017
loss 3.4761143
STEP 28 ================================
prereg loss 0.37506106 reg_l1 31.000616 reg_l2 12.859151
loss 3.4751227
STEP 29 ================================
prereg loss 0.3750337 reg_l1 30.99509 reg_l2 12.858224
loss 3.4745426
STEP 30 ================================
prereg loss 0.3750854 reg_l1 30.989485 reg_l2 12.857401
loss 3.4740338
STEP 31 ================================
prereg loss 0.37490726 reg_l1 30.984163 reg_l2 12.856641
loss 3.4733236
STEP 32 ================================
prereg loss 0.37455562 reg_l1 30.978998 reg_l2 12.855837
loss 3.4724555
STEP 33 ================================
prereg loss 0.3743316 reg_l1 30.97375 reg_l2 12.854909
loss 3.4717064
STEP 34 ================================
prereg loss 0.37431085 reg_l1 30.968142 reg_l2 12.853831
loss 3.4711251
STEP 35 ================================
prereg loss 0.37413582 reg_l1 30.962364 reg_l2 12.852666
loss 3.4703722
STEP 36 ================================
prereg loss 0.3741636 reg_l1 30.9571 reg_l2 12.851464
loss 3.4698737
STEP 37 ================================
prereg loss 0.37422672 reg_l1 30.951874 reg_l2 12.850308
loss 3.4694142
STEP 38 ================================
prereg loss 0.3738994 reg_l1 30.947035 reg_l2 12.84927
loss 3.468603
STEP 39 ================================
prereg loss 0.37364778 reg_l1 30.942268 reg_l2 12.848331
loss 3.4678745
STEP 40 ================================
prereg loss 0.37355778 reg_l1 30.937408 reg_l2 12.847433
loss 3.4672987
STEP 41 ================================
prereg loss 0.37388077 reg_l1 30.932302 reg_l2 12.846515
loss 3.467111
STEP 42 ================================
prereg loss 0.3736674 reg_l1 30.927137 reg_l2 12.845547
loss 3.4663813
STEP 43 ================================
prereg loss 0.37318385 reg_l1 30.92192 reg_l2 12.844535
loss 3.465376
STEP 44 ================================
prereg loss 0.37317103 reg_l1 30.916445 reg_l2 12.843476
loss 3.4648156
STEP 45 ================================
prereg loss 0.3732788 reg_l1 30.910698 reg_l2 12.842393
loss 3.4643488
STEP 46 ================================
prereg loss 0.373569 reg_l1 30.905088 reg_l2 12.841293
loss 3.464078
STEP 47 ================================
prereg loss 0.3729804 reg_l1 30.899818 reg_l2 12.840225
loss 3.4629622
STEP 48 ================================
prereg loss 0.3727244 reg_l1 30.894554 reg_l2 12.839139
loss 3.4621797
STEP 49 ================================
prereg loss 0.37263262 reg_l1 30.889017 reg_l2 12.838045
loss 3.4615345
STEP 50 ================================
prereg loss 0.3726988 reg_l1 30.883305 reg_l2 12.836948
loss 3.4610293
2022-08-14T17:53:04.633

julia> interleaving_steps!(90, 3)
2022-08-14T17:53:22.005
STEP 1 ================================
prereg loss 0.37307394 reg_l1 30.877487 reg_l2 12.835825
loss 3.4608226
cutoff 0.0001957174 network size 241
STEP 2 ================================
prereg loss 0.37256458 reg_l1 30.872046 reg_l2 12.834713
loss 3.4597692
STEP 3 ================================
prereg loss 0.37219355 reg_l1 30.866735 reg_l2 12.833641
loss 3.458867
STEP 4 ================================
prereg loss 0.37205714 reg_l1 30.861404 reg_l2 12.8326025
loss 3.4581976
cutoff 0.0013163399 network size 240
STEP 5 ================================
prereg loss 0.37206322 reg_l1 30.854671 reg_l2 12.831594
loss 3.4575303
STEP 6 ================================
prereg loss 0.37236023 reg_l1 30.849339 reg_l2 12.830579
loss 3.4572942
STEP 7 ================================
prereg loss 0.37201568 reg_l1 30.844128 reg_l2 12.829527
loss 3.4564285
cutoff 0.00066800433 network size 239
STEP 8 ================================
prereg loss 0.37173244 reg_l1 30.838133 reg_l2 12.828406
loss 3.455546
STEP 9 ================================
prereg loss 0.37155616 reg_l1 30.83285 reg_l2 12.827196
loss 3.4548411
STEP 10 ================================
prereg loss 0.37152287 reg_l1 30.827335 reg_l2 12.825947
loss 3.4542565
cutoff 0.0006355488 network size 238
STEP 11 ================================
prereg loss 0.37165016 reg_l1 30.821127 reg_l2 12.824693
loss 3.453763
STEP 12 ================================
prereg loss 0.37148052 reg_l1 30.81601 reg_l2 12.823506
loss 3.4530814
STEP 13 ================================
prereg loss 0.37131736 reg_l1 30.810966 reg_l2 12.82237
loss 3.452414
cutoff 0.00050829747 network size 237
STEP 14 ================================
prereg loss 0.3711789 reg_l1 30.805365 reg_l2 12.821269
loss 3.4517155
STEP 15 ================================
prereg loss 0.3712536 reg_l1 30.800303 reg_l2 12.820156
loss 3.451284
STEP 16 ================================
prereg loss 0.37101117 reg_l1 30.79523 reg_l2 12.819015
loss 3.4505343
cutoff 0.00053516746 network size 236
STEP 17 ================================
prereg loss 0.37078616 reg_l1 30.789696 reg_l2 12.81784
loss 3.449756
STEP 18 ================================
prereg loss 0.37097156 reg_l1 30.784729 reg_l2 12.816629
loss 3.4494443
STEP 19 ================================
prereg loss 0.37083822 reg_l1 30.779856 reg_l2 12.815442
loss 3.4488237
cutoff 0.0008047546 network size 235
STEP 20 ================================
prereg loss 0.3704775 reg_l1 30.774378 reg_l2 12.814277
loss 3.4479153
STEP 21 ================================
prereg loss 0.3702982 reg_l1 30.76966 reg_l2 12.813126
loss 3.4472642
STEP 22 ================================
prereg loss 0.37023365 reg_l1 30.764824 reg_l2 12.811948
loss 3.4467163
cutoff 0.0018154165 network size 234
STEP 23 ================================
prereg loss 0.37029356 reg_l1 30.75814 reg_l2 12.810716
loss 3.4461076
STEP 24 ================================
prereg loss 0.3703203 reg_l1 30.753292 reg_l2 12.809391
loss 3.4456496
STEP 25 ================================
prereg loss 0.3701276 reg_l1 30.748352 reg_l2 12.807957
loss 3.444963
cutoff 0.0017259921 network size 233
STEP 26 ================================
prereg loss 0.3702687 reg_l1 30.74152 reg_l2 12.806456
loss 3.4444208
STEP 27 ================================
prereg loss 0.36999273 reg_l1 30.736689 reg_l2 12.805161
loss 3.4436617
STEP 28 ================================
prereg loss 0.37010413 reg_l1 30.731934 reg_l2 12.804008
loss 3.4432974
cutoff 0.0020221493 network size 232
STEP 29 ================================
prereg loss 0.46101138 reg_l1 30.725237 reg_l2 12.80287
loss 3.5335352
STEP 30 ================================
prereg loss 0.40788367 reg_l1 30.71836 reg_l2 12.797126
loss 3.4797199
STEP 31 ================================
prereg loss 0.3979196 reg_l1 30.709461 reg_l2 12.78916
loss 3.4688659
cutoff 0.0023644248 network size 231
STEP 32 ================================
prereg loss 0.43236482 reg_l1 30.697845 reg_l2 12.781778
loss 3.5021496
STEP 33 ================================
prereg loss 0.44759077 reg_l1 30.690702 reg_l2 12.777206
loss 3.5166612
STEP 34 ================================
prereg loss 0.41973892 reg_l1 30.687742 reg_l2 12.776263
loss 3.4885132
cutoff 0.0026904831 network size 230
STEP 35 ================================
prereg loss 0.4003136 reg_l1 30.686409 reg_l2 12.778421
loss 3.4689546
STEP 36 ================================
prereg loss 0.41929996 reg_l1 30.68955 reg_l2 12.782122
loss 3.488255
STEP 37 ================================
prereg loss 0.42344168 reg_l1 30.690996 reg_l2 12.785397
loss 3.4925413
cutoff 0.00255246 network size 229
STEP 38 ================================
prereg loss 0.40565336 reg_l1 30.685585 reg_l2 12.786144
loss 3.4742117
STEP 39 ================================
prereg loss 0.4176421 reg_l1 30.677568 reg_l2 12.783501
loss 3.485399
STEP 40 ================================
prereg loss 0.44175416 reg_l1 30.66645 reg_l2 12.778109
loss 3.5083992
cutoff 0.0022264933 network size 228
STEP 41 ================================
prereg loss 0.42188022 reg_l1 30.6526 reg_l2 12.771745
loss 3.4871402
STEP 42 ================================
prereg loss 0.39531064 reg_l1 30.642931 reg_l2 12.766508
loss 3.4596038
STEP 43 ================================
prereg loss 0.40317994 reg_l1 30.635283 reg_l2 12.763765
loss 3.4667082
cutoff 0.0034373263 network size 227
STEP 44 ================================
prereg loss 0.41450882 reg_l1 30.62665 reg_l2 12.763997
loss 3.4771738
STEP 45 ================================
prereg loss 0.4020807 reg_l1 30.62387 reg_l2 12.766405
loss 3.464468
STEP 46 ================================
prereg loss 0.38718286 reg_l1 30.622463 reg_l2 12.769267
loss 3.4494293
cutoff 0.0035235204 network size 226
STEP 47 ================================
prereg loss 0.3918007 reg_l1 30.617151 reg_l2 12.770724
loss 3.4535158
STEP 48 ================================
prereg loss 0.40324423 reg_l1 30.613483 reg_l2 12.769829
loss 3.4645927
STEP 49 ================================
prereg loss 0.40024415 reg_l1 30.60716 reg_l2 12.766646
loss 3.4609604
cutoff 0.0034154071 network size 225
STEP 50 ================================
prereg loss 0.38658467 reg_l1 30.595844 reg_l2 12.762225
loss 3.4461691
STEP 51 ================================
prereg loss 0.3846714 reg_l1 30.5884 reg_l2 12.758044
loss 3.4435115
STEP 52 ================================
prereg loss 0.39617202 reg_l1 30.582973 reg_l2 12.755379
loss 3.4544694
cutoff 0.0036447966 network size 224
STEP 53 ================================
prereg loss 0.39987716 reg_l1 30.57639 reg_l2 12.754786
loss 3.457516
STEP 54 ================================
prereg loss 0.38643605 reg_l1 30.574873 reg_l2 12.755845
loss 3.4439232
STEP 55 ================================
prereg loss 0.37936905 reg_l1 30.573385 reg_l2 12.757312
loss 3.4367075
cutoff 0.0036114478 network size 223
STEP 56 ================================
prereg loss 0.3897603 reg_l1 30.566973 reg_l2 12.757886
loss 3.4464576
STEP 57 ================================
prereg loss 0.39892465 reg_l1 30.562487 reg_l2 12.7568245
loss 3.4551733
STEP 58 ================================
prereg loss 0.38937753 reg_l1 30.55645 reg_l2 12.754145
loss 3.4450226
cutoff 0.003712773 network size 222
STEP 59 ================================
prereg loss 0.38745415 reg_l1 30.54557 reg_l2 12.750558
loss 3.4420114
STEP 60 ================================
prereg loss 0.3813537 reg_l1 30.538988 reg_l2 12.7472925
loss 3.4352524
STEP 61 ================================
prereg loss 0.39404964 reg_l1 30.533352 reg_l2 12.745237
loss 3.4473848
cutoff 0.003767151 network size 221
STEP 62 ================================
prereg loss 0.39124236 reg_l1 30.524727 reg_l2 12.744627
loss 3.443715
STEP 63 ================================
prereg loss 0.3813543 reg_l1 30.520826 reg_l2 12.7450075
loss 3.433437
STEP 64 ================================
prereg loss 0.38069302 reg_l1 30.517292 reg_l2 12.745402
loss 3.4324222
cutoff 0.0037866249 network size 220
STEP 65 ================================
prereg loss 0.38944238 reg_l1 30.509481 reg_l2 12.744995
loss 3.4403906
STEP 66 ================================
prereg loss 0.38925704 reg_l1 30.504818 reg_l2 12.743532
loss 3.4397388
STEP 67 ================================
prereg loss 0.3813849 reg_l1 30.499289 reg_l2 12.741282
loss 3.4313138
cutoff 0.0038053233 network size 219
STEP 68 ================================
prereg loss 0.37797236 reg_l1 30.489403 reg_l2 12.738828
loss 3.4269128
STEP 69 ================================
prereg loss 0.38165888 reg_l1 30.483788 reg_l2 12.736961
loss 3.4300375
STEP 70 ================================
prereg loss 0.38401017 reg_l1 30.479698 reg_l2 12.736162
loss 3.43198
cutoff 0.0038128665 network size 218
STEP 71 ================================
prereg loss 0.37996343 reg_l1 30.473236 reg_l2 12.736344
loss 3.427287
STEP 72 ================================
prereg loss 0.37598982 reg_l1 30.471136 reg_l2 12.737036
loss 3.4231036
STEP 73 ================================
prereg loss 0.37749478 reg_l1 30.468388 reg_l2 12.737479
loss 3.4243336
cutoff 0.003977218 network size 217
STEP 74 ================================
prereg loss 0.38329774 reg_l1 30.460361 reg_l2 12.737061
loss 3.429334
STEP 75 ================================
prereg loss 0.38197866 reg_l1 30.455055 reg_l2 12.735586
loss 3.4274843
STEP 76 ================================
prereg loss 0.37698936 reg_l1 30.449108 reg_l2 12.733301
loss 3.4219003
cutoff 0.0040748003 network size 216
STEP 77 ================================
prereg loss 0.37635118 reg_l1 30.439135 reg_l2 12.730801
loss 3.4202647
STEP 78 ================================
prereg loss 0.3796552 reg_l1 30.433754 reg_l2 12.728739
loss 3.4230306
STEP 79 ================================
prereg loss 0.37847692 reg_l1 30.428837 reg_l2 12.727483
loss 3.4213605
cutoff 0.004083824 network size 215
STEP 80 ================================
prereg loss 0.37842813 reg_l1 30.420292 reg_l2 12.726946
loss 3.4204574
STEP 81 ================================
prereg loss 0.37742352 reg_l1 30.416403 reg_l2 12.726925
loss 3.4190638
STEP 82 ================================
prereg loss 0.37789622 reg_l1 30.412601 reg_l2 12.726851
loss 3.4191566
cutoff 0.0039652833 network size 214
STEP 83 ================================
prereg loss 0.37652567 reg_l1 30.404377 reg_l2 12.726315
loss 3.4169633
STEP 84 ================================
prereg loss 0.37499216 reg_l1 30.399307 reg_l2 12.7252035
loss 3.414923
STEP 85 ================================
prereg loss 0.3747152 reg_l1 30.39369 reg_l2 12.723701
loss 3.4140844
cutoff 0.004118206 network size 213
STEP 86 ================================
prereg loss 0.37552288 reg_l1 30.38417 reg_l2 12.722144
loss 3.41394
STEP 87 ================================
prereg loss 0.37498602 reg_l1 30.37971 reg_l2 12.720932
loss 3.412957
STEP 88 ================================
prereg loss 0.3731542 reg_l1 30.376192 reg_l2 12.720194
loss 3.4107735
cutoff 0.003916168 network size 212
STEP 89 ================================
prereg loss 0.37200975 reg_l1 30.368969 reg_l2 12.719833
loss 3.4089067
STEP 90 ================================
prereg loss 0.3723926 reg_l1 30.36562 reg_l2 12.719693
loss 3.4089546
2022-08-14T18:05:58.765

julia> serialize("cf-s2-212-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-212-parameters-opt.ser", opt)
```

OK, that worked. Let's do 20 sparsifications in 100 steps:

```
julia> interleaving_steps!(100, 5)
2022-08-14T18:08:28.363
STEP 1 ================================
prereg loss 0.37414423 reg_l1 30.362053 reg_l2 12.719448
loss 3.4103496
cutoff 0.0042057857 network size 211
STEP 2 ================================
prereg loss 0.37293002 reg_l1 30.354053 reg_l2 12.718899
loss 3.4083354
STEP 3 ================================
prereg loss 0.3706301 reg_l1 30.349976 reg_l2 12.718021
loss 3.4056277
STEP 4 ================================
prereg loss 0.37024403 reg_l1 30.34542 reg_l2 12.716913
loss 3.404786
STEP 5 ================================
prereg loss 0.3701908 reg_l1 30.340677 reg_l2 12.715784
loss 3.4042587
STEP 6 ================================
prereg loss 0.3698083 reg_l1 30.335999 reg_l2 12.714774
loss 3.403408
cutoff 0.0043480815 network size 210
STEP 7 ================================
prereg loss 0.36983263 reg_l1 30.32722 reg_l2 12.713878
loss 3.402555
STEP 8 ================================
prereg loss 0.37005508 reg_l1 30.322886 reg_l2 12.713051
loss 3.4023438
STEP 9 ================================
prereg loss 0.36972785 reg_l1 30.318493 reg_l2 12.712159
loss 3.4015772
STEP 10 ================================
prereg loss 0.36858833 reg_l1 30.314066 reg_l2 12.711213
loss 3.3999949
STEP 11 ================================
prereg loss 0.36826202 reg_l1 30.309563 reg_l2 12.710225
loss 3.3992183
cutoff 0.0046929824 network size 209
STEP 12 ================================
prereg loss 0.37443542 reg_l1 30.300285 reg_l2 12.709236
loss 3.404464
STEP 13 ================================
prereg loss 0.37322816 reg_l1 30.296116 reg_l2 12.708704
loss 3.4028397
STEP 14 ================================
prereg loss 0.37197104 reg_l1 30.292364 reg_l2 12.708413
loss 3.4012077
STEP 15 ================================
prereg loss 0.37109426 reg_l1 30.28862 reg_l2 12.708077
loss 3.3999562
STEP 16 ================================
prereg loss 0.369632 reg_l1 30.284645 reg_l2 12.707479
loss 3.3980966
cutoff 0.004345861 network size 208
STEP 17 ================================
prereg loss 0.36817795 reg_l1 30.275858 reg_l2 12.706546
loss 3.3957636
STEP 18 ================================
prereg loss 0.36788553 reg_l1 30.270992 reg_l2 12.705398
loss 3.394985
STEP 19 ================================
prereg loss 0.3682136 reg_l1 30.266031 reg_l2 12.704167
loss 3.3948169
STEP 20 ================================
prereg loss 0.36787832 reg_l1 30.26125 reg_l2 12.702971
loss 3.3940034
STEP 21 ================================
prereg loss 0.36669615 reg_l1 30.256363 reg_l2 12.701789
loss 3.3923326
cutoff 0.0046791295 network size 207
STEP 22 ================================
prereg loss 0.36594263 reg_l1 30.246704 reg_l2 12.700487
loss 3.390613
STEP 23 ================================
prereg loss 0.3659835 reg_l1 30.241491 reg_l2 12.699018
loss 3.3901327
STEP 24 ================================
prereg loss 0.36629844 reg_l1 30.235804 reg_l2 12.697293
loss 3.3898787
STEP 25 ================================
prereg loss 0.36566737 reg_l1 30.229877 reg_l2 12.695381
loss 3.3886552
STEP 26 ================================
prereg loss 0.3649684 reg_l1 30.223944 reg_l2 12.6934595
loss 3.3873627
cutoff 0.0031650641 network size 206
STEP 27 ================================
prereg loss 0.36477882 reg_l1 30.214912 reg_l2 12.691715
loss 3.38627
STEP 28 ================================
prereg loss 0.36493853 reg_l1 30.209442 reg_l2 12.6901655
loss 3.3858829
STEP 29 ================================
prereg loss 0.3650063 reg_l1 30.20427 reg_l2 12.688881
loss 3.3854332
STEP 30 ================================
prereg loss 0.36479932 reg_l1 30.199581 reg_l2 12.68787
loss 3.3847575
STEP 31 ================================
prereg loss 0.3643878 reg_l1 30.195364 reg_l2 12.687048
loss 3.3839242
cutoff 0.004258824 network size 205
STEP 32 ================================
prereg loss 0.36463505 reg_l1 30.18709 reg_l2 12.686304
loss 3.383344
STEP 33 ================================
prereg loss 0.36453557 reg_l1 30.18282 reg_l2 12.68548
loss 3.3828175
STEP 34 ================================
prereg loss 0.3645319 reg_l1 30.178679 reg_l2 12.684616
loss 3.3823998
STEP 35 ================================
prereg loss 0.3635945 reg_l1 30.174904 reg_l2 12.6838
loss 3.381085
STEP 36 ================================
prereg loss 0.36298358 reg_l1 30.171293 reg_l2 12.683084
loss 3.3801131
cutoff 0.0033587352 network size 204
STEP 37 ================================
prereg loss 0.36264458 reg_l1 30.164288 reg_l2 12.682484
loss 3.3790734
STEP 38 ================================
prereg loss 0.3622246 reg_l1 30.160656 reg_l2 12.681968
loss 3.3782902
STEP 39 ================================
prereg loss 0.36229625 reg_l1 30.156775 reg_l2 12.681371
loss 3.3779738
STEP 40 ================================
prereg loss 0.36236003 reg_l1 30.152626 reg_l2 12.6805525
loss 3.3776226
STEP 41 ================================
prereg loss 0.3618086 reg_l1 30.14821 reg_l2 12.679446
loss 3.3766296
cutoff 0.0037911027 network size 203
STEP 42 ================================
prereg loss 0.36125103 reg_l1 30.13969 reg_l2 12.678103
loss 3.3752203
STEP 43 ================================
prereg loss 0.36107448 reg_l1 30.13472 reg_l2 12.676643
loss 3.3745465
STEP 44 ================================
prereg loss 0.3611927 reg_l1 30.129585 reg_l2 12.675197
loss 3.3741512
STEP 45 ================================
prereg loss 0.36152077 reg_l1 30.12455 reg_l2 12.6738825
loss 3.3739758
STEP 46 ================================
prereg loss 0.3617512 reg_l1 30.119745 reg_l2 12.672716
loss 3.373726
cutoff 0.0037949497 network size 202
STEP 47 ================================
prereg loss 0.3614903 reg_l1 30.11161 reg_l2 12.671755
loss 3.3726513
STEP 48 ================================
prereg loss 0.36099988 reg_l1 30.10763 reg_l2 12.671083
loss 3.3717628
STEP 49 ================================
prereg loss 0.36093643 reg_l1 30.103651 reg_l2 12.670554
loss 3.3713017
STEP 50 ================================
prereg loss 0.36073455 reg_l1 30.099714 reg_l2 12.670005
loss 3.3707058
STEP 51 ================================
prereg loss 0.35992005 reg_l1 30.095804 reg_l2 12.669364
loss 3.3695004
cutoff 0.0038108623 network size 201
STEP 52 ================================
prereg loss 0.35983142 reg_l1 30.087973 reg_l2 12.668567
loss 3.3686287
STEP 53 ================================
prereg loss 0.3594543 reg_l1 30.084154 reg_l2 12.6677265
loss 3.3678699
STEP 54 ================================
prereg loss 0.3591038 reg_l1 30.08037 reg_l2 12.666899
loss 3.3671408
STEP 55 ================================
prereg loss 0.35879207 reg_l1 30.076591 reg_l2 12.6660795
loss 3.3664513
STEP 56 ================================
prereg loss 0.35859525 reg_l1 30.072712 reg_l2 12.66525
loss 3.3658667
cutoff 0.0044108056 network size 200
STEP 57 ================================
prereg loss 0.36050054 reg_l1 30.064232 reg_l2 12.664299
loss 3.3669238
STEP 58 ================================
prereg loss 0.3588725 reg_l1 30.06054 reg_l2 12.6637
loss 3.3649263
STEP 59 ================================
prereg loss 0.35917324 reg_l1 30.05685 reg_l2 12.663151
loss 3.3648584
STEP 60 ================================
prereg loss 0.35934123 reg_l1 30.053022 reg_l2 12.662472
loss 3.3646433
STEP 61 ================================
prereg loss 0.35807508 reg_l1 30.048878 reg_l2 12.661554
loss 3.362963
cutoff 0.004064085 network size 199
STEP 62 ================================
prereg loss 0.357923 reg_l1 30.04023 reg_l2 12.660426
loss 3.361946
STEP 63 ================================
prereg loss 0.35826677 reg_l1 30.035381 reg_l2 12.6591
loss 3.361805
STEP 64 ================================
prereg loss 0.358827 reg_l1 30.030825 reg_l2 12.657816
loss 3.3619094
STEP 65 ================================
prereg loss 0.3577227 reg_l1 30.026918 reg_l2 12.656773
loss 3.3604147
STEP 66 ================================
prereg loss 0.3567301 reg_l1 30.023441 reg_l2 12.656004
loss 3.359074
cutoff 0.0042310273 network size 198
STEP 67 ================================
prereg loss 0.35766873 reg_l1 30.015652 reg_l2 12.655358
loss 3.3592339
STEP 68 ================================
prereg loss 0.357702 reg_l1 30.011812 reg_l2 12.654745
loss 3.3588834
STEP 69 ================================
prereg loss 0.35877687 reg_l1 30.007654 reg_l2 12.653942
loss 3.3595424
STEP 70 ================================
prereg loss 0.35735568 reg_l1 30.003567 reg_l2 12.652876
loss 3.3577123
STEP 71 ================================
prereg loss 0.35632712 reg_l1 29.999517 reg_l2 12.651597
loss 3.356279
cutoff 0.0042602555 network size 197
STEP 72 ================================
prereg loss 0.35634658 reg_l1 29.990963 reg_l2 12.650175
loss 3.355443
STEP 73 ================================
prereg loss 0.35575223 reg_l1 29.98654 reg_l2 12.648852
loss 3.3544064
STEP 74 ================================
prereg loss 0.35599038 reg_l1 29.982155 reg_l2 12.647621
loss 3.3542058
STEP 75 ================================
prereg loss 0.35672635 reg_l1 29.977976 reg_l2 12.646436
loss 3.3545241
STEP 76 ================================
prereg loss 0.35524926 reg_l1 29.974176 reg_l2 12.645289
loss 3.3526669
cutoff 0.004278092 network size 196
STEP 77 ================================
prereg loss 0.3544076 reg_l1 29.966013 reg_l2 12.644086
loss 3.351009
STEP 78 ================================
prereg loss 0.35413447 reg_l1 29.961884 reg_l2 12.642875
loss 3.350323
STEP 79 ================================
prereg loss 0.3542745 reg_l1 29.957539 reg_l2 12.641637
loss 3.3500285
STEP 80 ================================
prereg loss 0.3547817 reg_l1 29.953238 reg_l2 12.640373
loss 3.3501055
STEP 81 ================================
prereg loss 0.35387686 reg_l1 29.949177 reg_l2 12.639116
loss 3.3487945
cutoff 0.0044709328 network size 195
STEP 82 ================================
prereg loss 0.35315132 reg_l1 29.940746 reg_l2 12.63787
loss 3.347226
STEP 83 ================================
prereg loss 0.3529505 reg_l1 29.936493 reg_l2 12.63666
loss 3.3465998
STEP 84 ================================
prereg loss 0.3532703 reg_l1 29.93206 reg_l2 12.6355095
loss 3.3464763
STEP 85 ================================
prereg loss 0.3530634 reg_l1 29.927898 reg_l2 12.634447
loss 3.3458533
STEP 86 ================================
prereg loss 0.35232133 reg_l1 29.924103 reg_l2 12.633477
loss 3.3447318
cutoff 0.0047678296 network size 194
STEP 87 ================================
prereg loss 0.35191533 reg_l1 29.915585 reg_l2 12.63251
loss 3.343474
STEP 88 ================================
prereg loss 0.3515555 reg_l1 29.911625 reg_l2 12.631532
loss 3.3427181
STEP 89 ================================
prereg loss 0.35163447 reg_l1 29.907417 reg_l2 12.630415
loss 3.3423762
STEP 90 ================================
prereg loss 0.35193127 reg_l1 29.903059 reg_l2 12.629111
loss 3.3422372
STEP 91 ================================
prereg loss 0.3515347 reg_l1 29.898611 reg_l2 12.627639
loss 3.3413959
cutoff 0.0045635533 network size 193
STEP 92 ================================
prereg loss 0.35138005 reg_l1 29.889547 reg_l2 12.626047
loss 3.340335
STEP 93 ================================
prereg loss 0.35119572 reg_l1 29.884949 reg_l2 12.62451
loss 3.3396907
STEP 94 ================================
prereg loss 0.35119295 reg_l1 29.880245 reg_l2 12.623067
loss 3.3392174
STEP 95 ================================
prereg loss 0.35148665 reg_l1 29.87557 reg_l2 12.621723
loss 3.3390439
STEP 96 ================================
prereg loss 0.35105085 reg_l1 29.871166 reg_l2 12.620496
loss 3.3381674
cutoff 0.005142027 network size 192
STEP 97 ================================
prereg loss 0.35091218 reg_l1 29.861784 reg_l2 12.619306
loss 3.3370905
STEP 98 ================================
prereg loss 0.35064092 reg_l1 29.85749 reg_l2 12.618023
loss 3.33639
STEP 99 ================================
prereg loss 0.3505802 reg_l1 29.853008 reg_l2 12.616666
loss 3.335881
STEP 100 ================================
prereg loss 0.35085648 reg_l1 29.848476 reg_l2 12.615307
loss 3.3357043
2022-08-14T18:21:37.353

julia> serialize("cf-s2-192-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-192-parameters-opt.ser", opt)
```

Let's try a more conservative rate: 10 sparsifications at 100 steps.

```
julia> interleaving_steps!(100, 10)
2022-08-14T18:24:29.888
STEP 1 ================================
prereg loss 0.35035914 reg_l1 29.844269 reg_l2 12.614031
loss 3.3347862
cutoff 0.0053967563 network size 191
STEP 2 ================================
prereg loss 0.35041845 reg_l1 29.83482 reg_l2 12.612855
loss 3.3339005
STEP 3 ================================
prereg loss 0.35010326 reg_l1 29.830671 reg_l2 12.611753
loss 3.3331704
STEP 4 ================================
prereg loss 0.35029453 reg_l1 29.826466 reg_l2 12.610673
loss 3.3329413
STEP 5 ================================
prereg loss 0.34993216 reg_l1 29.822409 reg_l2 12.609604
loss 3.332173
STEP 6 ================================
prereg loss 0.34921002 reg_l1 29.818434 reg_l2 12.60854
loss 3.3310535
STEP 7 ================================
prereg loss 0.34888345 reg_l1 29.814318 reg_l2 12.607486
loss 3.330315
STEP 8 ================================
prereg loss 0.34877163 reg_l1 29.810093 reg_l2 12.606487
loss 3.3297808
STEP 9 ================================
prereg loss 0.3487234 reg_l1 29.805893 reg_l2 12.605513
loss 3.3293128
STEP 10 ================================
prereg loss 0.34838825 reg_l1 29.80176 reg_l2 12.604507
loss 3.3285642
STEP 11 ================================
prereg loss 0.3479333 reg_l1 29.797554 reg_l2 12.603394
loss 3.3276887
cutoff 0.0053907637 network size 190
STEP 12 ================================
prereg loss 0.34768218 reg_l1 29.787733 reg_l2 12.602083
loss 3.3264556
STEP 13 ================================
prereg loss 0.34764394 reg_l1 29.78307 reg_l2 12.600662
loss 3.3259509
STEP 14 ================================
prereg loss 0.34786108 reg_l1 29.778255 reg_l2 12.599161
loss 3.3256867
STEP 15 ================================
prereg loss 0.34744382 reg_l1 29.773643 reg_l2 12.597704
loss 3.3248081
STEP 16 ================================
prereg loss 0.34723637 reg_l1 29.769148 reg_l2 12.596365
loss 3.3241513
STEP 17 ================================
prereg loss 0.34702715 reg_l1 29.764769 reg_l2 12.595186
loss 3.323504
STEP 18 ================================
prereg loss 0.3469692 reg_l1 29.760496 reg_l2 12.594109
loss 3.3230188
STEP 19 ================================
prereg loss 0.34650272 reg_l1 29.756405 reg_l2 12.593103
loss 3.3221433
STEP 20 ================================
prereg loss 0.34604156 reg_l1 29.752373 reg_l2 12.592076
loss 3.321279
STEP 21 ================================
prereg loss 0.34586257 reg_l1 29.748129 reg_l2 12.590979
loss 3.3206756
cutoff 0.005471695 network size 189
STEP 22 ================================
prereg loss 0.3459375 reg_l1 29.73827 reg_l2 12.589761
loss 3.3197646
STEP 23 ================================
prereg loss 0.3458546 reg_l1 29.733913 reg_l2 12.588527
loss 3.3192458
STEP 24 ================================
prereg loss 0.34539387 reg_l1 29.729666 reg_l2 12.587266
loss 3.3183606
STEP 25 ================================
prereg loss 0.34499624 reg_l1 29.725409 reg_l2 12.586009
loss 3.317537
STEP 26 ================================
prereg loss 0.3448255 reg_l1 29.72103 reg_l2 12.58474
loss 3.3169284
STEP 27 ================================
prereg loss 0.34487772 reg_l1 29.716509 reg_l2 12.583456
loss 3.3165286
STEP 28 ================================
prereg loss 0.34514272 reg_l1 29.71194 reg_l2 12.582135
loss 3.3163366
STEP 29 ================================
prereg loss 0.34445658 reg_l1 29.707596 reg_l2 12.580831
loss 3.3152163
STEP 30 ================================
prereg loss 0.34414098 reg_l1 29.703222 reg_l2 12.579553
loss 3.3144634
STEP 31 ================================
prereg loss 0.34400004 reg_l1 29.698778 reg_l2 12.578318
loss 3.313878
cutoff 0.005425969 network size 188
STEP 32 ================================
prereg loss 0.34425798 reg_l1 29.688847 reg_l2 12.577075
loss 3.3131428
STEP 33 ================================
prereg loss 0.34389684 reg_l1 29.684542 reg_l2 12.575869
loss 3.312351
STEP 34 ================================
prereg loss 0.3433545 reg_l1 29.68036 reg_l2 12.574684
loss 3.3113906
STEP 35 ================================
prereg loss 0.34315637 reg_l1 29.676058 reg_l2 12.573517
loss 3.3107622
STEP 36 ================================
prereg loss 0.34314045 reg_l1 29.671692 reg_l2 12.572349
loss 3.3103096
STEP 37 ================================
prereg loss 0.3431179 reg_l1 29.667383 reg_l2 12.571192
loss 3.3098564
STEP 38 ================================
prereg loss 0.34287733 reg_l1 29.663073 reg_l2 12.569971
loss 3.3091848
STEP 39 ================================
prereg loss 0.34253284 reg_l1 29.65868 reg_l2 12.568654
loss 3.3084009
STEP 40 ================================
prereg loss 0.34231487 reg_l1 29.654108 reg_l2 12.567261
loss 3.307726
STEP 41 ================================
prereg loss 0.34227195 reg_l1 29.649466 reg_l2 12.565845
loss 3.3072186
cutoff 0.0050184727 network size 187
STEP 42 ================================
prereg loss 0.34220615 reg_l1 29.639843 reg_l2 12.564453
loss 3.3061905
STEP 43 ================================
prereg loss 0.34216267 reg_l1 29.635397 reg_l2 12.563121
loss 3.3057024
STEP 44 ================================
prereg loss 0.34169602 reg_l1 29.63112 reg_l2 12.561843
loss 3.3048081
STEP 45 ================================
prereg loss 0.34151772 reg_l1 29.62678 reg_l2 12.560606
loss 3.3041956
STEP 46 ================================
prereg loss 0.34157446 reg_l1 29.622387 reg_l2 12.559394
loss 3.3038132
STEP 47 ================================
prereg loss 0.34125122 reg_l1 29.618134 reg_l2 12.558207
loss 3.3030646
STEP 48 ================================
prereg loss 0.34104282 reg_l1 29.613838 reg_l2 12.55701
loss 3.3024266
STEP 49 ================================
prereg loss 0.34089583 reg_l1 29.609512 reg_l2 12.555794
loss 3.3018472
STEP 50 ================================
prereg loss 0.34080207 reg_l1 29.605122 reg_l2 12.554539
loss 3.3013144
STEP 51 ================================
prereg loss 0.34065554 reg_l1 29.600708 reg_l2 12.553261
loss 3.3007264
cutoff 0.0056298366 network size 186
STEP 52 ================================
prereg loss 0.34046683 reg_l1 29.59064 reg_l2 12.551938
loss 3.2995307
STEP 53 ================================
prereg loss 0.3403296 reg_l1 29.586138 reg_l2 12.550609
loss 3.2989435
STEP 54 ================================
prereg loss 0.34025648 reg_l1 29.581608 reg_l2 12.5492735
loss 3.2984173
STEP 55 ================================
prereg loss 0.3399225 reg_l1 29.577206 reg_l2 12.547981
loss 3.297643
STEP 56 ================================
prereg loss 0.3397584 reg_l1 29.57277 reg_l2 12.546732
loss 3.2970355
STEP 57 ================================
prereg loss 0.33982113 reg_l1 29.56825 reg_l2 12.545486
loss 3.296646
STEP 58 ================================
prereg loss 0.33961156 reg_l1 29.563858 reg_l2 12.544246
loss 3.2959974
STEP 59 ================================
prereg loss 0.3394376 reg_l1 29.55945 reg_l2 12.542991
loss 3.2953825
STEP 60 ================================
prereg loss 0.33928695 reg_l1 29.555014 reg_l2 12.541738
loss 3.2947884
STEP 61 ================================
prereg loss 0.33919847 reg_l1 29.550549 reg_l2 12.540469
loss 3.2942533
cutoff 0.0049799797 network size 185
STEP 62 ================================
prereg loss 0.33904862 reg_l1 29.541136 reg_l2 12.53918
loss 3.2931623
STEP 63 ================================
prereg loss 0.33896077 reg_l1 29.536766 reg_l2 12.537876
loss 3.2926373
STEP 64 ================================
prereg loss 0.33864093 reg_l1 29.532497 reg_l2 12.536576
loss 3.2918906
STEP 65 ================================
prereg loss 0.33854997 reg_l1 29.528133 reg_l2 12.535292
loss 3.2913632
STEP 66 ================================
prereg loss 0.33853364 reg_l1 29.523787 reg_l2 12.534061
loss 3.2909124
STEP 67 ================================
prereg loss 0.33868623 reg_l1 29.519407 reg_l2 12.532826
loss 3.290627
STEP 68 ================================
prereg loss 0.33814126 reg_l1 29.515198 reg_l2 12.53159
loss 3.289661
STEP 69 ================================
prereg loss 0.3379463 reg_l1 29.510855 reg_l2 12.530323
loss 3.289032
STEP 70 ================================
prereg loss 0.33793005 reg_l1 29.506424 reg_l2 12.529053
loss 3.2885723
STEP 71 ================================
prereg loss 0.33799437 reg_l1 29.502007 reg_l2 12.52778
loss 3.2881951
cutoff 0.0056959675 network size 184
STEP 72 ================================
prereg loss 0.35135978 reg_l1 29.49192 reg_l2 12.52647
loss 3.300552
STEP 73 ================================
prereg loss 0.34991482 reg_l1 29.488409 reg_l2 12.525715
loss 3.2987556
STEP 74 ================================
prereg loss 0.34859827 reg_l1 29.485464 reg_l2 12.525255
loss 3.2971447
STEP 75 ================================
prereg loss 0.34729308 reg_l1 29.482723 reg_l2 12.524795
loss 3.2955656
STEP 76 ================================
prereg loss 0.3460201 reg_l1 29.479826 reg_l2 12.524112
loss 3.2940025
STEP 77 ================================
prereg loss 0.34544352 reg_l1 29.476469 reg_l2 12.5230875
loss 3.2930903
STEP 78 ================================
prereg loss 0.34548205 reg_l1 29.47279 reg_l2 12.521776
loss 3.292761
STEP 79 ================================
prereg loss 0.34551826 reg_l1 29.468933 reg_l2 12.520259
loss 3.2924118
STEP 80 ================================
prereg loss 0.3452768 reg_l1 29.464884 reg_l2 12.518614
loss 3.2917652
STEP 81 ================================
prereg loss 0.34496924 reg_l1 29.460604 reg_l2 12.516877
loss 3.2910297
cutoff 0.0055762455 network size 183
STEP 82 ================================
prereg loss 0.34480384 reg_l1 29.450418 reg_l2 12.515013
loss 3.2898457
STEP 83 ================================
prereg loss 0.34484676 reg_l1 29.44554 reg_l2 12.513072
loss 3.2894008
STEP 84 ================================
prereg loss 0.34462982 reg_l1 29.440443 reg_l2 12.51102
loss 3.288674
STEP 85 ================================
prereg loss 0.34426954 reg_l1 29.435268 reg_l2 12.508898
loss 3.2877965
STEP 86 ================================
prereg loss 0.34423268 reg_l1 29.430075 reg_l2 12.506814
loss 3.28724
STEP 87 ================================
prereg loss 0.34445456 reg_l1 29.424932 reg_l2 12.504853
loss 3.2869477
STEP 88 ================================
prereg loss 0.34490174 reg_l1 29.419981 reg_l2 12.503091
loss 3.2869
STEP 89 ================================
prereg loss 0.34585357 reg_l1 29.415304 reg_l2 12.50155
loss 3.287384
STEP 90 ================================
prereg loss 0.34593222 reg_l1 29.411125 reg_l2 12.500245
loss 3.2870448
STEP 91 ================================
prereg loss 0.3446336 reg_l1 29.4075 reg_l2 12.499165
loss 3.2853835
cutoff 0.006012612 network size 182
STEP 92 ================================
prereg loss 0.34380123 reg_l1 29.398266 reg_l2 12.49825
loss 3.283628
STEP 93 ================================
prereg loss 0.34351847 reg_l1 29.395205 reg_l2 12.497495
loss 3.283039
STEP 94 ================================
prereg loss 0.34359306 reg_l1 29.392225 reg_l2 12.496792
loss 3.2828157
STEP 95 ================================
prereg loss 0.3436907 reg_l1 29.389359 reg_l2 12.49603
loss 3.2826266
STEP 96 ================================
prereg loss 0.34337488 reg_l1 29.386606 reg_l2 12.495164
loss 3.2820356
STEP 97 ================================
prereg loss 0.34325466 reg_l1 29.38373 reg_l2 12.494172
loss 3.2816277
STEP 98 ================================
prereg loss 0.34345555 reg_l1 29.380575 reg_l2 12.493046
loss 3.2815132
STEP 99 ================================
prereg loss 0.34395003 reg_l1 29.377188 reg_l2 12.491836
loss 3.281669
STEP 100 ================================
prereg loss 0.3443928 reg_l1 29.373716 reg_l2 12.490567
loss 3.2817645
2022-08-14T18:36:37.223

julia> serialize("cf-s2-182-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-182-parameters-opt.ser", opt)
```

A longer run: 1 sparsification per 20 steps for 50 sparsifications and 1000 steps;
this ended up to be a nice, successful run:

```
julia> interleaving_steps!(1000, 20)
2022-08-14T18:46:27.795
STEP 1 ================================
prereg loss 0.34447548 reg_l1 29.370207 reg_l2 12.489225
loss 3.2814963
cutoff 0.005853093 network size 181
STEP 2 ================================
prereg loss 0.35573757 reg_l1 29.360811 reg_l2 12.487868
loss 3.2918186
STEP 3 ================================
prereg loss 0.35163394 reg_l1 29.356722 reg_l2 12.484972
loss 3.2873063
STEP 4 ================================
prereg loss 0.3542499 reg_l1 29.351664 reg_l2 12.481421
loss 3.2894163
STEP 5 ================================
prereg loss 0.35280856 reg_l1 29.345903 reg_l2 12.4781885
loss 3.2873988
STEP 6 ================================
prereg loss 0.356889 reg_l1 29.34076 reg_l2 12.475951
loss 3.290965
STEP 7 ================================
prereg loss 0.35653865 reg_l1 29.33737 reg_l2 12.47495
loss 3.2902756
STEP 8 ================================
prereg loss 0.35191151 reg_l1 29.335554 reg_l2 12.4749365
loss 3.285467
STEP 9 ================================
prereg loss 0.352936 reg_l1 29.333918 reg_l2 12.475349
loss 3.2863278
STEP 10 ================================
prereg loss 0.35181943 reg_l1 29.330898 reg_l2 12.475469
loss 3.2849095
STEP 11 ================================
prereg loss 0.35651204 reg_l1 29.326197 reg_l2 12.474816
loss 3.2891319
STEP 12 ================================
prereg loss 0.36087513 reg_l1 29.320665 reg_l2 12.473367
loss 3.2929418
STEP 13 ================================
prereg loss 0.35276458 reg_l1 29.31519 reg_l2 12.471614
loss 3.2842836
STEP 14 ================================
prereg loss 0.3501801 reg_l1 29.310074 reg_l2 12.470291
loss 3.2811875
STEP 15 ================================
prereg loss 0.35166726 reg_l1 29.305737 reg_l2 12.470047
loss 3.2822409
STEP 16 ================================
prereg loss 0.35018983 reg_l1 29.302471 reg_l2 12.471004
loss 3.280437
STEP 17 ================================
prereg loss 0.34850922 reg_l1 29.300276 reg_l2 12.472706
loss 3.278537
STEP 18 ================================
prereg loss 0.34765562 reg_l1 29.299322 reg_l2 12.474394
loss 3.277588
STEP 19 ================================
prereg loss 0.3474089 reg_l1 29.29803 reg_l2 12.475384
loss 3.2772121
STEP 20 ================================
prereg loss 0.34662732 reg_l1 29.29547 reg_l2 12.475431
loss 3.2761743
STEP 21 ================================
prereg loss 0.34571183 reg_l1 29.291739 reg_l2 12.474802
loss 3.2748857
cutoff 0.0001499923 network size 180
STEP 22 ================================
prereg loss 0.34673566 reg_l1 29.28797 reg_l2 12.474095
loss 3.2755327
STEP 23 ================================
prereg loss 0.347571 reg_l1 29.28544 reg_l2 12.473879
loss 3.276115
STEP 24 ================================
prereg loss 0.34655732 reg_l1 29.284054 reg_l2 12.47443
loss 3.274963
STEP 25 ================================
prereg loss 0.34485584 reg_l1 29.283304 reg_l2 12.475601
loss 3.2731862
STEP 26 ================================
prereg loss 0.34481356 reg_l1 29.282463 reg_l2 12.476979
loss 3.2730598
STEP 27 ================================
prereg loss 0.34629503 reg_l1 29.28112 reg_l2 12.478048
loss 3.2744071
STEP 28 ================================
prereg loss 0.34629187 reg_l1 29.279121 reg_l2 12.478559
loss 3.274204
STEP 29 ================================
prereg loss 0.34494466 reg_l1 29.276646 reg_l2 12.478615
loss 3.2726092
STEP 30 ================================
prereg loss 0.34461716 reg_l1 29.274054 reg_l2 12.478621
loss 3.2720225
STEP 31 ================================
prereg loss 0.3447721 reg_l1 29.271646 reg_l2 12.479005
loss 3.271937
STEP 32 ================================
prereg loss 0.34445307 reg_l1 29.269718 reg_l2 12.479998
loss 3.271425
STEP 33 ================================
prereg loss 0.3443117 reg_l1 29.268269 reg_l2 12.481516
loss 3.2711387
STEP 34 ================================
prereg loss 0.34387255 reg_l1 29.267124 reg_l2 12.483301
loss 3.270585
STEP 35 ================================
prereg loss 0.34248164 reg_l1 29.266018 reg_l2 12.485015
loss 3.2690835
STEP 36 ================================
prereg loss 0.34112278 reg_l1 29.264643 reg_l2 12.486492
loss 3.2675872
STEP 37 ================================
prereg loss 0.34015247 reg_l1 29.262968 reg_l2 12.487645
loss 3.2664495
STEP 38 ================================
prereg loss 0.33969456 reg_l1 29.261057 reg_l2 12.488623
loss 3.2658002
STEP 39 ================================
prereg loss 0.33943802 reg_l1 29.259142 reg_l2 12.489556
loss 3.2653522
STEP 40 ================================
prereg loss 0.33903673 reg_l1 29.257471 reg_l2 12.490567
loss 3.2647839
STEP 41 ================================
prereg loss 0.33824986 reg_l1 29.256025 reg_l2 12.491654
loss 3.2638526
cutoff 0.005646343 network size 179
STEP 42 ================================
prereg loss 0.34580514 reg_l1 29.248928 reg_l2 12.492719
loss 3.270698
STEP 43 ================================
prereg loss 0.3385356 reg_l1 29.248251 reg_l2 12.494246
loss 3.2633607
STEP 44 ================================
prereg loss 0.33871573 reg_l1 29.247143 reg_l2 12.495758
loss 3.26343
STEP 45 ================================
prereg loss 0.33743748 reg_l1 29.244967 reg_l2 12.49672
loss 3.261934
STEP 46 ================================
prereg loss 0.33787063 reg_l1 29.241844 reg_l2 12.496614
loss 3.2620552
STEP 47 ================================
prereg loss 0.3371891 reg_l1 29.237932 reg_l2 12.495315
loss 3.2609825
STEP 48 ================================
prereg loss 0.3361352 reg_l1 29.233484 reg_l2 12.493304
loss 3.2594836
STEP 49 ================================
prereg loss 0.33717445 reg_l1 29.22853 reg_l2 12.491381
loss 3.2600276
STEP 50 ================================
prereg loss 0.33849648 reg_l1 29.22403 reg_l2 12.49029
loss 3.2608995
STEP 51 ================================
prereg loss 0.33932963 reg_l1 29.220812 reg_l2 12.490326
loss 3.261411
STEP 52 ================================
prereg loss 0.33752635 reg_l1 29.219103 reg_l2 12.491304
loss 3.2594366
STEP 53 ================================
prereg loss 0.33503005 reg_l1 29.217972 reg_l2 12.492607
loss 3.2568274
STEP 54 ================================
prereg loss 0.33382308 reg_l1 29.2163 reg_l2 12.493506
loss 3.255453
STEP 55 ================================
prereg loss 0.3357142 reg_l1 29.213442 reg_l2 12.493433
loss 3.2570584
STEP 56 ================================
prereg loss 0.3344921 reg_l1 29.209589 reg_l2 12.492311
loss 3.2554512
STEP 57 ================================
prereg loss 0.33234784 reg_l1 29.205164 reg_l2 12.490608
loss 3.2528644
STEP 58 ================================
prereg loss 0.33446425 reg_l1 29.200645 reg_l2 12.489009
loss 3.254529
STEP 59 ================================
prereg loss 0.33507884 reg_l1 29.196482 reg_l2 12.487972
loss 3.254727
STEP 60 ================================
prereg loss 0.3335736 reg_l1 29.192936 reg_l2 12.487432
loss 3.2528672
STEP 61 ================================
prereg loss 0.33149773 reg_l1 29.189848 reg_l2 12.486966
loss 3.2504826
cutoff 0.0064092367 network size 178
STEP 62 ================================
prereg loss 0.33104557 reg_l1 29.179853 reg_l2 12.485998
loss 3.249031
STEP 63 ================================
prereg loss 0.331093 reg_l1 29.175045 reg_l2 12.484295
loss 3.2485976
STEP 64 ================================
prereg loss 0.33141482 reg_l1 29.169117 reg_l2 12.481857
loss 3.2483268
STEP 65 ================================
prereg loss 0.33121878 reg_l1 29.16303 reg_l2 12.479074
loss 3.2475219
STEP 66 ================================
prereg loss 0.3314725 reg_l1 29.15734 reg_l2 12.476475
loss 3.2472064
STEP 67 ================================
prereg loss 0.33123335 reg_l1 29.15233 reg_l2 12.474471
loss 3.2464664
STEP 68 ================================
prereg loss 0.33019027 reg_l1 29.147871 reg_l2 12.473101
loss 3.2449772
STEP 69 ================================
prereg loss 0.33071104 reg_l1 29.143663 reg_l2 12.472065
loss 3.2450776
STEP 70 ================================
prereg loss 0.3299645 reg_l1 29.139542 reg_l2 12.470919
loss 3.2439187
STEP 71 ================================
prereg loss 0.3284332 reg_l1 29.135061 reg_l2 12.469397
loss 3.2419395
STEP 72 ================================
prereg loss 0.32809776 reg_l1 29.129875 reg_l2 12.467493
loss 3.2410853
STEP 73 ================================
prereg loss 0.3283828 reg_l1 29.124338 reg_l2 12.465348
loss 3.2408166
STEP 74 ================================
prereg loss 0.32824463 reg_l1 29.11894 reg_l2 12.463151
loss 3.2401388
STEP 75 ================================
prereg loss 0.32784548 reg_l1 29.113863 reg_l2 12.4610815
loss 3.2392318
STEP 76 ================================
prereg loss 0.3272892 reg_l1 29.108843 reg_l2 12.459166
loss 3.2381735
STEP 77 ================================
prereg loss 0.32790032 reg_l1 29.103676 reg_l2 12.457286
loss 3.2382681
STEP 78 ================================
prereg loss 0.32651305 reg_l1 29.098728 reg_l2 12.455322
loss 3.2363858
STEP 79 ================================
prereg loss 0.32647976 reg_l1 29.093708 reg_l2 12.453196
loss 3.2358506
STEP 80 ================================
prereg loss 0.32601896 reg_l1 29.088142 reg_l2 12.45091
loss 3.2348332
STEP 81 ================================
prereg loss 0.3268657 reg_l1 29.082561 reg_l2 12.448518
loss 3.235122
cutoff 0.0060465815 network size 177
STEP 82 ================================
prereg loss 0.32561284 reg_l1 29.071278 reg_l2 12.446056
loss 3.2327406
STEP 83 ================================
prereg loss 0.32505903 reg_l1 29.066286 reg_l2 12.443733
loss 3.2316875
STEP 84 ================================
prereg loss 0.32522446 reg_l1 29.060968 reg_l2 12.4416065
loss 3.2313213
STEP 85 ================================
prereg loss 0.32558504 reg_l1 29.055967 reg_l2 12.439664
loss 3.2311819
STEP 86 ================================
prereg loss 0.32425502 reg_l1 29.051367 reg_l2 12.43785
loss 3.2293918
STEP 87 ================================
prereg loss 0.32375824 reg_l1 29.04684 reg_l2 12.436098
loss 3.2284422
STEP 88 ================================
prereg loss 0.3238414 reg_l1 29.041922 reg_l2 12.434301
loss 3.2280335
STEP 89 ================================
prereg loss 0.32422253 reg_l1 29.036995 reg_l2 12.432362
loss 3.2279222
STEP 90 ================================
prereg loss 0.32286194 reg_l1 29.032084 reg_l2 12.4302435
loss 3.2260704
STEP 91 ================================
prereg loss 0.3224391 reg_l1 29.026917 reg_l2 12.428069
loss 3.2251308
STEP 92 ================================
prereg loss 0.3224708 reg_l1 29.021564 reg_l2 12.426002
loss 3.2246275
STEP 93 ================================
prereg loss 0.32287714 reg_l1 29.016308 reg_l2 12.424097
loss 3.224508
STEP 94 ================================
prereg loss 0.3223786 reg_l1 29.011368 reg_l2 12.422341
loss 3.2235155
STEP 95 ================================
prereg loss 0.32145247 reg_l1 29.006685 reg_l2 12.420685
loss 3.222121
STEP 96 ================================
prereg loss 0.32103753 reg_l1 29.0019 reg_l2 12.419012
loss 3.2212276
STEP 97 ================================
prereg loss 0.32110983 reg_l1 28.99683 reg_l2 12.417227
loss 3.2207928
STEP 98 ================================
prereg loss 0.32142824 reg_l1 28.991602 reg_l2 12.415302
loss 3.2205884
STEP 99 ================================
prereg loss 0.3205337 reg_l1 28.986536 reg_l2 12.413308
loss 3.2191875
STEP 100 ================================
prereg loss 0.32038558 reg_l1 28.981522 reg_l2 12.41137
loss 3.2185378
STEP 101 ================================
prereg loss 0.320449 reg_l1 28.97634 reg_l2 12.40955
loss 3.218083
cutoff 3.2822078e-5 network size 176
STEP 102 ================================
prereg loss 0.32082745 reg_l1 28.97136 reg_l2 12.407855
loss 3.2179635
STEP 103 ================================
prereg loss 0.31934863 reg_l1 28.967592 reg_l2 12.406228
loss 3.2161078
STEP 104 ================================
prereg loss 0.31908464 reg_l1 28.96369 reg_l2 12.404617
loss 3.2154536
STEP 105 ================================
prereg loss 0.3197381 reg_l1 28.959328 reg_l2 12.402945
loss 3.215671
STEP 106 ================================
prereg loss 0.31947485 reg_l1 28.955156 reg_l2 12.401201
loss 3.2149906
STEP 107 ================================
prereg loss 0.3183907 reg_l1 28.951206 reg_l2 12.399444
loss 3.2135112
STEP 108 ================================
prereg loss 0.3180567 reg_l1 28.946943 reg_l2 12.397695
loss 3.212751
STEP 109 ================================
prereg loss 0.31847575 reg_l1 28.942707 reg_l2 12.396033
loss 3.2127464
STEP 110 ================================
prereg loss 0.31771287 reg_l1 28.938755 reg_l2 12.394436
loss 3.2115884
STEP 111 ================================
prereg loss 0.3171796 reg_l1 28.934828 reg_l2 12.392914
loss 3.2106626
STEP 112 ================================
prereg loss 0.3175953 reg_l1 28.93065 reg_l2 12.391377
loss 3.2106605
STEP 113 ================================
prereg loss 0.31701273 reg_l1 28.926655 reg_l2 12.389843
loss 3.2096784
STEP 114 ================================
prereg loss 0.3165177 reg_l1 28.92262 reg_l2 12.388261
loss 3.2087798
STEP 115 ================================
prereg loss 0.3163321 reg_l1 28.91847 reg_l2 12.386672
loss 3.2081792
STEP 116 ================================
prereg loss 0.31628984 reg_l1 28.914324 reg_l2 12.385124
loss 3.2077224
STEP 117 ================================
prereg loss 0.3161394 reg_l1 28.910254 reg_l2 12.383619
loss 3.2071648
STEP 118 ================================
prereg loss 0.31572363 reg_l1 28.906284 reg_l2 12.382149
loss 3.2063522
STEP 119 ================================
prereg loss 0.31538346 reg_l1 28.902275 reg_l2 12.380701
loss 3.205611
STEP 120 ================================
prereg loss 0.3151426 reg_l1 28.898306 reg_l2 12.379288
loss 3.2049732
STEP 121 ================================
prereg loss 0.31496066 reg_l1 28.894344 reg_l2 12.3778925
loss 3.2043953
cutoff 0.0052190316 network size 175
STEP 122 ================================
prereg loss 0.31517923 reg_l1 28.88506 reg_l2 12.376421
loss 3.2036853
STEP 123 ================================
prereg loss 0.31440482 reg_l1 28.881245 reg_l2 12.374969
loss 3.2025292
STEP 124 ================================
prereg loss 0.3146462 reg_l1 28.877071 reg_l2 12.373444
loss 3.2023535
STEP 125 ================================
prereg loss 0.31414598 reg_l1 28.873034 reg_l2 12.371931
loss 3.2014494
STEP 126 ================================
prereg loss 0.3138537 reg_l1 28.869085 reg_l2 12.370445
loss 3.2007623
STEP 127 ================================
prereg loss 0.31415698 reg_l1 28.864824 reg_l2 12.368999
loss 3.2006395
STEP 128 ================================
prereg loss 0.3137581 reg_l1 28.860847 reg_l2 12.367608
loss 3.199843
STEP 129 ================================
prereg loss 0.3130948 reg_l1 28.857063 reg_l2 12.36626
loss 3.1988013
STEP 130 ================================
prereg loss 0.3131998 reg_l1 28.852873 reg_l2 12.36488
loss 3.198487
STEP 131 ================================
prereg loss 0.31370968 reg_l1 28.848658 reg_l2 12.3634615
loss 3.1985755
STEP 132 ================================
prereg loss 0.3128482 reg_l1 28.84466 reg_l2 12.362041
loss 3.1973143
STEP 133 ================================
prereg loss 0.31226486 reg_l1 28.84069 reg_l2 12.36062
loss 3.1963341
STEP 134 ================================
prereg loss 0.31211266 reg_l1 28.836597 reg_l2 12.359244
loss 3.1957724
STEP 135 ================================
prereg loss 0.31237793 reg_l1 28.832462 reg_l2 12.357912
loss 3.195624
STEP 136 ================================
prereg loss 0.3123717 reg_l1 28.828438 reg_l2 12.356577
loss 3.1952155
STEP 137 ================================
prereg loss 0.31163713 reg_l1 28.824509 reg_l2 12.355224
loss 3.194088
STEP 138 ================================
prereg loss 0.31122345 reg_l1 28.82052 reg_l2 12.35384
loss 3.1932757
STEP 139 ================================
prereg loss 0.31112236 reg_l1 28.81645 reg_l2 12.352488
loss 3.1927674
STEP 140 ================================
prereg loss 0.3111827 reg_l1 28.812374 reg_l2 12.351143
loss 3.1924202
STEP 141 ================================
prereg loss 0.31091094 reg_l1 28.808416 reg_l2 12.3497925
loss 3.1917527
cutoff 0.0041809706 network size 174
STEP 142 ================================
prereg loss 0.31049848 reg_l1 28.800316 reg_l2 12.348389
loss 3.19053
STEP 143 ================================
prereg loss 0.3109602 reg_l1 28.7961 reg_l2 12.34692
loss 3.1905704
STEP 144 ================================
prereg loss 0.31017876 reg_l1 28.792107 reg_l2 12.34542
loss 3.1893895
STEP 145 ================================
prereg loss 0.31001619 reg_l1 28.787952 reg_l2 12.343875
loss 3.1888115
STEP 146 ================================
prereg loss 0.31093732 reg_l1 28.783556 reg_l2 12.342303
loss 3.1892931
STEP 147 ================================
prereg loss 0.31047016 reg_l1 28.779432 reg_l2 12.340766
loss 3.1884134
STEP 148 ================================
prereg loss 0.30950582 reg_l1 28.775446 reg_l2 12.339271
loss 3.1870503
STEP 149 ================================
prereg loss 0.30928758 reg_l1 28.771336 reg_l2 12.337822
loss 3.1864212
STEP 150 ================================
prereg loss 0.3095765 reg_l1 28.767126 reg_l2 12.336417
loss 3.186289
STEP 151 ================================
prereg loss 0.30928433 reg_l1 28.763075 reg_l2 12.335045
loss 3.1855917
STEP 152 ================================
prereg loss 0.3087336 reg_l1 28.759209 reg_l2 12.333719
loss 3.1846547
STEP 153 ================================
prereg loss 0.3089314 reg_l1 28.754992 reg_l2 12.33237
loss 3.1844306
STEP 154 ================================
prereg loss 0.30905294 reg_l1 28.75085 reg_l2 12.331012
loss 3.184138
STEP 155 ================================
prereg loss 0.3085361 reg_l1 28.746807 reg_l2 12.329642
loss 3.1832168
STEP 156 ================================
prereg loss 0.30792248 reg_l1 28.742767 reg_l2 12.328224
loss 3.1821995
STEP 157 ================================
prereg loss 0.3082611 reg_l1 28.738302 reg_l2 12.326699
loss 3.1820915
STEP 158 ================================
prereg loss 0.3084885 reg_l1 28.733852 reg_l2 12.325112
loss 3.1818738
STEP 159 ================================
prereg loss 0.30781296 reg_l1 28.729572 reg_l2 12.323517
loss 3.1807702
STEP 160 ================================
prereg loss 0.30744678 reg_l1 28.72535 reg_l2 12.321967
loss 3.1799817
STEP 161 ================================
prereg loss 0.30738392 reg_l1 28.721092 reg_l2 12.3204975
loss 3.1794932
cutoff 0.005151277 network size 173
STEP 162 ================================
prereg loss 0.3075875 reg_l1 28.711746 reg_l2 12.319073
loss 3.178762
STEP 163 ================================
prereg loss 0.3072331 reg_l1 28.707846 reg_l2 12.317729
loss 3.1780176
STEP 164 ================================
prereg loss 0.3067376 reg_l1 28.704145 reg_l2 12.316382
loss 3.1771522
STEP 165 ================================
prereg loss 0.30660328 reg_l1 28.70008 reg_l2 12.314998
loss 3.1766114
STEP 166 ================================
prereg loss 0.30698854 reg_l1 28.695932 reg_l2 12.313539
loss 3.1765819
STEP 167 ================================
prereg loss 0.30622533 reg_l1 28.691952 reg_l2 12.312029
loss 3.1754205
STEP 168 ================================
prereg loss 0.30603394 reg_l1 28.687923 reg_l2 12.310496
loss 3.1748261
STEP 169 ================================
prereg loss 0.30619568 reg_l1 28.683498 reg_l2 12.308921
loss 3.1745455
STEP 170 ================================
prereg loss 0.30694285 reg_l1 28.679102 reg_l2 12.307338
loss 3.174853
STEP 171 ================================
prereg loss 0.30569413 reg_l1 28.674961 reg_l2 12.305738
loss 3.1731904
STEP 172 ================================
prereg loss 0.30531427 reg_l1 28.670698 reg_l2 12.304142
loss 3.1723843
STEP 173 ================================
prereg loss 0.3059439 reg_l1 28.666157 reg_l2 12.302525
loss 3.1725597
STEP 174 ================================
prereg loss 0.30596268 reg_l1 28.66177 reg_l2 12.300916
loss 3.1721396
STEP 175 ================================
prereg loss 0.3051163 reg_l1 28.657616 reg_l2 12.299336
loss 3.170878
STEP 176 ================================
prereg loss 0.3049038 reg_l1 28.65339 reg_l2 12.297796
loss 3.1702428
STEP 177 ================================
prereg loss 0.30508998 reg_l1 28.649126 reg_l2 12.296325
loss 3.1700027
STEP 178 ================================
prereg loss 0.30481458 reg_l1 28.645016 reg_l2 12.294885
loss 3.1693163
STEP 179 ================================
prereg loss 0.3041826 reg_l1 28.641054 reg_l2 12.293464
loss 3.168288
STEP 180 ================================
prereg loss 0.3039858 reg_l1 28.63698 reg_l2 12.292033
loss 3.1676838
STEP 181 ================================
prereg loss 0.30448782 reg_l1 28.632547 reg_l2 12.290529
loss 3.1677427
cutoff 0.0053832512 network size 172
STEP 182 ================================
prereg loss 0.30420068 reg_l1 28.622921 reg_l2 12.288968
loss 3.1664927
STEP 183 ================================
prereg loss 0.30354097 reg_l1 28.618721 reg_l2 12.28738
loss 3.1654131
STEP 184 ================================
prereg loss 0.30334663 reg_l1 28.614416 reg_l2 12.285747
loss 3.1647882
STEP 185 ================================
prereg loss 0.30363256 reg_l1 28.60973 reg_l2 12.284066
loss 3.1646056
STEP 186 ================================
prereg loss 0.3043035 reg_l1 28.605118 reg_l2 12.282363
loss 3.1648152
STEP 187 ================================
prereg loss 0.3031757 reg_l1 28.600758 reg_l2 12.2806425
loss 3.1632514
STEP 188 ================================
prereg loss 0.30281496 reg_l1 28.596384 reg_l2 12.278935
loss 3.1624534
STEP 189 ================================
prereg loss 0.3027713 reg_l1 28.5919 reg_l2 12.277282
loss 3.1619613
STEP 190 ================================
prereg loss 0.30333495 reg_l1 28.587362 reg_l2 12.275679
loss 3.1620712
STEP 191 ================================
prereg loss 0.30297658 reg_l1 28.583105 reg_l2 12.274114
loss 3.161287
STEP 192 ================================
prereg loss 0.30215666 reg_l1 28.579035 reg_l2 12.2725725
loss 3.1600602
STEP 193 ================================
prereg loss 0.30208188 reg_l1 28.574846 reg_l2 12.271057
loss 3.1595664
STEP 194 ================================
prereg loss 0.3026922 reg_l1 28.570293 reg_l2 12.269522
loss 3.1597216
STEP 195 ================================
prereg loss 0.30261663 reg_l1 28.56599 reg_l2 12.267976
loss 3.1592157
STEP 196 ================================
prereg loss 0.30149874 reg_l1 28.561878 reg_l2 12.266411
loss 3.1576865
STEP 197 ================================
prereg loss 0.30143827 reg_l1 28.557362 reg_l2 12.264775
loss 3.1571746
STEP 198 ================================
prereg loss 0.3019164 reg_l1 28.552702 reg_l2 12.26313
loss 3.1571865
STEP 199 ================================
prereg loss 0.30210355 reg_l1 28.548124 reg_l2 12.261474
loss 3.156916
STEP 200 ================================
prereg loss 0.30116248 reg_l1 28.543777 reg_l2 12.259815
loss 3.1555402
STEP 201 ================================
prereg loss 0.30095616 reg_l1 28.539371 reg_l2 12.258208
loss 3.1548934
cutoff 0.0051365453 network size 171
STEP 202 ================================
prereg loss 0.3011603 reg_l1 28.529709 reg_l2 12.256634
loss 3.1541312
STEP 203 ================================
prereg loss 0.30185843 reg_l1 28.525234 reg_l2 12.255104
loss 3.154382
STEP 204 ================================
prereg loss 0.30081204 reg_l1 28.521088 reg_l2 12.253581
loss 3.1529207
STEP 205 ================================
prereg loss 0.30035296 reg_l1 28.51694 reg_l2 12.252079
loss 3.152047
STEP 206 ================================
prereg loss 0.3002953 reg_l1 28.51268 reg_l2 12.250631
loss 3.1515634
STEP 207 ================================
prereg loss 0.3007433 reg_l1 28.508348 reg_l2 12.249199
loss 3.1515782
STEP 208 ================================
prereg loss 0.30013335 reg_l1 28.504236 reg_l2 12.24776
loss 3.150557
STEP 209 ================================
prereg loss 0.29973415 reg_l1 28.500069 reg_l2 12.246305
loss 3.149741
STEP 210 ================================
prereg loss 0.29962918 reg_l1 28.4958 reg_l2 12.244857
loss 3.1492093
STEP 211 ================================
prereg loss 0.29976127 reg_l1 28.491488 reg_l2 12.243408
loss 3.14891
STEP 212 ================================
prereg loss 0.2992903 reg_l1 28.487373 reg_l2 12.241993
loss 3.148028
STEP 213 ================================
prereg loss 0.29984006 reg_l1 28.483 reg_l2 12.240524
loss 3.14814
STEP 214 ================================
prereg loss 0.29929614 reg_l1 28.478775 reg_l2 12.239037
loss 3.1471736
STEP 215 ================================
prereg loss 0.2989989 reg_l1 28.474573 reg_l2 12.237543
loss 3.1464562
STEP 216 ================================
prereg loss 0.29940072 reg_l1 28.470047 reg_l2 12.236043
loss 3.1464055
STEP 217 ================================
prereg loss 0.29935026 reg_l1 28.465767 reg_l2 12.234591
loss 3.145927
STEP 218 ================================
prereg loss 0.2985524 reg_l1 28.461668 reg_l2 12.233168
loss 3.1447191
STEP 219 ================================
prereg loss 0.29841486 reg_l1 28.45744 reg_l2 12.23179
loss 3.1441588
STEP 220 ================================
prereg loss 0.29881477 reg_l1 28.453081 reg_l2 12.230442
loss 3.1441228
STEP 221 ================================
prereg loss 0.29853505 reg_l1 28.448887 reg_l2 12.229103
loss 3.1434238
cutoff 0.0057304883 network size 170
STEP 222 ================================
prereg loss 0.2981441 reg_l1 28.439007 reg_l2 12.22773
loss 3.1420448
STEP 223 ================================
prereg loss 0.29790226 reg_l1 28.434847 reg_l2 12.226411
loss 3.141387
STEP 224 ================================
prereg loss 0.29787692 reg_l1 28.43064 reg_l2 12.225094
loss 3.140941
STEP 225 ================================
prereg loss 0.2979205 reg_l1 28.4264 reg_l2 12.223761
loss 3.1405604
STEP 226 ================================
prereg loss 0.29780895 reg_l1 28.422106 reg_l2 12.222371
loss 3.1400194
STEP 227 ================================
prereg loss 0.29730186 reg_l1 28.41799 reg_l2 12.2209835
loss 3.1391008
STEP 228 ================================
prereg loss 0.29775158 reg_l1 28.413507 reg_l2 12.219561
loss 3.1391025
STEP 229 ================================
prereg loss 0.2974935 reg_l1 28.409271 reg_l2 12.218178
loss 3.1384206
STEP 230 ================================
prereg loss 0.2969054 reg_l1 28.405191 reg_l2 12.216825
loss 3.1374245
STEP 231 ================================
prereg loss 0.29686153 reg_l1 28.400925 reg_l2 12.215491
loss 3.1369538
STEP 232 ================================
prereg loss 0.2974008 reg_l1 28.396542 reg_l2 12.214175
loss 3.137055
STEP 233 ================================
prereg loss 0.29687142 reg_l1 28.392422 reg_l2 12.2128725
loss 3.1361136
STEP 234 ================================
prereg loss 0.2964261 reg_l1 28.388327 reg_l2 12.211575
loss 3.1352587
STEP 235 ================================
prereg loss 0.29637757 reg_l1 28.384085 reg_l2 12.210266
loss 3.1347861
STEP 236 ================================
prereg loss 0.29658502 reg_l1 28.37978 reg_l2 12.208957
loss 3.1345632
STEP 237 ================================
prereg loss 0.29616517 reg_l1 28.375605 reg_l2 12.207632
loss 3.1337256
STEP 238 ================================
prereg loss 0.2959361 reg_l1 28.371395 reg_l2 12.206318
loss 3.1330757
STEP 239 ================================
prereg loss 0.29593268 reg_l1 28.36711 reg_l2 12.205002
loss 3.1326437
STEP 240 ================================
prereg loss 0.29609412 reg_l1 28.362785 reg_l2 12.2036705
loss 3.1323729
STEP 241 ================================
prereg loss 0.29553667 reg_l1 28.358576 reg_l2 12.202325
loss 3.1313944
cutoff 0.0032590458 network size 169
STEP 242 ================================
prereg loss 0.2964976 reg_l1 28.350744 reg_l2 12.20088
loss 3.131572
STEP 243 ================================
prereg loss 0.29571408 reg_l1 28.346785 reg_l2 12.199596
loss 3.1303926
STEP 244 ================================
prereg loss 0.29524788 reg_l1 28.34295 reg_l2 12.198394
loss 3.1295428
STEP 245 ================================
prereg loss 0.2960379 reg_l1 28.33858 reg_l2 12.197098
loss 3.129896
STEP 246 ================================
prereg loss 0.2962509 reg_l1 28.334253 reg_l2 12.19569
loss 3.129676
STEP 247 ================================
prereg loss 0.29496717 reg_l1 28.330124 reg_l2 12.194192
loss 3.1279795
STEP 248 ================================
prereg loss 0.29488915 reg_l1 28.32587 reg_l2 12.19272
loss 3.1274762
STEP 249 ================================
prereg loss 0.29601657 reg_l1 28.32132 reg_l2 12.191283
loss 3.1281486
STEP 250 ================================
prereg loss 0.29581428 reg_l1 28.317106 reg_l2 12.189909
loss 3.1275249
STEP 251 ================================
prereg loss 0.2945013 reg_l1 28.31316 reg_l2 12.188574
loss 3.1258173
STEP 252 ================================
prereg loss 0.2943995 reg_l1 28.309004 reg_l2 12.187296
loss 3.1253
STEP 253 ================================
prereg loss 0.2947939 reg_l1 28.304676 reg_l2 12.186047
loss 3.1252615
STEP 254 ================================
prereg loss 0.29530507 reg_l1 28.300404 reg_l2 12.184759
loss 3.1253455
STEP 255 ================================
prereg loss 0.29448852 reg_l1 28.296257 reg_l2 12.183358
loss 3.1241143
STEP 256 ================================
prereg loss 0.2939761 reg_l1 28.291983 reg_l2 12.18187
loss 3.1231744
STEP 257 ================================
prereg loss 0.2940366 reg_l1 28.28747 reg_l2 12.180361
loss 3.1227837
STEP 258 ================================
prereg loss 0.29448545 reg_l1 28.282991 reg_l2 12.178928
loss 3.1227846
STEP 259 ================================
prereg loss 0.29438624 reg_l1 28.278734 reg_l2 12.177574
loss 3.1222596
STEP 260 ================================
prereg loss 0.29378042 reg_l1 28.274672 reg_l2 12.176271
loss 3.1212475
STEP 261 ================================
prereg loss 0.29354918 reg_l1 28.270548 reg_l2 12.174971
loss 3.120604
cutoff 0.0057440535 network size 168
STEP 262 ================================
prereg loss 0.29378754 reg_l1 28.260466 reg_l2 12.173597
loss 3.1198342
STEP 263 ================================
prereg loss 0.29365453 reg_l1 28.256145 reg_l2 12.172205
loss 3.1192691
STEP 264 ================================
prereg loss 0.29340932 reg_l1 28.251928 reg_l2 12.17083
loss 3.1186023
STEP 265 ================================
prereg loss 0.29315817 reg_l1 28.247753 reg_l2 12.169514
loss 3.1179338
STEP 266 ================================
prereg loss 0.29308397 reg_l1 28.243603 reg_l2 12.168254
loss 3.1174443
STEP 267 ================================
prereg loss 0.2933363 reg_l1 28.239374 reg_l2 12.166988
loss 3.1172738
STEP 268 ================================
prereg loss 0.29267904 reg_l1 28.235224 reg_l2 12.165699
loss 3.1162014
STEP 269 ================================
prereg loss 0.29262933 reg_l1 28.231012 reg_l2 12.164418
loss 3.1157305
STEP 270 ================================
prereg loss 0.293664 reg_l1 28.226524 reg_l2 12.163089
loss 3.1163166
STEP 271 ================================
prereg loss 0.2927534 reg_l1 28.222303 reg_l2 12.161723
loss 3.1149838
STEP 272 ================================
prereg loss 0.2924042 reg_l1 28.218138 reg_l2 12.160331
loss 3.114218
STEP 273 ================================
prereg loss 0.29275334 reg_l1 28.213432 reg_l2 12.158902
loss 3.1140966
STEP 274 ================================
prereg loss 0.29316324 reg_l1 28.20889 reg_l2 12.157493
loss 3.1140525
STEP 275 ================================
prereg loss 0.2919909 reg_l1 28.204666 reg_l2 12.1561365
loss 3.1124578
STEP 276 ================================
prereg loss 0.29183632 reg_l1 28.20039 reg_l2 12.154856
loss 3.1118753
STEP 277 ================================
prereg loss 0.2933269 reg_l1 28.19573 reg_l2 12.153566
loss 3.1129
STEP 278 ================================
prereg loss 0.29271254 reg_l1 28.191387 reg_l2 12.152253
loss 3.1118512
STEP 279 ================================
prereg loss 0.29146427 reg_l1 28.187214 reg_l2 12.150898
loss 3.1101859
STEP 280 ================================
prereg loss 0.29143456 reg_l1 28.182755 reg_l2 12.149565
loss 3.10971
STEP 281 ================================
prereg loss 0.2919684 reg_l1 28.178198 reg_l2 12.148293
loss 3.1097882
cutoff 0.005282697 network size 167
STEP 282 ================================
prereg loss 0.29248688 reg_l1 28.168547 reg_l2 12.147013
loss 3.1093416
STEP 283 ================================
prereg loss 0.2910231 reg_l1 28.16452 reg_l2 12.1457405
loss 3.107475
STEP 284 ================================
prereg loss 0.29077646 reg_l1 28.160282 reg_l2 12.144453
loss 3.1068048
STEP 285 ================================
prereg loss 0.29111648 reg_l1 28.155771 reg_l2 12.143177
loss 3.1066937
STEP 286 ================================
prereg loss 0.29168847 reg_l1 28.151354 reg_l2 12.1418915
loss 3.106824
STEP 287 ================================
prereg loss 0.29087994 reg_l1 28.147156 reg_l2 12.140575
loss 3.1055956
STEP 288 ================================
prereg loss 0.2902927 reg_l1 28.142965 reg_l2 12.139213
loss 3.1045892
STEP 289 ================================
prereg loss 0.2902854 reg_l1 28.138535 reg_l2 12.137832
loss 3.1041389
STEP 290 ================================
prereg loss 0.29095954 reg_l1 28.133902 reg_l2 12.136418
loss 3.1043499
STEP 291 ================================
prereg loss 0.29093397 reg_l1 28.129429 reg_l2 12.134985
loss 3.103877
STEP 292 ================================
prereg loss 0.28993905 reg_l1 28.12527 reg_l2 12.133599
loss 3.102466
STEP 293 ================================
prereg loss 0.28972855 reg_l1 28.121014 reg_l2 12.132303
loss 3.10183
STEP 294 ================================
prereg loss 0.29155636 reg_l1 28.116362 reg_l2 12.131003
loss 3.1031926
STEP 295 ================================
prereg loss 0.29092717 reg_l1 28.112026 reg_l2 12.129652
loss 3.10213
STEP 296 ================================
prereg loss 0.28937456 reg_l1 28.107819 reg_l2 12.128229
loss 3.1001565
STEP 297 ================================
prereg loss 0.28946418 reg_l1 28.103312 reg_l2 12.12684
loss 3.0997953
STEP 298 ================================
prereg loss 0.2911578 reg_l1 28.098463 reg_l2 12.125487
loss 3.1010041
STEP 299 ================================
prereg loss 0.29048508 reg_l1 28.094093 reg_l2 12.124157
loss 3.0998945
STEP 300 ================================
prereg loss 0.28893203 reg_l1 28.08998 reg_l2 12.12281
loss 3.09793
STEP 301 ================================
prereg loss 0.28881878 reg_l1 28.085476 reg_l2 12.121508
loss 3.0973666
cutoff 0.005838918 network size 166
STEP 302 ================================
prereg loss 0.28978613 reg_l1 28.07483 reg_l2 12.12019
loss 3.097269
STEP 303 ================================
prereg loss 0.29042667 reg_l1 28.07029 reg_l2 12.118901
loss 3.0974557
STEP 304 ================================
prereg loss 0.28851196 reg_l1 28.066196 reg_l2 12.117597
loss 3.0951316
STEP 305 ================================
prereg loss 0.28849152 reg_l1 28.061928 reg_l2 12.11632
loss 3.0946844
STEP 306 ================================
prereg loss 0.28958073 reg_l1 28.05703 reg_l2 12.114991
loss 3.0952837
STEP 307 ================================
prereg loss 0.29111004 reg_l1 28.052284 reg_l2 12.113586
loss 3.0963385
STEP 308 ================================
prereg loss 0.28828207 reg_l1 28.048014 reg_l2 12.112117
loss 3.0930836
STEP 309 ================================
prereg loss 0.28853786 reg_l1 28.043604 reg_l2 12.110718
loss 3.0928984
STEP 310 ================================
prereg loss 0.28910026 reg_l1 28.038654 reg_l2 12.109373
loss 3.0929656
STEP 311 ================================
prereg loss 0.29141295 reg_l1 28.033907 reg_l2 12.108039
loss 3.0948038
STEP 312 ================================
prereg loss 0.2881938 reg_l1 28.029709 reg_l2 12.106679
loss 3.0911646
STEP 313 ================================
prereg loss 0.28806254 reg_l1 28.025341 reg_l2 12.105331
loss 3.0905967
STEP 314 ================================
prereg loss 0.28814542 reg_l1 28.020582 reg_l2 12.104075
loss 3.0902038
STEP 315 ================================
prereg loss 0.28978786 reg_l1 28.015902 reg_l2 12.1029005
loss 3.091378
STEP 316 ================================
prereg loss 0.28809023 reg_l1 28.011747 reg_l2 12.101747
loss 3.089265
STEP 317 ================================
prereg loss 0.28737897 reg_l1 28.007658 reg_l2 12.100608
loss 3.0881448
STEP 318 ================================
prereg loss 0.28734735 reg_l1 28.003185 reg_l2 12.0994835
loss 3.0876658
STEP 319 ================================
prereg loss 0.28845266 reg_l1 27.998547 reg_l2 12.098352
loss 3.0883074
STEP 320 ================================
prereg loss 0.2881765 reg_l1 27.994091 reg_l2 12.097152
loss 3.0875857
STEP 321 ================================
prereg loss 0.28701228 reg_l1 27.989786 reg_l2 12.095908
loss 3.085991
cutoff 0.006398685 network size 165
STEP 322 ================================
prereg loss 0.28682473 reg_l1 27.978947 reg_l2 12.09464
loss 3.0847194
STEP 323 ================================
prereg loss 0.2872917 reg_l1 27.974237 reg_l2 12.093427
loss 3.0847156
STEP 324 ================================
prereg loss 0.28800693 reg_l1 27.969595 reg_l2 12.092201
loss 3.0849667
STEP 325 ================================
prereg loss 0.287088 reg_l1 27.965178 reg_l2 12.090948
loss 3.0836058
STEP 326 ================================
prereg loss 0.2865537 reg_l1 27.960812 reg_l2 12.089707
loss 3.082635
STEP 327 ================================
prereg loss 0.28663847 reg_l1 27.95623 reg_l2 12.088497
loss 3.0822616
STEP 328 ================================
prereg loss 0.28737578 reg_l1 27.951576 reg_l2 12.087318
loss 3.0825334
STEP 329 ================================
prereg loss 0.28715572 reg_l1 27.94715 reg_l2 12.086146
loss 3.0818706
STEP 330 ================================
prereg loss 0.28617477 reg_l1 27.942976 reg_l2 12.085014
loss 3.0804725
STEP 331 ================================
prereg loss 0.28607774 reg_l1 27.938568 reg_l2 12.083912
loss 3.0799346
STEP 332 ================================
prereg loss 0.28693882 reg_l1 27.93394 reg_l2 12.082844
loss 3.080333
STEP 333 ================================
prereg loss 0.28642297 reg_l1 27.929619 reg_l2 12.081788
loss 3.0793848
STEP 334 ================================
prereg loss 0.28573975 reg_l1 27.92537 reg_l2 12.08072
loss 3.0782766
STEP 335 ================================
prereg loss 0.28560567 reg_l1 27.920979 reg_l2 12.079655
loss 3.0777035
STEP 336 ================================
prereg loss 0.28605798 reg_l1 27.916357 reg_l2 12.078549
loss 3.0776937
STEP 337 ================================
prereg loss 0.2862817 reg_l1 27.911762 reg_l2 12.077401
loss 3.077458
STEP 338 ================================
prereg loss 0.28555024 reg_l1 27.907364 reg_l2 12.076218
loss 3.0762868
STEP 339 ================================
prereg loss 0.28562415 reg_l1 27.902618 reg_l2 12.075013
loss 3.0758862
STEP 340 ================================
prereg loss 0.28654277 reg_l1 27.897804 reg_l2 12.073722
loss 3.076323
STEP 341 ================================
prereg loss 0.28553107 reg_l1 27.893261 reg_l2 12.07243
loss 3.0748572
cutoff 0.0067549027 network size 164
STEP 342 ================================
prereg loss 0.28551704 reg_l1 27.881924 reg_l2 12.071198
loss 3.0737095
STEP 343 ================================
prereg loss 0.2860308 reg_l1 27.877487 reg_l2 12.070221
loss 3.0737796
STEP 344 ================================
prereg loss 0.28595945 reg_l1 27.873373 reg_l2 12.069443
loss 3.0732968
STEP 345 ================================
prereg loss 0.28548324 reg_l1 27.869493 reg_l2 12.068779
loss 3.0724325
STEP 346 ================================
prereg loss 0.28558058 reg_l1 27.865501 reg_l2 12.06818
loss 3.072131
STEP 347 ================================
prereg loss 0.28628302 reg_l1 27.861433 reg_l2 12.067588
loss 3.0724263
STEP 348 ================================
prereg loss 0.28560024 reg_l1 27.85751 reg_l2 12.066988
loss 3.0713513
STEP 349 ================================
prereg loss 0.28551534 reg_l1 27.853483 reg_l2 12.066417
loss 3.0708637
STEP 350 ================================
prereg loss 0.2857461 reg_l1 27.84949 reg_l2 12.065983
loss 3.0706952
STEP 351 ================================
prereg loss 0.28589094 reg_l1 27.845613 reg_l2 12.065678
loss 3.0704522
STEP 352 ================================
prereg loss 0.28568017 reg_l1 27.841908 reg_l2 12.065498
loss 3.069871
STEP 353 ================================
prereg loss 0.28546545 reg_l1 27.838253 reg_l2 12.065406
loss 3.0692909
STEP 354 ================================
prereg loss 0.28550282 reg_l1 27.834589 reg_l2 12.065398
loss 3.0689619
STEP 355 ================================
prereg loss 0.28564397 reg_l1 27.830969 reg_l2 12.065465
loss 3.068741
STEP 356 ================================
prereg loss 0.28541714 reg_l1 27.827559 reg_l2 12.065627
loss 3.068173
STEP 357 ================================
prereg loss 0.28682595 reg_l1 27.82402 reg_l2 12.065862
loss 3.069228
STEP 358 ================================
prereg loss 0.28551188 reg_l1 27.821014 reg_l2 12.066262
loss 3.0676134
STEP 359 ================================
prereg loss 0.2853757 reg_l1 27.818243 reg_l2 12.06687
loss 3.0672002
STEP 360 ================================
prereg loss 0.2856828 reg_l1 27.815235 reg_l2 12.067693
loss 3.0672064
STEP 361 ================================
prereg loss 0.28575557 reg_l1 27.812653 reg_l2 12.068683
loss 3.067021
cutoff 0.006896167 network size 163
STEP 362 ================================
prereg loss 0.28531525 reg_l1 27.80361 reg_l2 12.069751
loss 3.0656762
STEP 363 ================================
prereg loss 0.2849198 reg_l1 27.801264 reg_l2 12.070702
loss 3.065046
STEP 364 ================================
prereg loss 0.28759587 reg_l1 27.798706 reg_l2 12.071616
loss 3.0674667
STEP 365 ================================
prereg loss 0.28552544 reg_l1 27.796753 reg_l2 12.072727
loss 3.0652008
STEP 366 ================================
prereg loss 0.28591773 reg_l1 27.795277 reg_l2 12.0742655
loss 3.0654454
STEP 367 ================================
prereg loss 0.2853436 reg_l1 27.793795 reg_l2 12.076351
loss 3.0647233
STEP 368 ================================
prereg loss 0.28720486 reg_l1 27.792435 reg_l2 12.078668
loss 3.0664482
STEP 369 ================================
prereg loss 0.2844485 reg_l1 27.791555 reg_l2 12.080842
loss 3.0636039
STEP 370 ================================
prereg loss 0.28447783 reg_l1 27.790194 reg_l2 12.082724
loss 3.063497
STEP 371 ================================
prereg loss 0.28437093 reg_l1 27.787884 reg_l2 12.084407
loss 3.0631592
STEP 372 ================================
prereg loss 0.28620943 reg_l1 27.785439 reg_l2 12.086045
loss 3.0647533
STEP 373 ================================
prereg loss 0.28443003 reg_l1 27.783693 reg_l2 12.08789
loss 3.0627995
STEP 374 ================================
prereg loss 0.2839264 reg_l1 27.78208 reg_l2 12.090094
loss 3.0621345
STEP 375 ================================
prereg loss 0.28341916 reg_l1 27.780333 reg_l2 12.092553
loss 3.0614524
STEP 376 ================================
prereg loss 0.28423584 reg_l1 27.77849 reg_l2 12.094888
loss 3.0620847
STEP 377 ================================
prereg loss 0.28217575 reg_l1 27.77659 reg_l2 12.096912
loss 3.059835
STEP 378 ================================
prereg loss 0.2819879 reg_l1 27.774046 reg_l2 12.098662
loss 3.0593925
STEP 379 ================================
prereg loss 0.28233984 reg_l1 27.771177 reg_l2 12.100284
loss 3.0594575
STEP 380 ================================
prereg loss 0.28203925 reg_l1 27.768456 reg_l2 12.101806
loss 3.0588849
STEP 381 ================================
prereg loss 0.28108457 reg_l1 27.765856 reg_l2 12.103302
loss 3.05767
cutoff 0.0050673205 network size 162
STEP 382 ================================
prereg loss 0.28073275 reg_l1 27.757957 reg_l2 12.10479
loss 3.0565286
STEP 383 ================================
prereg loss 0.28137797 reg_l1 27.755005 reg_l2 12.106287
loss 3.0568786
STEP 384 ================================
prereg loss 0.28019205 reg_l1 27.752245 reg_l2 12.107595
loss 3.0554166
STEP 385 ================================
prereg loss 0.2799186 reg_l1 27.749104 reg_l2 12.108626
loss 3.0548291
STEP 386 ================================
prereg loss 0.28004444 reg_l1 27.745506 reg_l2 12.10935
loss 3.054595
STEP 387 ================================
prereg loss 0.28038058 reg_l1 27.741669 reg_l2 12.109736
loss 3.0545473
STEP 388 ================================
prereg loss 0.27940097 reg_l1 27.737925 reg_l2 12.109945
loss 3.0531936
STEP 389 ================================
prereg loss 0.27935675 reg_l1 27.733902 reg_l2 12.110201
loss 3.052747
STEP 390 ================================
prereg loss 0.27971086 reg_l1 27.729689 reg_l2 12.1105
loss 3.0526798
STEP 391 ================================
prereg loss 0.28141332 reg_l1 27.725399 reg_l2 12.1105995
loss 3.0539532
STEP 392 ================================
prereg loss 0.2787159 reg_l1 27.72139 reg_l2 12.110371
loss 3.050855
STEP 393 ================================
prereg loss 0.27873325 reg_l1 27.71679 reg_l2 12.109929
loss 3.0504122
STEP 394 ================================
prereg loss 0.27938503 reg_l1 27.71155 reg_l2 12.109298
loss 3.0505402
STEP 395 ================================
prereg loss 0.279234 reg_l1 27.706455 reg_l2 12.108467
loss 3.0498796
STEP 396 ================================
prereg loss 0.27878517 reg_l1 27.701563 reg_l2 12.107536
loss 3.0489416
STEP 397 ================================
prereg loss 0.27859217 reg_l1 27.696491 reg_l2 12.106658
loss 3.0482414
STEP 398 ================================
prereg loss 0.27916884 reg_l1 27.691217 reg_l2 12.105748
loss 3.0482907
STEP 399 ================================
prereg loss 0.2786208 reg_l1 27.686062 reg_l2 12.104617
loss 3.047227
STEP 400 ================================
prereg loss 0.27832904 reg_l1 27.680605 reg_l2 12.103161
loss 3.0463896
STEP 401 ================================
prereg loss 0.27842665 reg_l1 27.674786 reg_l2 12.101478
loss 3.0459054
cutoff 0.005686971 network size 161
STEP 402 ================================
prereg loss 0.27904698 reg_l1 27.66307 reg_l2 12.099572
loss 3.0453541
STEP 403 ================================
prereg loss 0.2784716 reg_l1 27.657421 reg_l2 12.097697
loss 3.0442138
STEP 404 ================================
prereg loss 0.27838337 reg_l1 27.651619 reg_l2 12.095906
loss 3.0435452
STEP 405 ================================
prereg loss 0.28000534 reg_l1 27.645714 reg_l2 12.094149
loss 3.0445766
STEP 406 ================================
prereg loss 0.2781225 reg_l1 27.64021 reg_l2 12.092333
loss 3.0421433
STEP 407 ================================
prereg loss 0.27803078 reg_l1 27.634407 reg_l2 12.090409
loss 3.0414717
STEP 408 ================================
prereg loss 0.2790377 reg_l1 27.628311 reg_l2 12.088404
loss 3.041869
STEP 409 ================================
prereg loss 0.2780834 reg_l1 27.622522 reg_l2 12.08635
loss 3.0403357
STEP 410 ================================
prereg loss 0.27805242 reg_l1 27.616589 reg_l2 12.084322
loss 3.0397112
STEP 411 ================================
prereg loss 0.27886733 reg_l1 27.61058 reg_l2 12.082321
loss 3.0399253
STEP 412 ================================
prereg loss 0.27778864 reg_l1 27.604927 reg_l2 12.08036
loss 3.0382814
STEP 413 ================================
prereg loss 0.2777633 reg_l1 27.599018 reg_l2 12.078415
loss 3.0376651
STEP 414 ================================
prereg loss 0.27902174 reg_l1 27.592897 reg_l2 12.076447
loss 3.0383115
STEP 415 ================================
prereg loss 0.2777034 reg_l1 27.58714 reg_l2 12.074437
loss 3.0364175
STEP 416 ================================
prereg loss 0.27765948 reg_l1 27.581177 reg_l2 12.0724325
loss 3.035777
STEP 417 ================================
prereg loss 0.2786711 reg_l1 27.575476 reg_l2 12.070472
loss 3.0362186
STEP 418 ================================
prereg loss 0.27809623 reg_l1 27.570517 reg_l2 12.068531
loss 3.035148
STEP 419 ================================
prereg loss 0.2775305 reg_l1 27.565638 reg_l2 12.0666685
loss 3.0340943
STEP 420 ================================
prereg loss 0.27771446 reg_l1 27.56049 reg_l2 12.06492
loss 3.0337634
STEP 421 ================================
prereg loss 0.27802533 reg_l1 27.555445 reg_l2 12.0632715
loss 3.0335698
cutoff 0.0013554803 network size 160
STEP 422 ================================
prereg loss 0.27737507 reg_l1 27.549253 reg_l2 12.061668
loss 3.0323005
STEP 423 ================================
prereg loss 0.27727887 reg_l1 27.544212 reg_l2 12.060111
loss 3.0317001
STEP 424 ================================
prereg loss 0.27823132 reg_l1 27.539028 reg_l2 12.058528
loss 3.0321343
STEP 425 ================================
prereg loss 0.27714086 reg_l1 27.534042 reg_l2 12.056923
loss 3.0305452
STEP 426 ================================
prereg loss 0.27714905 reg_l1 27.528887 reg_l2 12.055333
loss 3.0300376
STEP 427 ================================
prereg loss 0.27787054 reg_l1 27.52364 reg_l2 12.053798
loss 3.0302343
STEP 428 ================================
prereg loss 0.2771305 reg_l1 27.518694 reg_l2 12.052359
loss 3.029
STEP 429 ================================
prereg loss 0.27718496 reg_l1 27.513695 reg_l2 12.050992
loss 3.0285544
STEP 430 ================================
prereg loss 0.2769263 reg_l1 27.50882 reg_l2 12.049712
loss 3.0278082
STEP 431 ================================
prereg loss 0.27781355 reg_l1 27.503763 reg_l2 12.048468
loss 3.0281901
STEP 432 ================================
prereg loss 0.27682889 reg_l1 27.499065 reg_l2 12.047286
loss 3.0267353
STEP 433 ================================
prereg loss 0.27682865 reg_l1 27.49424 reg_l2 12.046201
loss 3.0262527
STEP 434 ================================
prereg loss 0.27800506 reg_l1 27.489334 reg_l2 12.045174
loss 3.0269387
STEP 435 ================================
prereg loss 0.27649552 reg_l1 27.484882 reg_l2 12.04418
loss 3.0249836
STEP 436 ================================
prereg loss 0.27639747 reg_l1 27.480162 reg_l2 12.0432205
loss 3.0244136
STEP 437 ================================
prereg loss 0.27752155 reg_l1 27.475227 reg_l2 12.042274
loss 3.0250444
STEP 438 ================================
prereg loss 0.27634278 reg_l1 27.470633 reg_l2 12.04131
loss 3.0234063
STEP 439 ================================
prereg loss 0.27624965 reg_l1 27.46589 reg_l2 12.040351
loss 3.0228388
STEP 440 ================================
prereg loss 0.2767698 reg_l1 27.461008 reg_l2 12.039429
loss 3.0228708
STEP 441 ================================
prereg loss 0.27621022 reg_l1 27.456347 reg_l2 12.038507
loss 3.021845
cutoff 0.005903244 network size 159
STEP 442 ================================
prereg loss 0.3164837 reg_l1 27.445892 reg_l2 12.037587
loss 3.061073
STEP 443 ================================
prereg loss 0.2817776 reg_l1 27.440296 reg_l2 12.0333395
loss 3.0258074
STEP 444 ================================
prereg loss 0.30444723 reg_l1 27.432291 reg_l2 12.027883
loss 3.0476763
STEP 445 ================================
prereg loss 0.32786983 reg_l1 27.422863 reg_l2 12.023269
loss 3.0701563
STEP 446 ================================
prereg loss 0.30039176 reg_l1 27.416754 reg_l2 12.021295
loss 3.042067
STEP 447 ================================
prereg loss 0.32172346 reg_l1 27.415564 reg_l2 12.022555
loss 3.0632799
STEP 448 ================================
prereg loss 0.32305855 reg_l1 27.41617 reg_l2 12.025866
loss 3.0646756
STEP 449 ================================
prereg loss 0.2844364 reg_l1 27.414848 reg_l2 12.028324
loss 3.0259213
STEP 450 ================================
prereg loss 0.34336686 reg_l1 27.409794 reg_l2 12.026879
loss 3.0843463
STEP 451 ================================
prereg loss 0.31215328 reg_l1 27.401438 reg_l2 12.021451
loss 3.052297
STEP 452 ================================
prereg loss 0.28438693 reg_l1 27.391716 reg_l2 12.014818
loss 3.0235586
STEP 453 ================================
prereg loss 0.31902066 reg_l1 27.383432 reg_l2 12.010787
loss 3.057364
STEP 454 ================================
prereg loss 0.31118232 reg_l1 27.379284 reg_l2 12.011086
loss 3.0491107
STEP 455 ================================
prereg loss 0.28125092 reg_l1 27.379158 reg_l2 12.014212
loss 3.0191667
STEP 456 ================================
prereg loss 0.29022563 reg_l1 27.379671 reg_l2 12.017102
loss 3.028193
STEP 457 ================================
prereg loss 0.3008131 reg_l1 27.376547 reg_l2 12.0171175
loss 3.038468
STEP 458 ================================
prereg loss 0.28092036 reg_l1 27.368862 reg_l2 12.013894
loss 3.0178065
STEP 459 ================================
prereg loss 0.28418267 reg_l1 27.360014 reg_l2 12.009477
loss 3.020184
STEP 460 ================================
prereg loss 0.29856727 reg_l1 27.35342 reg_l2 12.006437
loss 3.0339093
STEP 461 ================================
prereg loss 0.2888194 reg_l1 27.350372 reg_l2 12.00624
loss 3.0238566
cutoff 0.005758533 network size 158
STEP 462 ================================
prereg loss 0.27766743 reg_l1 27.343586 reg_l2 12.008398
loss 3.012026
STEP 463 ================================
prereg loss 0.29247278 reg_l1 27.34209 reg_l2 12.010937
loss 3.026682
STEP 464 ================================
prereg loss 0.2879552 reg_l1 27.339334 reg_l2 12.011959
loss 3.0218887
STEP 465 ================================
prereg loss 0.27651602 reg_l1 27.335142 reg_l2 12.011062
loss 3.0100303
STEP 466 ================================
prereg loss 0.2832592 reg_l1 27.329657 reg_l2 12.009257
loss 3.0162249
STEP 467 ================================
prereg loss 0.27944955 reg_l1 27.324306 reg_l2 12.007924
loss 3.0118802
STEP 468 ================================
prereg loss 0.279781 reg_l1 27.320204 reg_l2 12.007748
loss 3.0118015
STEP 469 ================================
prereg loss 0.28027013 reg_l1 27.317577 reg_l2 12.00856
loss 3.012028
STEP 470 ================================
prereg loss 0.27524516 reg_l1 27.315567 reg_l2 12.009891
loss 3.0068018
STEP 471 ================================
prereg loss 0.27408102 reg_l1 27.312744 reg_l2 12.011184
loss 3.0053554
STEP 472 ================================
prereg loss 0.27637953 reg_l1 27.309202 reg_l2 12.012019
loss 3.0073
STEP 473 ================================
prereg loss 0.27501416 reg_l1 27.305683 reg_l2 12.012227
loss 3.0055826
STEP 474 ================================
prereg loss 0.27324718 reg_l1 27.30189 reg_l2 12.011869
loss 3.0034363
STEP 475 ================================
prereg loss 0.27485272 reg_l1 27.297586 reg_l2 12.011347
loss 3.0046115
STEP 476 ================================
prereg loss 0.27560994 reg_l1 27.293264 reg_l2 12.011096
loss 3.0049365
STEP 477 ================================
prereg loss 0.27453294 reg_l1 27.289623 reg_l2 12.0114565
loss 3.0034955
STEP 478 ================================
prereg loss 0.27510342 reg_l1 27.286942 reg_l2 12.01251
loss 3.0037975
STEP 479 ================================
prereg loss 0.27366313 reg_l1 27.284658 reg_l2 12.013929
loss 3.0021288
STEP 480 ================================
prereg loss 0.2727717 reg_l1 27.282005 reg_l2 12.015062
loss 3.0009723
STEP 481 ================================
prereg loss 0.27489066 reg_l1 27.278677 reg_l2 12.015441
loss 3.0027585
cutoff 0.004403651 network size 157
STEP 482 ================================
prereg loss 0.27227482 reg_l1 27.270317 reg_l2 12.015057
loss 2.9993064
STEP 483 ================================
prereg loss 0.27229783 reg_l1 27.265953 reg_l2 12.014499
loss 2.9988933
STEP 484 ================================
prereg loss 0.27401236 reg_l1 27.261978 reg_l2 12.014407
loss 3.0002103
STEP 485 ================================
prereg loss 0.27332374 reg_l1 27.259022 reg_l2 12.015041
loss 2.999226
STEP 486 ================================
prereg loss 0.2713489 reg_l1 27.256712 reg_l2 12.016124
loss 2.9970202
STEP 487 ================================
prereg loss 0.27176264 reg_l1 27.254059 reg_l2 12.017042
loss 2.9971685
STEP 488 ================================
prereg loss 0.27265093 reg_l1 27.250652 reg_l2 12.017329
loss 2.9977162
STEP 489 ================================
prereg loss 0.27089548 reg_l1 27.24667 reg_l2 12.016985
loss 2.9955626
STEP 490 ================================
prereg loss 0.2717476 reg_l1 27.242298 reg_l2 12.016424
loss 2.9959774
STEP 491 ================================
prereg loss 0.27247944 reg_l1 27.238354 reg_l2 12.016161
loss 2.996315
STEP 492 ================================
prereg loss 0.27122146 reg_l1 27.235376 reg_l2 12.016472
loss 2.994759
STEP 493 ================================
prereg loss 0.2704628 reg_l1 27.232649 reg_l2 12.017187
loss 2.9937277
STEP 494 ================================
prereg loss 0.27231818 reg_l1 27.229668 reg_l2 12.017882
loss 2.995285
STEP 495 ================================
prereg loss 0.27030295 reg_l1 27.226599 reg_l2 12.018237
loss 2.9929628
STEP 496 ================================
prereg loss 0.26997998 reg_l1 27.22318 reg_l2 12.018208
loss 2.9922981
STEP 497 ================================
prereg loss 0.27014363 reg_l1 27.21931 reg_l2 12.018003
loss 2.992075
STEP 498 ================================
prereg loss 0.27094734 reg_l1 27.215437 reg_l2 12.017815
loss 2.9924912
STEP 499 ================================
prereg loss 0.26968864 reg_l1 27.21208 reg_l2 12.017784
loss 2.9908967
STEP 500 ================================
prereg loss 0.26928025 reg_l1 27.20874 reg_l2 12.0178795
loss 2.9901543
STEP 501 ================================
prereg loss 0.26930916 reg_l1 27.205172 reg_l2 12.018026
loss 2.9898262
cutoff 0.004298332 network size 156
STEP 502 ================================
prereg loss 0.2692008 reg_l1 27.19735 reg_l2 12.018167
loss 2.9889357
STEP 503 ================================
prereg loss 0.2693094 reg_l1 27.193712 reg_l2 12.018206
loss 2.9886808
STEP 504 ================================
prereg loss 0.26860458 reg_l1 27.19021 reg_l2 12.018147
loss 2.9876256
STEP 505 ================================
prereg loss 0.26844528 reg_l1 27.18645 reg_l2 12.018042
loss 2.98709
STEP 506 ================================
prereg loss 0.27020234 reg_l1 27.182493 reg_l2 12.017922
loss 2.9884517
STEP 507 ================================
prereg loss 0.26829368 reg_l1 27.178953 reg_l2 12.017902
loss 2.986189
STEP 508 ================================
prereg loss 0.26812994 reg_l1 27.17542 reg_l2 12.018042
loss 2.985672
STEP 509 ================================
prereg loss 0.2682423 reg_l1 27.171919 reg_l2 12.018358
loss 2.9854343
STEP 510 ================================
prereg loss 0.26775235 reg_l1 27.168663 reg_l2 12.018726
loss 2.9846187
STEP 511 ================================
prereg loss 0.26722628 reg_l1 27.165432 reg_l2 12.019024
loss 2.9837694
STEP 512 ================================
prereg loss 0.2673705 reg_l1 27.16183 reg_l2 12.019173
loss 2.9835536
STEP 513 ================================
prereg loss 0.26722997 reg_l1 27.158205 reg_l2 12.019206
loss 2.9830506
STEP 514 ================================
prereg loss 0.26730537 reg_l1 27.154472 reg_l2 12.019186
loss 2.9827526
STEP 515 ================================
prereg loss 0.26684278 reg_l1 27.15105 reg_l2 12.019278
loss 2.981948
STEP 516 ================================
prereg loss 0.26666516 reg_l1 27.147606 reg_l2 12.019531
loss 2.9814258
STEP 517 ================================
prereg loss 0.2670228 reg_l1 27.144115 reg_l2 12.019848
loss 2.9814343
STEP 518 ================================
prereg loss 0.26611745 reg_l1 27.140856 reg_l2 12.020094
loss 2.9802032
STEP 519 ================================
prereg loss 0.2659921 reg_l1 27.137249 reg_l2 12.02019
loss 2.979717
STEP 520 ================================
prereg loss 0.26691544 reg_l1 27.133316 reg_l2 12.020156
loss 2.980247
STEP 521 ================================
prereg loss 0.26584694 reg_l1 27.12967 reg_l2 12.0200815
loss 2.978814
cutoff 0.0041018426 network size 155
STEP 522 ================================
prereg loss 0.26576555 reg_l1 27.121822 reg_l2 12.020074
loss 2.9779477
STEP 523 ================================
prereg loss 0.26616132 reg_l1 27.118132 reg_l2 12.020224
loss 2.9779744
STEP 524 ================================
prereg loss 0.26550522 reg_l1 27.114704 reg_l2 12.020436
loss 2.9769757
STEP 525 ================================
prereg loss 0.2649865 reg_l1 27.11134 reg_l2 12.02065
loss 2.9761205
STEP 526 ================================
prereg loss 0.26496187 reg_l1 27.10762 reg_l2 12.0208
loss 2.975724
STEP 527 ================================
prereg loss 0.266447 reg_l1 27.103664 reg_l2 12.020819
loss 2.9768136
STEP 528 ================================
prereg loss 0.26453426 reg_l1 27.100065 reg_l2 12.020773
loss 2.9745407
STEP 529 ================================
prereg loss 0.26434532 reg_l1 27.096281 reg_l2 12.020754
loss 2.9739735
STEP 530 ================================
prereg loss 0.2645716 reg_l1 27.092276 reg_l2 12.020812
loss 2.9737992
STEP 531 ================================
prereg loss 0.26445252 reg_l1 27.088455 reg_l2 12.0209
loss 2.973298
STEP 532 ================================
prereg loss 0.26371008 reg_l1 27.084839 reg_l2 12.020996
loss 2.972194
STEP 533 ================================
prereg loss 0.26354995 reg_l1 27.081015 reg_l2 12.02112
loss 2.9716516
STEP 534 ================================
prereg loss 0.26409394 reg_l1 27.076988 reg_l2 12.021223
loss 2.9717927
STEP 535 ================================
prereg loss 0.26339114 reg_l1 27.073164 reg_l2 12.02122
loss 2.9707074
STEP 536 ================================
prereg loss 0.26311165 reg_l1 27.069223 reg_l2 12.021165
loss 2.970034
STEP 537 ================================
prereg loss 0.26316988 reg_l1 27.06514 reg_l2 12.021092
loss 2.9696841
STEP 538 ================================
prereg loss 0.26366633 reg_l1 27.060976 reg_l2 12.02098
loss 2.969764
STEP 539 ================================
prereg loss 0.26272973 reg_l1 27.05715 reg_l2 12.0209055
loss 2.9684446
STEP 540 ================================
prereg loss 0.26261136 reg_l1 27.05316 reg_l2 12.020951
loss 2.9679275
STEP 541 ================================
prereg loss 0.26333424 reg_l1 27.049082 reg_l2 12.021068
loss 2.9682424
cutoff 0.004322265 network size 154
STEP 542 ================================
prereg loss 0.2625346 reg_l1 27.04091 reg_l2 12.021142
loss 2.9666257
STEP 543 ================================
prereg loss 0.2621487 reg_l1 27.036993 reg_l2 12.02114
loss 2.965848
STEP 544 ================================
prereg loss 0.26216075 reg_l1 27.032871 reg_l2 12.021098
loss 2.965448
STEP 545 ================================
prereg loss 0.26277617 reg_l1 27.028639 reg_l2 12.021033
loss 2.96564
STEP 546 ================================
prereg loss 0.26182377 reg_l1 27.024687 reg_l2 12.021039
loss 2.9642925
STEP 547 ================================
prereg loss 0.26166537 reg_l1 27.02073 reg_l2 12.021175
loss 2.9637382
STEP 548 ================================
prereg loss 0.26203844 reg_l1 27.01673 reg_l2 12.021419
loss 2.9637115
STEP 549 ================================
prereg loss 0.26132715 reg_l1 27.012932 reg_l2 12.021676
loss 2.9626203
STEP 550 ================================
prereg loss 0.261152 reg_l1 27.008951 reg_l2 12.021854
loss 2.962047
STEP 551 ================================
prereg loss 0.26150396 reg_l1 27.004818 reg_l2 12.021956
loss 2.9619858
STEP 552 ================================
prereg loss 0.26075745 reg_l1 27.000896 reg_l2 12.022099
loss 2.9608471
STEP 553 ================================
prereg loss 0.26064765 reg_l1 26.99693 reg_l2 12.022364
loss 2.9603405
STEP 554 ================================
prereg loss 0.2607646 reg_l1 26.993048 reg_l2 12.022747
loss 2.9600694
STEP 555 ================================
prereg loss 0.2600806 reg_l1 26.98942 reg_l2 12.023186
loss 2.9590225
STEP 556 ================================
prereg loss 0.25981447 reg_l1 26.985662 reg_l2 12.023595
loss 2.9583807
STEP 557 ================================
prereg loss 0.2600007 reg_l1 26.981709 reg_l2 12.023921
loss 2.9581716
STEP 558 ================================
prereg loss 0.25941393 reg_l1 26.977869 reg_l2 12.024188
loss 2.957201
STEP 559 ================================
prereg loss 0.2592406 reg_l1 26.973948 reg_l2 12.0244665
loss 2.9566355
STEP 560 ================================
prereg loss 0.2592644 reg_l1 26.97004 reg_l2 12.024835
loss 2.9562685
STEP 561 ================================
prereg loss 0.25918517 reg_l1 26.966187 reg_l2 12.025217
loss 2.9558039
cutoff 0.0053670486 network size 153
STEP 562 ================================
prereg loss 0.25927728 reg_l1 26.956974 reg_l2 12.025512
loss 2.954975
STEP 563 ================================
prereg loss 0.25880086 reg_l1 26.95297 reg_l2 12.025645
loss 2.9540977
STEP 564 ================================
prereg loss 0.2586985 reg_l1 26.94877 reg_l2 12.025708
loss 2.9535756
STEP 565 ================================
prereg loss 0.25902537 reg_l1 26.94453 reg_l2 12.025838
loss 2.9534783
STEP 566 ================================
prereg loss 0.2582784 reg_l1 26.940672 reg_l2 12.026153
loss 2.9523456
STEP 567 ================================
prereg loss 0.25790238 reg_l1 26.936882 reg_l2 12.02665
loss 2.9515905
STEP 568 ================================
prereg loss 0.2579172 reg_l1 26.933039 reg_l2 12.027202
loss 2.951221
STEP 569 ================================
prereg loss 0.2571557 reg_l1 26.929289 reg_l2 12.027656
loss 2.9500847
STEP 570 ================================
prereg loss 0.25688377 reg_l1 26.925205 reg_l2 12.027926
loss 2.9494045
STEP 571 ================================
prereg loss 0.25696582 reg_l1 26.920881 reg_l2 12.028062
loss 2.949054
STEP 572 ================================
prereg loss 0.25688198 reg_l1 26.91652 reg_l2 12.028145
loss 2.948534
STEP 573 ================================
prereg loss 0.2563772 reg_l1 26.91227 reg_l2 12.028269
loss 2.9476042
STEP 574 ================================
prereg loss 0.25596106 reg_l1 26.908045 reg_l2 12.028468
loss 2.9467654
STEP 575 ================================
prereg loss 0.25586206 reg_l1 26.903723 reg_l2 12.028667
loss 2.9462342
STEP 576 ================================
prereg loss 0.25632852 reg_l1 26.89922 reg_l2 12.028745
loss 2.9462507
STEP 577 ================================
prereg loss 0.2552238 reg_l1 26.894846 reg_l2 12.028694
loss 2.9447083
STEP 578 ================================
prereg loss 0.25507745 reg_l1 26.89218 reg_l2 12.028594
loss 2.9442954
STEP 579 ================================
prereg loss 0.25548503 reg_l1 26.889168 reg_l2 12.028534
loss 2.944402
STEP 580 ================================
prereg loss 0.2546515 reg_l1 26.886305 reg_l2 12.028556
loss 2.9432821
STEP 581 ================================
prereg loss 0.25420305 reg_l1 26.883245 reg_l2 12.028636
loss 2.9425278
cutoff 0.0015028887 network size 152
STEP 582 ================================
prereg loss 0.25405538 reg_l1 26.878414 reg_l2 12.028702
loss 2.941897
STEP 583 ================================
prereg loss 0.25425905 reg_l1 26.875126 reg_l2 12.028633
loss 2.9417717
STEP 584 ================================
prereg loss 0.25315112 reg_l1 26.871954 reg_l2 12.02842
loss 2.9403467
STEP 585 ================================
prereg loss 0.25291583 reg_l1 26.868559 reg_l2 12.028174
loss 2.939772
STEP 586 ================================
prereg loss 0.25315028 reg_l1 26.865025 reg_l2 12.027931
loss 2.9396527
STEP 587 ================================
prereg loss 0.25341484 reg_l1 26.861534 reg_l2 12.027651
loss 2.9395683
STEP 588 ================================
prereg loss 0.25204444 reg_l1 26.858353 reg_l2 12.027328
loss 2.9378798
STEP 589 ================================
prereg loss 0.2517633 reg_l1 26.854952 reg_l2 12.027036
loss 2.9372585
STEP 590 ================================
prereg loss 0.25215694 reg_l1 26.851315 reg_l2 12.026732
loss 2.9372885
STEP 591 ================================
prereg loss 0.25310186 reg_l1 26.847681 reg_l2 12.026289
loss 2.93787
STEP 592 ================================
prereg loss 0.25110888 reg_l1 26.844385 reg_l2 12.025743
loss 2.9355474
STEP 593 ================================
prereg loss 0.25088286 reg_l1 26.840908 reg_l2 12.025253
loss 2.9349737
STEP 594 ================================
prereg loss 0.25116217 reg_l1 26.837236 reg_l2 12.024863
loss 2.934886
STEP 595 ================================
prereg loss 0.25104436 reg_l1 26.8338 reg_l2 12.024474
loss 2.9344242
STEP 596 ================================
prereg loss 0.25004628 reg_l1 26.830593 reg_l2 12.024037
loss 2.9331057
STEP 597 ================================
prereg loss 0.2497347 reg_l1 26.827164 reg_l2 12.023624
loss 2.932451
STEP 598 ================================
prereg loss 0.2501238 reg_l1 26.823957 reg_l2 12.023184
loss 2.9325194
STEP 599 ================================
prereg loss 0.25056687 reg_l1 26.820677 reg_l2 12.022572
loss 2.9326346
STEP 600 ================================
prereg loss 0.2489354 reg_l1 26.817535 reg_l2 12.021864
loss 2.930689
STEP 601 ================================
prereg loss 0.24873497 reg_l1 26.814625 reg_l2 12.02123
loss 2.9301975
cutoff 7.403603e-5 network size 151
STEP 602 ================================
prereg loss 0.24898258 reg_l1 26.811357 reg_l2 12.020728
loss 2.9301186
STEP 603 ================================
prereg loss 0.24994874 reg_l1 26.808352 reg_l2 12.020174
loss 2.930784
STEP 604 ================================
prereg loss 0.24788128 reg_l1 26.805618 reg_l2 12.019481
loss 2.928443
STEP 605 ================================
prereg loss 0.24761969 reg_l1 26.80252 reg_l2 12.018705
loss 2.9278717
STEP 606 ================================
prereg loss 0.24792992 reg_l1 26.799004 reg_l2 12.0179205
loss 2.9278302
STEP 607 ================================
prereg loss 0.24971257 reg_l1 26.795431 reg_l2 12.017051
loss 2.9292557
STEP 608 ================================
prereg loss 0.24708375 reg_l1 26.792437 reg_l2 12.016193
loss 2.9263275
STEP 609 ================================
prereg loss 0.24692322 reg_l1 26.789284 reg_l2 12.015513
loss 2.9258516
STEP 610 ================================
prereg loss 0.24691918 reg_l1 26.78586 reg_l2 12.014994
loss 2.9255052
STEP 611 ================================
prereg loss 0.24933963 reg_l1 26.782356 reg_l2 12.014323
loss 2.9275753
STEP 612 ================================
prereg loss 0.24576178 reg_l1 26.779266 reg_l2 12.013416
loss 2.9236887
STEP 613 ================================
prereg loss 0.24575438 reg_l1 26.775763 reg_l2 12.012474
loss 2.9233308
STEP 614 ================================
prereg loss 0.24583936 reg_l1 26.771885 reg_l2 12.01166
loss 2.923028
STEP 615 ================================
prereg loss 0.24819249 reg_l1 26.768084 reg_l2 12.010837
loss 2.925001
STEP 616 ================================
prereg loss 0.24481726 reg_l1 26.764856 reg_l2 12.009952
loss 2.921303
STEP 617 ================================
prereg loss 0.24463703 reg_l1 26.76164 reg_l2 12.009064
loss 2.9208012
STEP 618 ================================
prereg loss 0.24459925 reg_l1 26.758001 reg_l2 12.008204
loss 2.9203994
STEP 619 ================================
prereg loss 0.24562155 reg_l1 26.754366 reg_l2 12.007263
loss 2.9210582
STEP 620 ================================
prereg loss 0.24394357 reg_l1 26.751003 reg_l2 12.006237
loss 2.919044
STEP 621 ================================
prereg loss 0.24361286 reg_l1 26.747784 reg_l2 12.005268
loss 2.9183912
cutoff 0.00014893405 network size 150
STEP 622 ================================
prereg loss 0.24376751 reg_l1 26.744156 reg_l2 12.004386
loss 2.918183
STEP 623 ================================
prereg loss 0.2435605 reg_l1 26.740795 reg_l2 12.003492
loss 2.9176402
STEP 624 ================================
prereg loss 0.24270299 reg_l1 26.737547 reg_l2 12.002535
loss 2.9164577
STEP 625 ================================
prereg loss 0.24253568 reg_l1 26.734007 reg_l2 12.00156
loss 2.9159362
STEP 626 ================================
prereg loss 0.24320135 reg_l1 26.730328 reg_l2 12.000566
loss 2.916234
STEP 627 ================================
prereg loss 0.24209477 reg_l1 26.726912 reg_l2 11.999531
loss 2.9147859
STEP 628 ================================
prereg loss 0.24173059 reg_l1 26.723558 reg_l2 11.998475
loss 2.9140866
STEP 629 ================================
prereg loss 0.24198934 reg_l1 26.719946 reg_l2 11.997399
loss 2.913984
STEP 630 ================================
prereg loss 0.24142773 reg_l1 26.716433 reg_l2 11.996319
loss 2.913071
STEP 631 ================================
prereg loss 0.24139273 reg_l1 26.712759 reg_l2 11.995211
loss 2.9126687
STEP 632 ================================
prereg loss 0.24088939 reg_l1 26.70924 reg_l2 11.994138
loss 2.9118133
STEP 633 ================================
prereg loss 0.24091043 reg_l1 26.705713 reg_l2 11.993089
loss 2.9114819
STEP 634 ================================
prereg loss 0.24049537 reg_l1 26.702396 reg_l2 11.992045
loss 2.9107351
STEP 635 ================================
prereg loss 0.24067754 reg_l1 26.698961 reg_l2 11.99098
loss 2.9105737
STEP 636 ================================
prereg loss 0.23994906 reg_l1 26.69562 reg_l2 11.989901
loss 2.909511
STEP 637 ================================
prereg loss 0.23984736 reg_l1 26.692087 reg_l2 11.988823
loss 2.9090562
STEP 638 ================================
prereg loss 0.24047856 reg_l1 26.688433 reg_l2 11.98773
loss 2.9093218
STEP 639 ================================
prereg loss 0.23929735 reg_l1 26.684984 reg_l2 11.986623
loss 2.907796
STEP 640 ================================
prereg loss 0.23907912 reg_l1 26.681427 reg_l2 11.985506
loss 2.9072218
STEP 641 ================================
prereg loss 0.23988907 reg_l1 26.67768 reg_l2 11.984363
loss 2.9076571
cutoff 0.00012538175 network size 149
STEP 642 ================================
prereg loss 0.23896386 reg_l1 26.673973 reg_l2 11.983193
loss 2.906361
STEP 643 ================================
prereg loss 0.23846516 reg_l1 26.67046 reg_l2 11.982076
loss 2.9055111
STEP 644 ================================
prereg loss 0.23848781 reg_l1 26.6667 reg_l2 11.981034
loss 2.9051578
STEP 645 ================================
prereg loss 0.23859294 reg_l1 26.663103 reg_l2 11.9800415
loss 2.9049032
STEP 646 ================================
prereg loss 0.2377133 reg_l1 26.659704 reg_l2 11.979031
loss 2.903684
STEP 647 ================================
prereg loss 0.23752588 reg_l1 26.656221 reg_l2 11.977989
loss 2.9031482
STEP 648 ================================
prereg loss 0.2382816 reg_l1 26.65254 reg_l2 11.976874
loss 2.9035356
STEP 649 ================================
prereg loss 0.2371454 reg_l1 26.649084 reg_l2 11.975733
loss 2.9020538
STEP 650 ================================
prereg loss 0.23689282 reg_l1 26.645636 reg_l2 11.974621
loss 2.9014564
STEP 651 ================================
prereg loss 0.23675932 reg_l1 26.641916 reg_l2 11.973589
loss 2.9009511
STEP 652 ================================
prereg loss 0.23832072 reg_l1 26.63812 reg_l2 11.972538
loss 2.902133
STEP 653 ================================
prereg loss 0.23639277 reg_l1 26.634659 reg_l2 11.971408
loss 2.8998587
STEP 654 ================================
prereg loss 0.23601946 reg_l1 26.631153 reg_l2 11.97026
loss 2.8991346
STEP 655 ================================
prereg loss 0.23588614 reg_l1 26.627209 reg_l2 11.969165
loss 2.898607
STEP 656 ================================
prereg loss 0.23842132 reg_l1 26.623213 reg_l2 11.968089
loss 2.9007425
STEP 657 ================================
prereg loss 0.23566362 reg_l1 26.619877 reg_l2 11.967004
loss 2.8976514
STEP 658 ================================
prereg loss 0.23546755 reg_l1 26.6167 reg_l2 11.965981
loss 2.8971376
STEP 659 ================================
prereg loss 0.2348951 reg_l1 26.612978 reg_l2 11.965044
loss 2.8961928
STEP 660 ================================
prereg loss 0.23843212 reg_l1 26.609026 reg_l2 11.964063
loss 2.899335
STEP 661 ================================
prereg loss 0.23518857 reg_l1 26.605639 reg_l2 11.962941
loss 2.8957524
cutoff 5.232748e-5 network size 148
STEP 662 ================================
prereg loss 0.23481838 reg_l1 26.602303 reg_l2 11.961806
loss 2.8950489
STEP 663 ================================
prereg loss 0.23408407 reg_l1 26.598486 reg_l2 11.960827
loss 2.8939328
STEP 664 ================================
prereg loss 0.23742233 reg_l1 26.594448 reg_l2 11.959898
loss 2.896867
STEP 665 ================================
prereg loss 0.2347605 reg_l1 26.59105 reg_l2 11.958859
loss 2.8938656
STEP 666 ================================
prereg loss 0.23385659 reg_l1 26.58782 reg_l2 11.95774
loss 2.8926387
STEP 667 ================================
prereg loss 0.23319928 reg_l1 26.583956 reg_l2 11.956709
loss 2.891595
STEP 668 ================================
prereg loss 0.2365089 reg_l1 26.57979 reg_l2 11.955688
loss 2.8944879
STEP 669 ================================
prereg loss 0.23416628 reg_l1 26.576246 reg_l2 11.95456
loss 2.891791
STEP 670 ================================
prereg loss 0.23312488 reg_l1 26.573006 reg_l2 11.953424
loss 2.8904257
STEP 671 ================================
prereg loss 0.23244315 reg_l1 26.569206 reg_l2 11.952441
loss 2.8893638
STEP 672 ================================
prereg loss 0.23574275 reg_l1 26.565077 reg_l2 11.951469
loss 2.8922505
STEP 673 ================================
prereg loss 0.2336838 reg_l1 26.56149 reg_l2 11.950352
loss 2.8898327
STEP 674 ================================
prereg loss 0.23224497 reg_l1 26.558151 reg_l2 11.949168
loss 2.88806
STEP 675 ================================
prereg loss 0.23171073 reg_l1 26.55426 reg_l2 11.948098
loss 2.8871367
STEP 676 ================================
prereg loss 0.23476315 reg_l1 26.55007 reg_l2 11.947075
loss 2.8897703
STEP 677 ================================
prereg loss 0.23290682 reg_l1 26.546488 reg_l2 11.945976
loss 2.8875556
STEP 678 ================================
prereg loss 0.2313967 reg_l1 26.543184 reg_l2 11.944834
loss 2.8857152
STEP 679 ================================
prereg loss 0.23087491 reg_l1 26.539299 reg_l2 11.943776
loss 2.884805
STEP 680 ================================
prereg loss 0.23406684 reg_l1 26.53508 reg_l2 11.9427185
loss 2.8875747
STEP 681 ================================
prereg loss 0.2320946 reg_l1 26.531418 reg_l2 11.941546
loss 2.8852363
cutoff 0.0018076412 network size 147
STEP 682 ================================
prereg loss 0.2306939 reg_l1 26.526213 reg_l2 11.940346
loss 2.883315
STEP 683 ================================
prereg loss 0.23014744 reg_l1 26.52229 reg_l2 11.939252
loss 2.8823764
STEP 684 ================================
prereg loss 0.23350222 reg_l1 26.518057 reg_l2 11.938171
loss 2.8853078
STEP 685 ================================
prereg loss 0.23141962 reg_l1 26.514397 reg_l2 11.936994
loss 2.8828592
STEP 686 ================================
prereg loss 0.22997831 reg_l1 26.510962 reg_l2 11.935762
loss 2.8810744
STEP 687 ================================
prereg loss 0.22949171 reg_l1 26.506947 reg_l2 11.934624
loss 2.8801863
STEP 688 ================================
prereg loss 0.23135598 reg_l1 26.502808 reg_l2 11.93354
loss 2.8816366
STEP 689 ================================
prereg loss 0.2302467 reg_l1 26.499123 reg_l2 11.932428
loss 2.8801591
STEP 690 ================================
prereg loss 0.22902067 reg_l1 26.495672 reg_l2 11.931253
loss 2.878588
STEP 691 ================================
prereg loss 0.2289174 reg_l1 26.49184 reg_l2 11.930127
loss 2.8781013
STEP 692 ================================
prereg loss 0.23009405 reg_l1 26.488043 reg_l2 11.9291115
loss 2.8788984
STEP 693 ================================
prereg loss 0.22913738 reg_l1 26.484678 reg_l2 11.928253
loss 2.8776052
STEP 694 ================================
prereg loss 0.2284077 reg_l1 26.481539 reg_l2 11.927435
loss 2.8765616
STEP 695 ================================
prereg loss 0.22836678 reg_l1 26.478012 reg_l2 11.926607
loss 2.876168
STEP 696 ================================
prereg loss 0.22913875 reg_l1 26.474413 reg_l2 11.925725
loss 2.8765802
STEP 697 ================================
prereg loss 0.22829737 reg_l1 26.470865 reg_l2 11.924717
loss 2.875384
STEP 698 ================================
prereg loss 0.22772637 reg_l1 26.467237 reg_l2 11.923637
loss 2.8744502
STEP 699 ================================
prereg loss 0.22829728 reg_l1 26.463259 reg_l2 11.922576
loss 2.874623
STEP 700 ================================
prereg loss 0.2284121 reg_l1 26.459455 reg_l2 11.921538
loss 2.8743577
STEP 701 ================================
prereg loss 0.22746083 reg_l1 26.455872 reg_l2 11.920484
loss 2.873048
cutoff 0.003325648 network size 146
STEP 702 ================================
prereg loss 0.22727628 reg_l1 26.44893 reg_l2 11.919475
loss 2.8721695
STEP 703 ================================
prereg loss 0.22856228 reg_l1 26.445185 reg_l2 11.918642
loss 2.873081
STEP 704 ================================
prereg loss 0.22809191 reg_l1 26.441755 reg_l2 11.917862
loss 2.8722675
STEP 705 ================================
prereg loss 0.22683287 reg_l1 26.438505 reg_l2 11.91699
loss 2.8706834
STEP 706 ================================
prereg loss 0.22691949 reg_l1 26.434816 reg_l2 11.916028
loss 2.8704011
STEP 707 ================================
prereg loss 0.22763932 reg_l1 26.431097 reg_l2 11.915061
loss 2.870749
STEP 708 ================================
prereg loss 0.22702985 reg_l1 26.427631 reg_l2 11.914207
loss 2.869793
STEP 709 ================================
prereg loss 0.22640939 reg_l1 26.424337 reg_l2 11.913504
loss 2.8688433
STEP 710 ================================
prereg loss 0.22675999 reg_l1 26.42077 reg_l2 11.912773
loss 2.868837
STEP 711 ================================
prereg loss 0.22689109 reg_l1 26.417233 reg_l2 11.911939
loss 2.8686144
STEP 712 ================================
prereg loss 0.22623236 reg_l1 26.41375 reg_l2 11.911101
loss 2.8676074
STEP 713 ================================
prereg loss 0.22588074 reg_l1 26.410185 reg_l2 11.910267
loss 2.8668995
STEP 714 ================================
prereg loss 0.22674845 reg_l1 26.406324 reg_l2 11.909359
loss 2.8673809
STEP 715 ================================
prereg loss 0.22659281 reg_l1 26.402727 reg_l2 11.908494
loss 2.8668656
STEP 716 ================================
prereg loss 0.2256308 reg_l1 26.399374 reg_l2 11.907787
loss 2.8655682
STEP 717 ================================
prereg loss 0.22543858 reg_l1 26.395956 reg_l2 11.907168
loss 2.8650343
STEP 718 ================================
prereg loss 0.22590403 reg_l1 26.392424 reg_l2 11.906606
loss 2.8651464
STEP 719 ================================
prereg loss 0.22596061 reg_l1 26.389046 reg_l2 11.906047
loss 2.8648653
STEP 720 ================================
prereg loss 0.22510892 reg_l1 26.385977 reg_l2 11.905552
loss 2.8637066
STEP 721 ================================
prereg loss 0.22484541 reg_l1 26.383125 reg_l2 11.905113
loss 2.863158
cutoff 0.0004648046 network size 145
STEP 722 ================================
prereg loss 0.22572954 reg_l1 26.37953 reg_l2 11.904555
loss 2.8636825
STEP 723 ================================
prereg loss 0.22537014 reg_l1 26.37631 reg_l2 11.903845
loss 2.8630013
STEP 724 ================================
prereg loss 0.22445361 reg_l1 26.373219 reg_l2 11.903177
loss 2.8617756
STEP 725 ================================
prereg loss 0.22466415 reg_l1 26.369862 reg_l2 11.902553
loss 2.8616505
STEP 726 ================================
prereg loss 0.22497454 reg_l1 26.366653 reg_l2 11.902008
loss 2.86164
STEP 727 ================================
prereg loss 0.22425082 reg_l1 26.36366 reg_l2 11.901578
loss 2.860617
STEP 728 ================================
prereg loss 0.22379121 reg_l1 26.360651 reg_l2 11.901105
loss 2.8598564
STEP 729 ================================
prereg loss 0.22482823 reg_l1 26.357258 reg_l2 11.900523
loss 2.860554
STEP 730 ================================
prereg loss 0.22457352 reg_l1 26.354002 reg_l2 11.899899
loss 2.859974
STEP 731 ================================
prereg loss 0.2234137 reg_l1 26.350891 reg_l2 11.8992195
loss 2.8585029
STEP 732 ================================
prereg loss 0.2233058 reg_l1 26.347696 reg_l2 11.89869
loss 2.8580754
STEP 733 ================================
prereg loss 0.2246029 reg_l1 26.34427 reg_l2 11.898184
loss 2.85903
STEP 734 ================================
prereg loss 0.22397633 reg_l1 26.341177 reg_l2 11.897616
loss 2.8580942
STEP 735 ================================
prereg loss 0.22267933 reg_l1 26.338217 reg_l2 11.896983
loss 2.856501
STEP 736 ================================
prereg loss 0.2230994 reg_l1 26.334803 reg_l2 11.896346
loss 2.8565798
STEP 737 ================================
prereg loss 0.22372973 reg_l1 26.331434 reg_l2 11.895814
loss 2.856873
STEP 738 ================================
prereg loss 0.22268887 reg_l1 26.328327 reg_l2 11.895246
loss 2.8555217
STEP 739 ================================
prereg loss 0.22215778 reg_l1 26.325216 reg_l2 11.894649
loss 2.8546793
STEP 740 ================================
prereg loss 0.22303587 reg_l1 26.32171 reg_l2 11.894017
loss 2.855207
STEP 741 ================================
prereg loss 0.2233532 reg_l1 26.318333 reg_l2 11.893314
loss 2.8551865
cutoff 0.0031661922 network size 144
STEP 742 ================================
prereg loss 0.2219054 reg_l1 26.31201 reg_l2 11.892651
loss 2.8531065
STEP 743 ================================
prereg loss 0.22166018 reg_l1 26.30882 reg_l2 11.892029
loss 2.8525422
STEP 744 ================================
prereg loss 0.22265694 reg_l1 26.305325 reg_l2 11.891432
loss 2.8531895
STEP 745 ================================
prereg loss 0.22252233 reg_l1 26.302084 reg_l2 11.890778
loss 2.8527308
STEP 746 ================================
prereg loss 0.22116256 reg_l1 26.299015 reg_l2 11.890049
loss 2.8510642
STEP 747 ================================
prereg loss 0.22094864 reg_l1 26.29571 reg_l2 11.889339
loss 2.8505197
STEP 748 ================================
prereg loss 0.2216281 reg_l1 26.29223 reg_l2 11.888714
loss 2.8508513
STEP 749 ================================
prereg loss 0.22159305 reg_l1 26.28903 reg_l2 11.888163
loss 2.8504963
STEP 750 ================================
prereg loss 0.22053464 reg_l1 26.286106 reg_l2 11.88771
loss 2.8491452
STEP 751 ================================
prereg loss 0.22014144 reg_l1 26.28305 reg_l2 11.887257
loss 2.8484466
STEP 752 ================================
prereg loss 0.22138503 reg_l1 26.279436 reg_l2 11.886566
loss 2.8493288
STEP 753 ================================
prereg loss 0.22129959 reg_l1 26.275833 reg_l2 11.885651
loss 2.848883
STEP 754 ================================
prereg loss 0.21992204 reg_l1 26.272411 reg_l2 11.884698
loss 2.8471632
STEP 755 ================================
prereg loss 0.21992067 reg_l1 26.268991 reg_l2 11.883979
loss 2.8468199
STEP 756 ================================
prereg loss 0.22129346 reg_l1 26.265467 reg_l2 11.883531
loss 2.84784
STEP 757 ================================
prereg loss 0.22083858 reg_l1 26.262383 reg_l2 11.883152
loss 2.847077
STEP 758 ================================
prereg loss 0.21929397 reg_l1 26.259438 reg_l2 11.882669
loss 2.845238
STEP 759 ================================
prereg loss 0.21915045 reg_l1 26.256044 reg_l2 11.882074
loss 2.844755
STEP 760 ================================
prereg loss 0.2201585 reg_l1 26.252367 reg_l2 11.881456
loss 2.8453953
STEP 761 ================================
prereg loss 0.22000574 reg_l1 26.249016 reg_l2 11.88088
loss 2.8449073
cutoff 0.003159137 network size 143
STEP 762 ================================
prereg loss 0.21865916 reg_l1 26.242903 reg_l2 11.880494
loss 2.8429494
STEP 763 ================================
prereg loss 0.218797 reg_l1 26.239727 reg_l2 11.880137
loss 2.8427696
STEP 764 ================================
prereg loss 0.21929093 reg_l1 26.236397 reg_l2 11.879648
loss 2.8429308
STEP 765 ================================
prereg loss 0.21863376 reg_l1 26.233025 reg_l2 11.878971
loss 2.841936
STEP 766 ================================
prereg loss 0.21809155 reg_l1 26.229622 reg_l2 11.8782425
loss 2.8410537
STEP 767 ================================
prereg loss 0.21810974 reg_l1 26.226187 reg_l2 11.877652
loss 2.8407285
STEP 768 ================================
prereg loss 0.21830624 reg_l1 26.222885 reg_l2 11.877231
loss 2.8405948
STEP 769 ================================
prereg loss 0.21797231 reg_l1 26.219763 reg_l2 11.876846
loss 2.8399487
STEP 770 ================================
prereg loss 0.21749146 reg_l1 26.21661 reg_l2 11.876389
loss 2.8391523
STEP 771 ================================
prereg loss 0.21741769 reg_l1 26.21328 reg_l2 11.8758545
loss 2.8387458
STEP 772 ================================
prereg loss 0.21758875 reg_l1 26.209888 reg_l2 11.875305
loss 2.8385775
STEP 773 ================================
prereg loss 0.217336 reg_l1 26.206678 reg_l2 11.874827
loss 2.8380039
STEP 774 ================================
prereg loss 0.21688592 reg_l1 26.203651 reg_l2 11.874449
loss 2.8372512
STEP 775 ================================
prereg loss 0.2168011 reg_l1 26.200579 reg_l2 11.874142
loss 2.836859
STEP 776 ================================
prereg loss 0.21702576 reg_l1 26.1974 reg_l2 11.873786
loss 2.8367658
STEP 777 ================================
prereg loss 0.2168214 reg_l1 26.194178 reg_l2 11.873335
loss 2.8362393
STEP 778 ================================
prereg loss 0.21654524 reg_l1 26.190943 reg_l2 11.872905
loss 2.8356397
STEP 779 ================================
prereg loss 0.21640475 reg_l1 26.18775 reg_l2 11.872577
loss 2.8351798
STEP 780 ================================
prereg loss 0.21633855 reg_l1 26.184631 reg_l2 11.8723545
loss 2.834802
STEP 781 ================================
prereg loss 0.2162105 reg_l1 26.18158 reg_l2 11.872204
loss 2.8343687
cutoff 0.0010109781 network size 142
STEP 782 ================================
prereg loss 0.21603256 reg_l1 26.17753 reg_l2 11.872059
loss 2.8337855
STEP 783 ================================
prereg loss 0.21583995 reg_l1 26.174593 reg_l2 11.871913
loss 2.8332992
STEP 784 ================================
prereg loss 0.21571715 reg_l1 26.171642 reg_l2 11.871792
loss 2.8328815
STEP 785 ================================
prereg loss 0.21556123 reg_l1 26.16876 reg_l2 11.871735
loss 2.8324373
STEP 786 ================================
prereg loss 0.21534035 reg_l1 26.16597 reg_l2 11.87175
loss 2.8319373
STEP 787 ================================
prereg loss 0.2151605 reg_l1 26.163208 reg_l2 11.871813
loss 2.8314815
STEP 788 ================================
prereg loss 0.21503657 reg_l1 26.160418 reg_l2 11.871881
loss 2.8310785
STEP 789 ================================
prereg loss 0.21486886 reg_l1 26.157608 reg_l2 11.871925
loss 2.8306296
STEP 790 ================================
prereg loss 0.21564788 reg_l1 26.154613 reg_l2 11.871914
loss 2.8311093
STEP 791 ================================
prereg loss 0.21470192 reg_l1 26.151848 reg_l2 11.871865
loss 2.8298867
STEP 792 ================================
prereg loss 0.21440133 reg_l1 26.149084 reg_l2 11.871858
loss 2.8293097
STEP 793 ================================
prereg loss 0.21513192 reg_l1 26.146029 reg_l2 11.871909
loss 2.8297348
STEP 794 ================================
prereg loss 0.21531458 reg_l1 26.143171 reg_l2 11.871993
loss 2.8296318
STEP 795 ================================
prereg loss 0.21404266 reg_l1 26.140503 reg_l2 11.872071
loss 2.828093
STEP 796 ================================
prereg loss 0.21377768 reg_l1 26.137705 reg_l2 11.872181
loss 2.8275483
STEP 797 ================================
prereg loss 0.21528879 reg_l1 26.134586 reg_l2 11.872313
loss 2.8287475
STEP 798 ================================
prereg loss 0.21488447 reg_l1 26.13177 reg_l2 11.872448
loss 2.8280616
STEP 799 ================================
prereg loss 0.21329762 reg_l1 26.129177 reg_l2 11.872595
loss 2.8262153
STEP 800 ================================
prereg loss 0.21359569 reg_l1 26.126232 reg_l2 11.87278
loss 2.8262188
STEP 801 ================================
prereg loss 0.21425824 reg_l1 26.123283 reg_l2 11.872985
loss 2.8265865
cutoff 0.0012269963 network size 141
STEP 802 ================================
prereg loss 0.21343656 reg_l1 26.119284 reg_l2 11.873172
loss 2.825365
STEP 803 ================================
prereg loss 0.21267411 reg_l1 26.116623 reg_l2 11.873365
loss 2.8243365
STEP 804 ================================
prereg loss 0.21265592 reg_l1 26.113796 reg_l2 11.873594
loss 2.8240356
STEP 805 ================================
prereg loss 0.21312502 reg_l1 26.11092 reg_l2 11.873839
loss 2.824217
STEP 806 ================================
prereg loss 0.21257848 reg_l1 26.108162 reg_l2 11.87397
loss 2.8233948
STEP 807 ================================
prereg loss 0.21201707 reg_l1 26.105438 reg_l2 11.87401
loss 2.822561
STEP 808 ================================
prereg loss 0.21234705 reg_l1 26.102432 reg_l2 11.87397
loss 2.8225904
STEP 809 ================================
prereg loss 0.212507 reg_l1 26.09948 reg_l2 11.87386
loss 2.822455
STEP 810 ================================
prereg loss 0.21178249 reg_l1 26.096601 reg_l2 11.873667
loss 2.8214426
STEP 811 ================================
prereg loss 0.21154188 reg_l1 26.093641 reg_l2 11.873415
loss 2.8209062
STEP 812 ================================
prereg loss 0.2126472 reg_l1 26.090378 reg_l2 11.873102
loss 2.821685
STEP 813 ================================
prereg loss 0.21239015 reg_l1 26.087307 reg_l2 11.872737
loss 2.821121
STEP 814 ================================
prereg loss 0.2112257 reg_l1 26.08435 reg_l2 11.872323
loss 2.819661
STEP 815 ================================
prereg loss 0.21114008 reg_l1 26.081236 reg_l2 11.871926
loss 2.8192637
STEP 816 ================================
prereg loss 0.21284516 reg_l1 26.077843 reg_l2 11.871532
loss 2.8206294
STEP 817 ================================
prereg loss 0.21215945 reg_l1 26.074814 reg_l2 11.8711195
loss 2.8196409
STEP 818 ================================
prereg loss 0.21074663 reg_l1 26.071968 reg_l2 11.870671
loss 2.8179433
STEP 819 ================================
prereg loss 0.21096234 reg_l1 26.068663 reg_l2 11.870164
loss 2.8178287
STEP 820 ================================
prereg loss 0.21179284 reg_l1 26.065287 reg_l2 11.869626
loss 2.8183217
STEP 821 ================================
prereg loss 0.21087365 reg_l1 26.062132 reg_l2 11.869083
loss 2.817087
cutoff 0.0026361353 network size 140
STEP 822 ================================
prereg loss 0.21035974 reg_l1 26.056404 reg_l2 11.868541
loss 2.8160002
STEP 823 ================================
prereg loss 0.21070084 reg_l1 26.052996 reg_l2 11.868002
loss 2.8160005
STEP 824 ================================
prereg loss 0.2110476 reg_l1 26.049633 reg_l2 11.867406
loss 2.816011
STEP 825 ================================
prereg loss 0.20997731 reg_l1 26.04637 reg_l2 11.866714
loss 2.8146143
STEP 826 ================================
prereg loss 0.21018192 reg_l1 26.04282 reg_l2 11.865939
loss 2.814464
STEP 827 ================================
prereg loss 0.21007106 reg_l1 26.039314 reg_l2 11.865168
loss 2.8140025
STEP 828 ================================
prereg loss 0.20971856 reg_l1 26.03589 reg_l2 11.864437
loss 2.8133075
STEP 829 ================================
prereg loss 0.21041645 reg_l1 26.032299 reg_l2 11.863701
loss 2.8136466
STEP 830 ================================
prereg loss 0.20993528 reg_l1 26.028866 reg_l2 11.86295
loss 2.8128219
STEP 831 ================================
prereg loss 0.20922792 reg_l1 26.025412 reg_l2 11.862155
loss 2.811769
STEP 832 ================================
prereg loss 0.2099969 reg_l1 26.021626 reg_l2 11.861325
loss 2.8121595
STEP 833 ================================
prereg loss 0.20985284 reg_l1 26.018017 reg_l2 11.860511
loss 2.8116546
STEP 834 ================================
prereg loss 0.20888756 reg_l1 26.014591 reg_l2 11.859759
loss 2.8103468
STEP 835 ================================
prereg loss 0.20925805 reg_l1 26.010933 reg_l2 11.859011
loss 2.8103514
STEP 836 ================================
prereg loss 0.20927745 reg_l1 26.007326 reg_l2 11.858267
loss 2.81001
STEP 837 ================================
prereg loss 0.20852146 reg_l1 26.003792 reg_l2 11.857498
loss 2.8089006
STEP 838 ================================
prereg loss 0.2082599 reg_l1 26.000183 reg_l2 11.856719
loss 2.808278
STEP 839 ================================
prereg loss 0.20916484 reg_l1 25.996315 reg_l2 11.855932
loss 2.8087964
STEP 840 ================================
prereg loss 0.20860235 reg_l1 25.992697 reg_l2 11.855138
loss 2.807872
STEP 841 ================================
prereg loss 0.20773554 reg_l1 25.98917 reg_l2 11.854325
loss 2.8066525
cutoff 0.0019525698 network size 139
STEP 842 ================================
prereg loss 0.20808046 reg_l1 25.983303 reg_l2 11.853478
loss 2.8064108
STEP 843 ================================
prereg loss 0.20839742 reg_l1 25.979471 reg_l2 11.852618
loss 2.8063445
STEP 844 ================================
prereg loss 0.20747615 reg_l1 25.975801 reg_l2 11.851751
loss 2.8050563
STEP 845 ================================
prereg loss 0.2072393 reg_l1 25.97214 reg_l2 11.850924
loss 2.8044534
STEP 846 ================================
prereg loss 0.20812985 reg_l1 25.968172 reg_l2 11.850103
loss 2.8049471
STEP 847 ================================
prereg loss 0.2080002 reg_l1 25.96441 reg_l2 11.849265
loss 2.8044412
STEP 848 ================================
prereg loss 0.20675443 reg_l1 25.960817 reg_l2 11.848401
loss 2.8028362
STEP 849 ================================
prereg loss 0.20708592 reg_l1 25.956882 reg_l2 11.847512
loss 2.8027742
STEP 850 ================================
prereg loss 0.20731018 reg_l1 25.953022 reg_l2 11.846666
loss 2.8026125
STEP 851 ================================
prereg loss 0.20660526 reg_l1 25.949366 reg_l2 11.845856
loss 2.8015418
STEP 852 ================================
prereg loss 0.20622046 reg_l1 25.945717 reg_l2 11.845097
loss 2.8007922
STEP 853 ================================
prereg loss 0.2072968 reg_l1 25.94177 reg_l2 11.844295
loss 2.8014739
STEP 854 ================================
prereg loss 0.20727088 reg_l1 25.938002 reg_l2 11.843543
loss 2.8010712
STEP 855 ================================
prereg loss 0.20593657 reg_l1 25.934402 reg_l2 11.842817
loss 2.799377
STEP 856 ================================
prereg loss 0.20578565 reg_l1 25.930676 reg_l2 11.842069
loss 2.7988534
STEP 857 ================================
prereg loss 0.20660885 reg_l1 25.926636 reg_l2 11.84132
loss 2.7992723
STEP 858 ================================
prereg loss 0.20694397 reg_l1 25.922882 reg_l2 11.840676
loss 2.7992322
STEP 859 ================================
prereg loss 0.20546953 reg_l1 25.9194 reg_l2 11.840083
loss 2.7974095
STEP 860 ================================
prereg loss 0.20511721 reg_l1 25.91576 reg_l2 11.839482
loss 2.7966933
STEP 861 ================================
prereg loss 0.20661837 reg_l1 25.911655 reg_l2 11.838837
loss 2.7977839
cutoff 0.0024012432 network size 138
STEP 862 ================================
prereg loss 0.20659237 reg_l1 25.905474 reg_l2 11.838222
loss 2.7971396
STEP 863 ================================
prereg loss 0.20487453 reg_l1 25.902039 reg_l2 11.83768
loss 2.7950785
STEP 864 ================================
prereg loss 0.2046861 reg_l1 25.898438 reg_l2 11.837202
loss 2.79453
STEP 865 ================================
prereg loss 0.20642455 reg_l1 25.894436 reg_l2 11.836716
loss 2.7958682
STEP 866 ================================
prereg loss 0.2065249 reg_l1 25.890762 reg_l2 11.8362465
loss 2.7956011
STEP 867 ================================
prereg loss 0.20434621 reg_l1 25.887402 reg_l2 11.83577
loss 2.7930863
STEP 868 ================================
prereg loss 0.20415041 reg_l1 25.883802 reg_l2 11.835295
loss 2.7925308
STEP 869 ================================
prereg loss 0.20571588 reg_l1 25.879753 reg_l2 11.834813
loss 2.7936912
STEP 870 ================================
prereg loss 0.20600288 reg_l1 25.876091 reg_l2 11.834377
loss 2.793612
STEP 871 ================================
prereg loss 0.20352307 reg_l1 25.872887 reg_l2 11.833971
loss 2.7908118
STEP 872 ================================
prereg loss 0.20347485 reg_l1 25.869293 reg_l2 11.8335285
loss 2.790404
STEP 873 ================================
prereg loss 0.20542061 reg_l1 25.865145 reg_l2 11.833018
loss 2.791935
STEP 874 ================================
prereg loss 0.20569707 reg_l1 25.861397 reg_l2 11.832494
loss 2.7918367
STEP 875 ================================
prereg loss 0.20284466 reg_l1 25.85813 reg_l2 11.831948
loss 2.7886577
STEP 876 ================================
prereg loss 0.20261846 reg_l1 25.854239 reg_l2 11.831322
loss 2.7880423
STEP 877 ================================
prereg loss 0.20418179 reg_l1 25.850023 reg_l2 11.830657
loss 2.789184
STEP 878 ================================
prereg loss 0.20411731 reg_l1 25.84612 reg_l2 11.829993
loss 2.7887294
STEP 879 ================================
prereg loss 0.20222796 reg_l1 25.842524 reg_l2 11.829315
loss 2.7864804
STEP 880 ================================
prereg loss 0.20192713 reg_l1 25.838642 reg_l2 11.828603
loss 2.7857914
STEP 881 ================================
prereg loss 0.20267294 reg_l1 25.834423 reg_l2 11.827854
loss 2.7861154
cutoff 0.0028227323 network size 137
STEP 882 ================================
prereg loss 0.20327072 reg_l1 25.827536 reg_l2 11.827103
loss 2.7860243
STEP 883 ================================
prereg loss 0.20140016 reg_l1 25.82392 reg_l2 11.8263855
loss 2.783792
STEP 884 ================================
prereg loss 0.20111057 reg_l1 25.820103 reg_l2 11.825625
loss 2.7831209
STEP 885 ================================
prereg loss 0.20266518 reg_l1 25.81576 reg_l2 11.824758
loss 2.7842412
STEP 886 ================================
prereg loss 0.20202419 reg_l1 25.811754 reg_l2 11.823848
loss 2.7831998
STEP 887 ================================
prereg loss 0.20053442 reg_l1 25.80789 reg_l2 11.822885
loss 2.7813234
STEP 888 ================================
prereg loss 0.20038475 reg_l1 25.80374 reg_l2 11.821863
loss 2.7807589
STEP 889 ================================
prereg loss 0.20232062 reg_l1 25.799181 reg_l2 11.820786
loss 2.7822387
STEP 890 ================================
prereg loss 0.20109895 reg_l1 25.795082 reg_l2 11.819722
loss 2.7806072
STEP 891 ================================
prereg loss 0.19999127 reg_l1 25.791142 reg_l2 11.818629
loss 2.7791054
STEP 892 ================================
prereg loss 0.20015495 reg_l1 25.786596 reg_l2 11.817443
loss 2.7788148
STEP 893 ================================
prereg loss 0.20182583 reg_l1 25.78203 reg_l2 11.816221
loss 2.7800288
STEP 894 ================================
prereg loss 0.19975966 reg_l1 25.777933 reg_l2 11.81504
loss 2.777553
STEP 895 ================================
prereg loss 0.19921939 reg_l1 25.773764 reg_l2 11.813841
loss 2.7765958
STEP 896 ================================
prereg loss 0.20038818 reg_l1 25.769083 reg_l2 11.812574
loss 2.7772965
STEP 897 ================================
prereg loss 0.20032723 reg_l1 25.76465 reg_l2 11.811303
loss 2.7767923
STEP 898 ================================
prereg loss 0.19873512 reg_l1 25.760496 reg_l2 11.810026
loss 2.7747846
STEP 899 ================================
prereg loss 0.19854674 reg_l1 25.755997 reg_l2 11.808702
loss 2.7741463
STEP 900 ================================
prereg loss 0.1993462 reg_l1 25.751263 reg_l2 11.80735
loss 2.7744727
STEP 901 ================================
prereg loss 0.19900216 reg_l1 25.746777 reg_l2 11.805998
loss 2.77368
cutoff 0.003134674 network size 136
STEP 902 ================================
prereg loss 0.19807068 reg_l1 25.73924 reg_l2 11.804619
loss 2.7719948
STEP 903 ================================
prereg loss 0.19785638 reg_l1 25.734646 reg_l2 11.80318
loss 2.771321
STEP 904 ================================
prereg loss 0.19827469 reg_l1 25.729836 reg_l2 11.801678
loss 2.771258
STEP 905 ================================
prereg loss 0.19833896 reg_l1 25.725115 reg_l2 11.800161
loss 2.7708504
STEP 906 ================================
prereg loss 0.19730793 reg_l1 25.720604 reg_l2 11.798655
loss 2.7693682
STEP 907 ================================
prereg loss 0.19714913 reg_l1 25.7159 reg_l2 11.797132
loss 2.7687392
STEP 908 ================================
prereg loss 0.19782974 reg_l1 25.710999 reg_l2 11.7955885
loss 2.7689297
STEP 909 ================================
prereg loss 0.19766447 reg_l1 25.706297 reg_l2 11.794153
loss 2.7682943
STEP 910 ================================
prereg loss 0.19660884 reg_l1 25.701782 reg_l2 11.792809
loss 2.766787
STEP 911 ================================
prereg loss 0.19649565 reg_l1 25.697075 reg_l2 11.791486
loss 2.7662034
STEP 912 ================================
prereg loss 0.19725065 reg_l1 25.69222 reg_l2 11.79017
loss 2.7664728
STEP 913 ================================
prereg loss 0.19653548 reg_l1 25.687632 reg_l2 11.788919
loss 2.7652988
STEP 914 ================================
prereg loss 0.19582216 reg_l1 25.683102 reg_l2 11.787707
loss 2.7641325
STEP 915 ================================
prereg loss 0.19691615 reg_l1 25.678196 reg_l2 11.786464
loss 2.7647357
STEP 916 ================================
prereg loss 0.19656584 reg_l1 25.673565 reg_l2 11.785277
loss 2.7639225
STEP 917 ================================
prereg loss 0.19535215 reg_l1 25.669104 reg_l2 11.784126
loss 2.7622626
STEP 918 ================================
prereg loss 0.1953866 reg_l1 25.664328 reg_l2 11.782923
loss 2.7618194
STEP 919 ================================
prereg loss 0.19601507 reg_l1 25.65947 reg_l2 11.781743
loss 2.7619622
STEP 920 ================================
prereg loss 0.1955421 reg_l1 25.654888 reg_l2 11.78064
loss 2.761031
STEP 921 ================================
prereg loss 0.19472861 reg_l1 25.650438 reg_l2 11.779599
loss 2.7597725
cutoff 0.0029746061 network size 135
STEP 922 ================================
prereg loss 0.19992387 reg_l1 25.642843 reg_l2 11.778521
loss 2.7642083
STEP 923 ================================
prereg loss 0.19767228 reg_l1 25.640423 reg_l2 11.780554
loss 2.7617147
STEP 924 ================================
prereg loss 0.19752765 reg_l1 25.639389 reg_l2 11.783477
loss 2.7614665
STEP 925 ================================
prereg loss 0.19727825 reg_l1 25.637312 reg_l2 11.784507
loss 2.7610095
STEP 926 ================================
prereg loss 0.19599833 reg_l1 25.631739 reg_l2 11.782318
loss 2.7591724
STEP 927 ================================
prereg loss 0.19942561 reg_l1 25.62384 reg_l2 11.778118
loss 2.7618098
STEP 928 ================================
prereg loss 0.19781041 reg_l1 25.616909 reg_l2 11.774592
loss 2.7595015
STEP 929 ================================
prereg loss 0.20027238 reg_l1 25.612825 reg_l2 11.773623
loss 2.761555
STEP 930 ================================
prereg loss 0.19724289 reg_l1 25.610783 reg_l2 11.775048
loss 2.7583213
STEP 931 ================================
prereg loss 0.19424765 reg_l1 25.608833 reg_l2 11.776861
loss 2.755131
STEP 932 ================================
prereg loss 0.20237134 reg_l1 25.60532 reg_l2 11.776847
loss 2.7629035
STEP 933 ================================
prereg loss 0.19731413 reg_l1 25.599426 reg_l2 11.774278
loss 2.7572567
STEP 934 ================================
prereg loss 0.19233078 reg_l1 25.591785 reg_l2 11.77028
loss 2.7515094
STEP 935 ================================
prereg loss 0.19698408 reg_l1 25.58451 reg_l2 11.766993
loss 2.7554352
STEP 936 ================================
prereg loss 0.19692966 reg_l1 25.579672 reg_l2 11.765949
loss 2.7548969
STEP 937 ================================
prereg loss 0.19208628 reg_l1 25.577166 reg_l2 11.766825
loss 2.7498028
STEP 938 ================================
prereg loss 0.19390863 reg_l1 25.574783 reg_l2 11.767856
loss 2.7513871
STEP 939 ================================
prereg loss 0.19573973 reg_l1 25.570639 reg_l2 11.767366
loss 2.7528036
STEP 940 ================================
prereg loss 0.19226533 reg_l1 25.564714 reg_l2 11.765062
loss 2.7487369
STEP 941 ================================
prereg loss 0.19122723 reg_l1 25.558327 reg_l2 11.76198
loss 2.7470598
cutoff 0.004990911 network size 134
STEP 942 ================================
prereg loss 0.19239862 reg_l1 25.547459 reg_l2 11.759435
loss 2.7471445
STEP 943 ================================
prereg loss 0.19112644 reg_l1 25.542492 reg_l2 11.758192
loss 2.7453756
STEP 944 ================================
prereg loss 0.19207056 reg_l1 25.538565 reg_l2 11.757942
loss 2.745927
STEP 945 ================================
prereg loss 0.19001672 reg_l1 25.534632 reg_l2 11.757626
loss 2.74348
STEP 946 ================================
prereg loss 0.18876776 reg_l1 25.529482 reg_l2 11.756254
loss 2.741716
STEP 947 ================================
prereg loss 0.18881033 reg_l1 25.523079 reg_l2 11.753715
loss 2.7411182
STEP 948 ================================
prereg loss 0.18893522 reg_l1 25.516504 reg_l2 11.750757
loss 2.7405858
STEP 949 ================================
prereg loss 0.18866959 reg_l1 25.510601 reg_l2 11.748256
loss 2.73973
STEP 950 ================================
prereg loss 0.18825687 reg_l1 25.505436 reg_l2 11.746613
loss 2.7388005
STEP 951 ================================
prereg loss 0.18791114 reg_l1 25.500652 reg_l2 11.745579
loss 2.7379763
STEP 952 ================================
prereg loss 0.18781318 reg_l1 25.495932 reg_l2 11.744585
loss 2.7374065
STEP 953 ================================
prereg loss 0.18721709 reg_l1 25.49083 reg_l2 11.743047
loss 2.7363
STEP 954 ================================
prereg loss 0.18685788 reg_l1 25.48496 reg_l2 11.740764
loss 2.735354
STEP 955 ================================
prereg loss 0.18746121 reg_l1 25.478485 reg_l2 11.73804
loss 2.7353096
STEP 956 ================================
prereg loss 0.18761016 reg_l1 25.472214 reg_l2 11.735536
loss 2.7348316
STEP 957 ================================
prereg loss 0.18673903 reg_l1 25.466692 reg_l2 11.7336645
loss 2.7334082
STEP 958 ================================
prereg loss 0.18631275 reg_l1 25.461763 reg_l2 11.732339
loss 2.732489
STEP 959 ================================
prereg loss 0.18682024 reg_l1 25.456686 reg_l2 11.731006
loss 2.7324889
STEP 960 ================================
prereg loss 0.18726064 reg_l1 25.451435 reg_l2 11.729271
loss 2.7324042
STEP 961 ================================
prereg loss 0.18615717 reg_l1 25.445704 reg_l2 11.726957
loss 2.7307277
cutoff 0.0051257475 network size 133
STEP 962 ================================
prereg loss 0.18574509 reg_l1 25.434294 reg_l2 11.724271
loss 2.7291744
STEP 963 ================================
prereg loss 0.18636732 reg_l1 25.428082 reg_l2 11.7217865
loss 2.7291756
STEP 964 ================================
prereg loss 0.1861335 reg_l1 25.422535 reg_l2 11.71987
loss 2.7283869
STEP 965 ================================
prereg loss 0.18536763 reg_l1 25.417536 reg_l2 11.718397
loss 2.727121
STEP 966 ================================
prereg loss 0.18544444 reg_l1 25.412397 reg_l2 11.7169
loss 2.726684
STEP 967 ================================
prereg loss 0.18578804 reg_l1 25.406769 reg_l2 11.71501
loss 2.7264647
STEP 968 ================================
prereg loss 0.18533225 reg_l1 25.40087 reg_l2 11.712724
loss 2.7254193
STEP 969 ================================
prereg loss 0.1847563 reg_l1 25.394966 reg_l2 11.710289
loss 2.724253
STEP 970 ================================
prereg loss 0.18464087 reg_l1 25.389145 reg_l2 11.70799
loss 2.7235553
STEP 971 ================================
prereg loss 0.18495479 reg_l1 25.38353 reg_l2 11.70599
loss 2.7233078
STEP 972 ================================
prereg loss 0.18500762 reg_l1 25.378185 reg_l2 11.704237
loss 2.7228262
STEP 973 ================================
prereg loss 0.1843254 reg_l1 25.372887 reg_l2 11.702481
loss 2.7216141
STEP 974 ================================
prereg loss 0.183996 reg_l1 25.367258 reg_l2 11.700471
loss 2.7207217
STEP 975 ================================
prereg loss 0.18414725 reg_l1 25.36127 reg_l2 11.698176
loss 2.7202744
STEP 976 ================================
prereg loss 0.18430677 reg_l1 25.355242 reg_l2 11.695806
loss 2.719831
STEP 977 ================================
prereg loss 0.18399303 reg_l1 25.349438 reg_l2 11.6935835
loss 2.718937
STEP 978 ================================
prereg loss 0.18367907 reg_l1 25.343863 reg_l2 11.6916
loss 2.7180655
STEP 979 ================================
prereg loss 0.18364988 reg_l1 25.338387 reg_l2 11.689793
loss 2.7174885
STEP 980 ================================
prereg loss 0.18361965 reg_l1 25.332964 reg_l2 11.688016
loss 2.716916
STEP 981 ================================
prereg loss 0.18332084 reg_l1 25.327505 reg_l2 11.686132
loss 2.7160714
cutoff 0.0045455024 network size 132
STEP 982 ================================
prereg loss 0.18337424 reg_l1 25.317305 reg_l2 11.684054
loss 2.7151046
STEP 983 ================================
prereg loss 0.18338281 reg_l1 25.311543 reg_l2 11.681913
loss 2.7145371
STEP 984 ================================
prereg loss 0.18336189 reg_l1 25.305822 reg_l2 11.679865
loss 2.713944
STEP 985 ================================
prereg loss 0.18309073 reg_l1 25.300318 reg_l2 11.678001
loss 2.7131226
STEP 986 ================================
prereg loss 0.1828305 reg_l1 25.294954 reg_l2 11.676292
loss 2.712326
STEP 987 ================================
prereg loss 0.18284422 reg_l1 25.289553 reg_l2 11.674578
loss 2.7117994
STEP 988 ================================
prereg loss 0.18296623 reg_l1 25.284063 reg_l2 11.672765
loss 2.7113726
STEP 989 ================================
prereg loss 0.18279217 reg_l1 25.27848 reg_l2 11.670847
loss 2.7106402
STEP 990 ================================
prereg loss 0.18254045 reg_l1 25.27288 reg_l2 11.668905
loss 2.7098286
STEP 991 ================================
prereg loss 0.1824983 reg_l1 25.267315 reg_l2 11.667037
loss 2.7092297
STEP 992 ================================
prereg loss 0.18254592 reg_l1 25.2619 reg_l2 11.665308
loss 2.708736
STEP 993 ================================
prereg loss 0.18247704 reg_l1 25.256586 reg_l2 11.663699
loss 2.7081356
STEP 994 ================================
prereg loss 0.18231453 reg_l1 25.25129 reg_l2 11.662103
loss 2.7074437
STEP 995 ================================
prereg loss 0.1822139 reg_l1 25.245872 reg_l2 11.660434
loss 2.7068014
STEP 996 ================================
prereg loss 0.18217745 reg_l1 25.240368 reg_l2 11.658694
loss 2.7062144
STEP 997 ================================
prereg loss 0.1820779 reg_l1 25.234888 reg_l2 11.656968
loss 2.7055666
STEP 998 ================================
prereg loss 0.1819414 reg_l1 25.229515 reg_l2 11.655318
loss 2.704893
STEP 999 ================================
prereg loss 0.18188249 reg_l1 25.224216 reg_l2 11.653763
loss 2.704304
STEP 1000 ================================
prereg loss 0.18185644 reg_l1 25.218945 reg_l2 11.652271
loss 2.7037508
2022-08-14T20:26:51.034

julia> serialize("cf-s2-132-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-s2-132-parameters-opt.ser", opt)
```

We are going to make another attempt at 50 sparsifications in 1000 steps,
and we'll resume next day.

Continuing here: [Day-2.md](Day-2.md)
