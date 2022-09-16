# Sparsification phase of the compare-free experiment

We are trying a new scheme which lets us to preserve the state of the optimizer while
sparsifying by trimming first and second moment tensors in sync with trimming the parameters.

Then we can in principle do "smooth sparsification", e.g. drop one parameter, do one training step,
drop one parameter, do one training step, etc. Here are the first 200 steps, it works OK at the beginning,
but then less so:

```
julia> cd("Desktop/GitHub/sandbox_mishka/arxiv-1606-09470-section3/v0-1/compare-free/sparsify")

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

julia> sparsifying_steps!(5)
2022-07-19T19:36:03.874
STEP 1 ================================
prereg loss 0.10171912 reg_l1 85.602264 reg_l2 26.10721
loss 0.1873214
cutoff 2.6965774e-9 network size 941
STEP 2 ================================
prereg loss 0.10161547 reg_l1 85.601875 reg_l2 26.108368
loss 0.18721735
cutoff 1.1257072e-7 network size 940
STEP 3 ================================
prereg loss 0.10137128 reg_l1 85.601395 reg_l2 26.109524
loss 0.18697268
cutoff 3.0793765e-7 network size 939
STEP 4 ================================
prereg loss 0.10104765 reg_l1 85.60012 reg_l2 26.110575
loss 0.18664777
cutoff 9.533628e-8 network size 938
STEP 5 ================================
prereg loss 0.10092122 reg_l1 85.59957 reg_l2 26.11157
loss 0.1865208
cutoff 3.1686199e-7 network size 937
2022-07-19T19:39:14.933

julia> sparsifying_steps!(5)
2022-07-19T19:39:18.193
STEP 1 ================================
prereg loss 0.1006564 reg_l1 85.59898 reg_l2 26.112623
loss 0.1862554
cutoff 2.9748526e-7 network size 936
STEP 2 ================================
prereg loss 0.1003873 reg_l1 85.59801 reg_l2 26.11367
loss 0.18598531
cutoff 1.2101913e-7 network size 935
STEP 3 ================================
prereg loss 0.1002721 reg_l1 85.59689 reg_l2 26.114578
loss 0.185869
cutoff 4.6666466e-7 network size 934
STEP 4 ================================
prereg loss 0.099957 reg_l1 85.596054 reg_l2 26.11555
loss 0.18555304
cutoff 7.6414534e-7 network size 933
STEP 5 ================================
prereg loss 0.099782065 reg_l1 85.595245 reg_l2 26.116558
loss 0.18537731
cutoff 5.1858103e-7 network size 932
2022-07-19T19:41:51.687

julia> sparsifying_steps!(90)
2022-07-19T19:42:31.334
STEP 1 ================================
prereg loss 0.099630594 reg_l1 85.59425 reg_l2 26.117603
loss 0.18522486
cutoff 7.5690696e-7 network size 931
STEP 2 ================================
prereg loss 0.09928542 reg_l1 85.59387 reg_l2 26.118792
loss 0.1848793
cutoff 1.4511971e-7 network size 930
STEP 3 ================================
prereg loss 0.09917623 reg_l1 85.59311 reg_l2 26.119925
loss 0.18476933
cutoff 4.2280155e-7 network size 929
STEP 4 ================================
prereg loss 0.09896029 reg_l1 85.59203 reg_l2 26.120958
loss 0.18455233
cutoff 1.993165e-7 network size 928
STEP 5 ================================
prereg loss 0.098685674 reg_l1 85.591225 reg_l2 26.122082
loss 0.18427691
cutoff 9.020679e-7 network size 927
STEP 6 ================================
prereg loss 0.09856509 reg_l1 85.59036 reg_l2 26.123161
loss 0.18415546
cutoff 7.19583e-8 network size 926
STEP 7 ================================
prereg loss 0.098279774 reg_l1 85.58978 reg_l2 26.124203
loss 0.18386957
cutoff 8.271174e-7 network size 925
STEP 8 ================================
prereg loss 0.098097146 reg_l1 85.589325 reg_l2 26.125263
loss 0.18368647
cutoff 2.2637815e-7 network size 924
STEP 9 ================================
prereg loss 0.09791372 reg_l1 85.588554 reg_l2 26.126358
loss 0.18350229
cutoff 1.9750595e-7 network size 923
STEP 10 ================================
prereg loss 0.09761743 reg_l1 85.58777 reg_l2 26.127487
loss 0.1832052
cutoff 2.8418592e-7 network size 922
STEP 11 ================================
prereg loss 0.09747253 reg_l1 85.58749 reg_l2 26.128632
loss 0.18306002
cutoff 7.1793704e-7 network size 921
STEP 12 ================================
prereg loss 0.09729698 reg_l1 85.587265 reg_l2 26.129751
loss 0.18288425
cutoff 7.255087e-7 network size 920
STEP 13 ================================
prereg loss 0.097030886 reg_l1 85.58719 reg_l2 26.13075
loss 0.18261808
cutoff 9.081545e-8 network size 919
STEP 14 ================================
prereg loss 0.09682763 reg_l1 85.587425 reg_l2 26.131882
loss 0.18241507
cutoff 8.964975e-8 network size 918
STEP 15 ================================
prereg loss 0.09665759 reg_l1 85.58668 reg_l2 26.132963
loss 0.18224427
cutoff 1.1031389e-6 network size 917
STEP 16 ================================
prereg loss 0.09642546 reg_l1 85.58508 reg_l2 26.133976
loss 0.18201055
cutoff 4.0333433e-7 network size 916
STEP 17 ================================
prereg loss 0.0961926 reg_l1 85.58406 reg_l2 26.134995
loss 0.18177667
cutoff 3.7710953e-7 network size 915
STEP 18 ================================
prereg loss 0.096016526 reg_l1 85.58407 reg_l2 26.136074
loss 0.1816006
cutoff 1.8191933e-7 network size 914
STEP 19 ================================
prereg loss 0.095808856 reg_l1 85.58356 reg_l2 26.137257
loss 0.18139242
cutoff 2.1916814e-7 network size 913
STEP 20 ================================
prereg loss 0.095600754 reg_l1 85.58325 reg_l2 26.138441
loss 0.18118401
cutoff 5.5636217e-8 network size 912
STEP 21 ================================
prereg loss 0.09539074 reg_l1 85.58258 reg_l2 26.139528
loss 0.18097332
cutoff 1.075811e-6 network size 911
STEP 22 ================================
prereg loss 0.09521096 reg_l1 85.58271 reg_l2 26.140673
loss 0.18079367
cutoff 8.104032e-7 network size 910
STEP 23 ================================
prereg loss 0.09501656 reg_l1 85.58286 reg_l2 26.14187
loss 0.18059942
cutoff 1.1966137e-6 network size 909
STEP 24 ================================
prereg loss 0.09479045 reg_l1 85.58239 reg_l2 26.143114
loss 0.18037283
cutoff 1.3743222e-6 network size 908
STEP 25 ================================
prereg loss 0.09462243 reg_l1 85.58173 reg_l2 26.144302
loss 0.18020417
cutoff 2.2943797e-7 network size 907
STEP 26 ================================
prereg loss 0.09443393 reg_l1 85.580536 reg_l2 26.145493
loss 0.18001448
cutoff 8.5591887e-7 network size 906
STEP 27 ================================
prereg loss 0.09424266 reg_l1 85.57921 reg_l2 26.146667
loss 0.17982188
cutoff 4.1442357e-7 network size 905
STEP 28 ================================
prereg loss 0.094019786 reg_l1 85.577805 reg_l2 26.147757
loss 0.17959759
cutoff 8.331884e-7 network size 904
STEP 29 ================================
prereg loss 0.09386853 reg_l1 85.57667 reg_l2 26.148823
loss 0.1794452
cutoff 5.8640927e-7 network size 903
STEP 30 ================================
prereg loss 0.09366459 reg_l1 85.575874 reg_l2 26.149794
loss 0.17924047
cutoff 6.7525156e-7 network size 902
STEP 31 ================================
prereg loss 0.09346097 reg_l1 85.57429 reg_l2 26.15068
loss 0.17903526
cutoff 2.4265648e-7 network size 901
STEP 32 ================================
prereg loss 0.093293734 reg_l1 85.57326 reg_l2 26.151566
loss 0.178867
cutoff 3.49426e-7 network size 900
STEP 33 ================================
prereg loss 0.093096875 reg_l1 85.57264 reg_l2 26.152617
loss 0.17866951
cutoff 3.5317385e-7 network size 899
STEP 34 ================================
prereg loss 0.09296909 reg_l1 85.57152 reg_l2 26.15382
loss 0.17854062
cutoff 4.1278938e-7 network size 898
STEP 35 ================================
prereg loss 0.09272168 reg_l1 85.571045 reg_l2 26.155046
loss 0.17829272
cutoff 2.9316996e-7 network size 897
STEP 36 ================================
prereg loss 0.092506655 reg_l1 85.57094 reg_l2 26.156336
loss 0.1780776
cutoff 1.6758952e-6 network size 896
STEP 37 ================================
prereg loss 0.09233875 reg_l1 85.57084 reg_l2 26.157665
loss 0.17790958
cutoff 2.6263397e-8 network size 895
STEP 38 ================================
prereg loss 0.092170425 reg_l1 85.57036 reg_l2 26.158901
loss 0.17774078
cutoff 7.314786e-7 network size 894
STEP 39 ================================
prereg loss 0.09193531 reg_l1 85.5694 reg_l2 26.159946
loss 0.17750472
cutoff 1.8773504e-7 network size 893
STEP 40 ================================
prereg loss 0.09174737 reg_l1 85.56771 reg_l2 26.160887
loss 0.17731509
cutoff 6.056023e-7 network size 892
STEP 41 ================================
prereg loss 0.09157573 reg_l1 85.56591 reg_l2 26.161688
loss 0.17714164
cutoff 4.1312467e-7 network size 891
STEP 42 ================================
prereg loss 0.091361865 reg_l1 85.56436 reg_l2 26.162529
loss 0.17692623
cutoff 3.1019118e-7 network size 890
STEP 43 ================================
prereg loss 0.09116023 reg_l1 85.56314 reg_l2 26.163387
loss 0.17672338
cutoff 5.7486258e-8 network size 889
STEP 44 ================================
prereg loss 0.09100145 reg_l1 85.562294 reg_l2 26.164312
loss 0.17656374
cutoff 5.6252225e-7 network size 888
STEP 45 ================================
prereg loss 0.090857655 reg_l1 85.56103 reg_l2 26.16528
loss 0.17641869
cutoff 6.4025696e-9 network size 887
STEP 46 ================================
prereg loss 0.090593405 reg_l1 85.55939 reg_l2 26.166319
loss 0.1761528
cutoff 1.4847991e-7 network size 886
STEP 47 ================================
prereg loss 0.090402536 reg_l1 85.55901 reg_l2 26.167364
loss 0.17596155
cutoff 6.3200434e-7 network size 885
STEP 48 ================================
prereg loss 0.090189904 reg_l1 85.55859 reg_l2 26.168455
loss 0.1757485
cutoff 3.0642497e-7 network size 884
STEP 49 ================================
prereg loss 0.08996753 reg_l1 85.557304 reg_l2 26.169605
loss 0.17552483
cutoff 4.7645486e-7 network size 883
STEP 50 ================================
prereg loss 0.08978277 reg_l1 85.55523 reg_l2 26.170639
loss 0.175338
cutoff 1.3817076e-6 network size 882
STEP 51 ================================
prereg loss 0.089579634 reg_l1 85.55313 reg_l2 26.171581
loss 0.17513277
cutoff 1.8657088e-6 network size 881
STEP 52 ================================
prereg loss 0.08936575 reg_l1 85.55172 reg_l2 26.17243
loss 0.17491747
cutoff 2.2490013e-7 network size 880
STEP 53 ================================
prereg loss 0.08916944 reg_l1 85.55059 reg_l2 26.173231
loss 0.17472003
cutoff 6.0444665e-7 network size 879
STEP 54 ================================
prereg loss 0.08896932 reg_l1 85.54922 reg_l2 26.174068
loss 0.17451854
cutoff 3.2902594e-7 network size 878
STEP 55 ================================
prereg loss 0.08877632 reg_l1 85.547714 reg_l2 26.174976
loss 0.17432404
cutoff 1.3711721e-6 network size 877
STEP 56 ================================
prereg loss 0.08856101 reg_l1 85.54622 reg_l2 26.17597
loss 0.17410724
cutoff 1.6713554e-6 network size 876
STEP 57 ================================
prereg loss 0.08836272 reg_l1 85.54465 reg_l2 26.176899
loss 0.17390737
cutoff 4.7027106e-7 network size 875
STEP 58 ================================
prereg loss 0.08816702 reg_l1 85.54331 reg_l2 26.177786
loss 0.17371035
cutoff 8.453964e-8 network size 874
STEP 59 ================================
prereg loss 0.087987915 reg_l1 85.541695 reg_l2 26.17855
loss 0.17352961
cutoff 7.516387e-7 network size 873
STEP 60 ================================
prereg loss 0.08781992 reg_l1 85.539764 reg_l2 26.179245
loss 0.17335969
cutoff 2.6888956e-6 network size 872
STEP 61 ================================
prereg loss 0.087629296 reg_l1 85.53776 reg_l2 26.180033
loss 0.17316705
cutoff 6.568853e-7 network size 871
STEP 62 ================================
prereg loss 0.08742056 reg_l1 85.536194 reg_l2 26.180832
loss 0.17295676
cutoff 7.9527814e-7 network size 870
STEP 63 ================================
prereg loss 0.08728568 reg_l1 85.53497 reg_l2 26.181648
loss 0.17282066
cutoff 3.1230338e-6 network size 868
STEP 64 ================================
prereg loss 0.0870816 reg_l1 85.53342 reg_l2 26.182512
loss 0.17261502
cutoff 4.5414704e-6 network size 867
STEP 65 ================================
prereg loss 0.08690152 reg_l1 85.53156 reg_l2 26.183182
loss 0.1724331
cutoff 2.1370069e-7 network size 866
STEP 66 ================================
prereg loss 0.08672745 reg_l1 85.52983 reg_l2 26.183867
loss 0.17225727
cutoff 3.3022507e-6 network size 865
STEP 67 ================================
prereg loss 0.08653293 reg_l1 85.52901 reg_l2 26.184639
loss 0.17206195
cutoff 2.3156763e-6 network size 864
STEP 68 ================================
prereg loss 0.08636589 reg_l1 85.52781 reg_l2 26.185417
loss 0.17189372
cutoff 3.2487853e-7 network size 863
STEP 69 ================================
prereg loss 0.08621289 reg_l1 85.52608 reg_l2 26.186161
loss 0.17173897
cutoff 1.8378407e-7 network size 862
STEP 70 ================================
prereg loss 0.086032785 reg_l1 85.52348 reg_l2 26.186785
loss 0.17155626
cutoff 2.140775e-6 network size 861
STEP 71 ================================
prereg loss 0.08583694 reg_l1 85.521126 reg_l2 26.187277
loss 0.17135808
cutoff 7.676764e-7 network size 860
STEP 72 ================================
prereg loss 0.08567297 reg_l1 85.519646 reg_l2 26.187807
loss 0.17119262
cutoff 1.926852e-7 network size 859
STEP 73 ================================
prereg loss 0.08550704 reg_l1 85.51784 reg_l2 26.188374
loss 0.17102489
cutoff 1.1152284e-6 network size 858
STEP 74 ================================
prereg loss 0.085313074 reg_l1 85.515976 reg_l2 26.189087
loss 0.17082906
cutoff 2.0864588e-6 network size 857
STEP 75 ================================
prereg loss 0.08512359 reg_l1 85.51373 reg_l2 26.189795
loss 0.17063732
cutoff 2.0242041e-8 network size 856
STEP 76 ================================
prereg loss 0.08496746 reg_l1 85.511284 reg_l2 26.190512
loss 0.17047875
cutoff 3.888833e-6 network size 855
STEP 77 ================================
prereg loss 0.084789194 reg_l1 85.5096 reg_l2 26.191196
loss 0.17029878
cutoff 2.0103518e-7 network size 854
STEP 78 ================================
prereg loss 0.08462903 reg_l1 85.50802 reg_l2 26.191973
loss 0.17013705
cutoff 6.9280754e-6 network size 853
STEP 79 ================================
prereg loss 0.08442277 reg_l1 85.50678 reg_l2 26.192593
loss 0.16992956
cutoff 2.017501e-5 network size 852
STEP 80 ================================
prereg loss 0.08426621 reg_l1 85.50505 reg_l2 26.193235
loss 0.16977125
cutoff 2.5267132e-6 network size 851
STEP 81 ================================
prereg loss 0.08408444 reg_l1 85.50288 reg_l2 26.19394
loss 0.16958731
cutoff 7.3322267e-6 network size 850
STEP 82 ================================
prereg loss 0.08390375 reg_l1 85.50121 reg_l2 26.194603
loss 0.16940497
cutoff 1.2099534e-6 network size 849
STEP 83 ================================
prereg loss 0.08372074 reg_l1 85.49952 reg_l2 26.195162
loss 0.16922027
cutoff 1.7497442e-6 network size 848
STEP 84 ================================
prereg loss 0.08354971 reg_l1 85.49788 reg_l2 26.195683
loss 0.1690476
cutoff 3.026018e-6 network size 847
STEP 85 ================================
prereg loss 0.08335986 reg_l1 85.49593 reg_l2 26.196196
loss 0.16885579
cutoff 4.381487e-6 network size 846
STEP 86 ================================
prereg loss 0.083185986 reg_l1 85.493454 reg_l2 26.196686
loss 0.16867945
cutoff 8.213643e-6 network size 845
STEP 87 ================================
prereg loss 0.083010636 reg_l1 85.49124 reg_l2 26.197111
loss 0.16850188
cutoff 8.0231086e-7 network size 844
STEP 88 ================================
prereg loss 0.08284328 reg_l1 85.48944 reg_l2 26.197542
loss 0.16833273
cutoff 1.7873637e-5 network size 843
STEP 89 ================================
prereg loss 0.082657605 reg_l1 85.48781 reg_l2 26.198057
loss 0.16814542
cutoff 2.029346e-5 network size 842
STEP 90 ================================
prereg loss 0.08250516 reg_l1 85.48574 reg_l2 26.198599
loss 0.1679909
cutoff 2.2249058e-5 network size 841
2022-07-19T20:28:56.648

julia> sparsifying_steps!(100)
2022-07-19T20:30:43.859
STEP 1 ================================
prereg loss 0.08230745 reg_l1 85.48411 reg_l2 26.199253
loss 0.16779156
cutoff 5.379801e-6 network size 840
STEP 2 ================================
prereg loss 0.08215003 reg_l1 85.48264 reg_l2 26.199833
loss 0.16763267
cutoff 4.887823e-6 network size 839
STEP 3 ================================
prereg loss 0.08198594 reg_l1 85.48086 reg_l2 26.200342
loss 0.1674668
cutoff 3.8678872e-6 network size 838
STEP 4 ================================
prereg loss 0.08181272 reg_l1 85.47892 reg_l2 26.200903
loss 0.16729164
cutoff 9.558477e-7 network size 837
STEP 5 ================================
prereg loss 0.08164721 reg_l1 85.47659 reg_l2 26.201391
loss 0.16712381
cutoff 2.4169012e-5 network size 836
STEP 6 ================================
prereg loss 0.08149984 reg_l1 85.47409 reg_l2 26.201828
loss 0.16697393
cutoff 4.308834e-6 network size 835
STEP 7 ================================
prereg loss 0.081321344 reg_l1 85.47188 reg_l2 26.202251
loss 0.16679323
cutoff 1.3492796e-7 network size 834
STEP 8 ================================
prereg loss 0.0811454 reg_l1 85.46994 reg_l2 26.202732
loss 0.16661534
cutoff 2.8997538e-5 network size 833
STEP 9 ================================
prereg loss 0.0810123 reg_l1 85.46803 reg_l2 26.203175
loss 0.16648033
cutoff 1.1353635e-5 network size 832
STEP 10 ================================
prereg loss 0.08081923 reg_l1 85.46575 reg_l2 26.203714
loss 0.16628498
cutoff 1.9574249e-5 network size 831
STEP 11 ================================
prereg loss 0.08064746 reg_l1 85.46321 reg_l2 26.204277
loss 0.16611068
cutoff 4.7098776e-5 network size 830
STEP 12 ================================
prereg loss 0.08051788 reg_l1 85.46124 reg_l2 26.204773
loss 0.16597912
cutoff 3.1239677e-5 network size 829
STEP 13 ================================
prereg loss 0.080342144 reg_l1 85.4589 reg_l2 26.205187
loss 0.16580105
cutoff 2.6583952e-5 network size 828
STEP 14 ================================
prereg loss 0.08014737 reg_l1 85.4563 reg_l2 26.205542
loss 0.16560367
cutoff 4.7298443e-5 network size 827
STEP 15 ================================
prereg loss 0.079998285 reg_l1 85.45368 reg_l2 26.205873
loss 0.16545197
cutoff 4.2146214e-5 network size 826
STEP 16 ================================
prereg loss 0.07987473 reg_l1 85.45037 reg_l2 26.206202
loss 0.1653251
cutoff 3.273799e-5 network size 825
STEP 17 ================================
prereg loss 0.079726584 reg_l1 85.44774 reg_l2 26.206608
loss 0.16517434
cutoff 1.5501251e-5 network size 824
STEP 18 ================================
prereg loss 0.0795342 reg_l1 85.44568 reg_l2 26.207087
loss 0.16497988
cutoff 2.8246028e-5 network size 823
STEP 19 ================================
prereg loss 0.07936377 reg_l1 85.44375 reg_l2 26.2076
loss 0.16480753
cutoff 8.42404e-5 network size 822
STEP 20 ================================
prereg loss 0.079255305 reg_l1 85.441345 reg_l2 26.208057
loss 0.16469666
cutoff 3.051105e-5 network size 821
STEP 21 ================================
prereg loss 0.07907949 reg_l1 85.43886 reg_l2 26.208527
loss 0.16451836
cutoff 3.9276172e-5 network size 820
STEP 22 ================================
prereg loss 0.07889163 reg_l1 85.43629 reg_l2 26.20893
loss 0.16432792
cutoff 0.0001807393 network size 819
STEP 23 ================================
prereg loss 0.078744076 reg_l1 85.433846 reg_l2 26.209312
loss 0.16417792
cutoff 0.00024340003 network size 818
STEP 24 ================================
prereg loss 0.07864755 reg_l1 85.43141 reg_l2 26.209711
loss 0.16407897
cutoff 0.00025256455 network size 817
STEP 25 ================================
prereg loss 0.078577265 reg_l1 85.429245 reg_l2 26.210106
loss 0.16400652
cutoff 0.00026416057 network size 816
STEP 26 ================================
prereg loss 0.079803325 reg_l1 85.42692 reg_l2 26.210493
loss 0.16523024
cutoff 0.00027014315 network size 815
STEP 27 ================================
prereg loss 0.08150034 reg_l1 85.424545 reg_l2 26.21087
loss 0.1669249
cutoff 0.00028146157 network size 814
STEP 28 ================================
prereg loss 0.07954884 reg_l1 85.42212 reg_l2 26.211386
loss 0.16497096
cutoff 0.00035457133 network size 813
STEP 29 ================================
prereg loss 0.078225896 reg_l1 85.41997 reg_l2 26.211765
loss 0.16364586
cutoff 0.00034071945 network size 812
STEP 30 ================================
prereg loss 0.07864244 reg_l1 85.41705 reg_l2 26.21201
loss 0.16405949
cutoff 0.0004840586 network size 811
STEP 31 ================================
prereg loss 0.07856338 reg_l1 85.41363 reg_l2 26.2123
loss 0.16397701
cutoff 0.0006118385 network size 810
STEP 32 ================================
prereg loss 0.078268334 reg_l1 85.410805 reg_l2 26.212605
loss 0.16367915
cutoff 0.0007186276 network size 808
STEP 33 ================================
prereg loss 0.07913013 reg_l1 85.40707 reg_l2 26.212898
loss 0.16453719
cutoff 0.00071771804 network size 807
STEP 34 ================================
prereg loss 0.3756466 reg_l1 85.40402 reg_l2 26.213276
loss 0.46105063
cutoff 0.0004679954 network size 806
STEP 35 ================================
prereg loss 0.7585806 reg_l1 85.40245 reg_l2 26.213152
loss 0.84398305
cutoff 0.0007930063 network size 805
STEP 36 ================================
prereg loss 0.32527396 reg_l1 85.40614 reg_l2 26.215847
loss 0.41068012
cutoff 0.0007379257 network size 804
STEP 37 ================================
prereg loss 0.6417178 reg_l1 85.40415 reg_l2 26.218872
loss 0.72712195
cutoff 0.000852261 network size 803
STEP 38 ================================
prereg loss 0.33609086 reg_l1 85.39862 reg_l2 26.217579
loss 0.42148948
cutoff 0.0008748865 network size 802
STEP 39 ================================
prereg loss 0.3026037 reg_l1 85.39867 reg_l2 26.218311
loss 0.38800237
cutoff 0.00087345287 network size 801
STEP 40 ================================
prereg loss 0.69909906 reg_l1 85.39693 reg_l2 26.220709
loss 0.784496
cutoff 0.0008598817 network size 800
STEP 41 ================================
prereg loss 0.14916806 reg_l1 85.3892 reg_l2 26.219126
loss 0.23455727
cutoff 0.00092172646 network size 799
STEP 42 ================================
prereg loss 0.38616133 reg_l1 85.38666 reg_l2 26.219133
loss 0.471548
cutoff 0.0012198397 network size 798
STEP 43 ================================
prereg loss 0.3036202 reg_l1 85.38927 reg_l2 26.222204
loss 0.38900948
cutoff 0.0012487477 network size 797
STEP 44 ================================
prereg loss 0.15054801 reg_l1 85.38367 reg_l2 26.222082
loss 0.23593168
cutoff 0.001311187 network size 796
STEP 45 ================================
prereg loss 0.47275135 reg_l1 85.372635 reg_l2 26.220469
loss 0.558124
cutoff 0.0013451396 network size 795
STEP 46 ================================
prereg loss 0.23486136 reg_l1 85.368996 reg_l2 26.221785
loss 0.32023036
cutoff 0.0013280429 network size 794
STEP 47 ================================
prereg loss 0.25335824 reg_l1 85.36555 reg_l2 26.221426
loss 0.33872378
cutoff 0.0017784354 network size 793
STEP 48 ================================
prereg loss 0.2311514 reg_l1 85.355255 reg_l2 26.22024
loss 0.31650665
cutoff 0.0016932173 network size 792
STEP 49 ================================
prereg loss 0.16767986 reg_l1 85.348785 reg_l2 26.221508
loss 0.25302866
cutoff 0.002002943 network size 791
STEP 50 ================================
prereg loss 0.18481229 reg_l1 85.349396 reg_l2 26.223318
loss 0.2701617
cutoff 0.0020017219 network size 790
STEP 51 ================================
prereg loss 0.106031105 reg_l1 85.349556 reg_l2 26.224354
loss 0.19138066
cutoff 0.002162662 network size 789
STEP 52 ================================
prereg loss 0.22312105 reg_l1 85.344666 reg_l2 26.224594
loss 0.30846572
cutoff 0.002214662 network size 787
STEP 53 ================================
prereg loss 8.868364 reg_l1 85.33589 reg_l2 26.224953
loss 8.9537
cutoff 0.0014520668 network size 786
STEP 54 ================================
prereg loss 4.4247527 reg_l1 85.33488 reg_l2 26.223139
loss 4.5100875
cutoff 0.0013326419 network size 785
STEP 55 ================================
prereg loss 7.58223 reg_l1 85.364006 reg_l2 26.22993
loss 7.667594
cutoff 0.002081238 network size 784
STEP 56 ================================
prereg loss 3.6283395 reg_l1 85.384674 reg_l2 26.241701
loss 3.7137241
cutoff 0.0022193007 network size 783
STEP 57 ================================
prereg loss 5.1206145 reg_l1 85.371704 reg_l2 26.24525
loss 5.205986
cutoff 0.001943348 network size 782
STEP 58 ================================
prereg loss 3.4689171 reg_l1 85.34705 reg_l2 26.23639
loss 3.554264
cutoff 0.0019233588 network size 781
STEP 59 ================================
prereg loss 5.0834756 reg_l1 85.34654 reg_l2 26.231886
loss 5.1688223
cutoff 0.0016591195 network size 780
STEP 60 ================================
prereg loss 2.1801596 reg_l1 85.36529 reg_l2 26.235605
loss 2.2655249
cutoff 0.0017745127 network size 779
STEP 61 ================================
prereg loss 4.0122814 reg_l1 85.356766 reg_l2 26.233433
loss 4.097638
cutoff 0.0019858922 network size 778
STEP 62 ================================
prereg loss 2.8024242 reg_l1 85.32603 reg_l2 26.229305
loss 2.8877501
cutoff 0.0020503902 network size 777
STEP 63 ================================
prereg loss 1.8711053 reg_l1 85.308975 reg_l2 26.231306
loss 1.9564143
cutoff 0.0019887954 network size 776
STEP 64 ================================
prereg loss 3.914507 reg_l1 85.3215 reg_l2 26.234436
loss 3.9998283
cutoff 0.0021154643 network size 775
STEP 65 ================================
prereg loss 2.1087773 reg_l1 85.32759 reg_l2 26.235695
loss 2.194105
cutoff 0.002231124 network size 772
STEP 66 ================================
prereg loss 3.2459273 reg_l1 85.32492 reg_l2 26.23831
loss 3.3312523
cutoff 0.002381645 network size 771
STEP 67 ================================
prereg loss 3.378219 reg_l1 85.30739 reg_l2 26.235569
loss 3.4635262
cutoff 0.0021266371 network size 770
STEP 68 ================================
prereg loss 0.75667757 reg_l1 85.26176 reg_l2 26.224167
loss 0.84193933
cutoff 0.0016089099 network size 769
STEP 69 ================================
prereg loss 2.548544 reg_l1 85.216415 reg_l2 26.215183
loss 2.6337605
cutoff 0.0029344715 network size 768
STEP 70 ================================
prereg loss 1.696638 reg_l1 85.195724 reg_l2 26.216192
loss 1.7818338
cutoff 0.002980861 network size 767
STEP 71 ================================
prereg loss 1.5015045 reg_l1 85.19175 reg_l2 26.220306
loss 1.5866963
cutoff 0.0029301622 network size 766
STEP 72 ================================
prereg loss 1.1583407 reg_l1 85.189354 reg_l2 26.221687
loss 1.24353
cutoff 0.0030503916 network size 765
STEP 73 ================================
prereg loss 1.4081192 reg_l1 85.18501 reg_l2 26.223034
loss 1.4933043
cutoff 0.0032766848 network size 764
STEP 74 ================================
prereg loss 2.2907004 reg_l1 85.16988 reg_l2 26.223429
loss 2.3758702
cutoff 0.0029818092 network size 763
STEP 75 ================================
prereg loss 0.8083395 reg_l1 85.13288 reg_l2 26.219584
loss 0.8934724
cutoff 0.002691136 network size 762
STEP 76 ================================
prereg loss 2.1169956 reg_l1 85.08776 reg_l2 26.214518
loss 2.2020833
cutoff 0.0033889278 network size 760
STEP 77 ================================
prereg loss 2.0008578 reg_l1 85.05827 reg_l2 26.21508
loss 2.085916
cutoff 0.0033481787 network size 759
STEP 78 ================================
prereg loss 1.2912049 reg_l1 85.04988 reg_l2 26.219421
loss 1.3762548
cutoff 0.0035746128 network size 758
STEP 79 ================================
prereg loss 0.6999134 reg_l1 85.050095 reg_l2 26.225096
loss 0.7849635
cutoff 0.0030550286 network size 757
STEP 80 ================================
prereg loss 1.6772583 reg_l1 85.05235 reg_l2 26.23231
loss 1.7623106
cutoff 0.0030205506 network size 756
STEP 81 ================================
prereg loss 2.1822674 reg_l1 85.04961 reg_l2 26.239847
loss 2.267317
cutoff 0.0032355522 network size 755
STEP 82 ================================
prereg loss 1.6648756 reg_l1 85.03153 reg_l2 26.243488
loss 1.7499071
cutoff 0.0034785734 network size 754
STEP 83 ================================
prereg loss 0.6520125 reg_l1 85.00111 reg_l2 26.242184
loss 0.73701364
cutoff 0.0038124581 network size 752
STEP 84 ================================
prereg loss 1.613679 reg_l1 84.96673 reg_l2 26.238783
loss 1.6986458
cutoff 0.0040347567 network size 750
STEP 85 ================================
prereg loss 2.745436 reg_l1 84.94505 reg_l2 26.237312
loss 2.830381
cutoff 0.004188362 network size 748
STEP 86 ================================
prereg loss 2.9361925 reg_l1 84.94601 reg_l2 26.243067
loss 3.0211384
cutoff 0.0042944853 network size 747
STEP 87 ================================
prereg loss 1.1885475 reg_l1 84.9643 reg_l2 26.252113
loss 1.2735118
cutoff 0.0047392533 network size 745
STEP 88 ================================
prereg loss 1.8619307 reg_l1 84.97928 reg_l2 26.261715
loss 1.94691
cutoff 0.0047343755 network size 743
STEP 89 ================================
prereg loss 1.8624897 reg_l1 84.990265 reg_l2 26.270948
loss 1.94748
cutoff 0.0048928666 network size 742
STEP 90 ================================
prereg loss 1.5770146 reg_l1 84.9946 reg_l2 26.278425
loss 1.6620091
cutoff 0.0048773377 network size 741
STEP 91 ================================
prereg loss 1.3305738 reg_l1 84.98605 reg_l2 26.28265
loss 1.4155599
cutoff 0.0047597736 network size 740
STEP 92 ================================
prereg loss 2.7655177 reg_l1 84.97411 reg_l2 26.28556
loss 2.8504918
cutoff 0.004625138 network size 739
STEP 93 ================================
prereg loss 3.0052733 reg_l1 84.97144 reg_l2 26.29163
loss 3.0902448
cutoff 0.0049591525 network size 738
STEP 94 ================================
prereg loss 2.120202 reg_l1 84.99297 reg_l2 26.301804
loss 2.205195
cutoff 0.004784975 network size 737
STEP 95 ================================
prereg loss 0.91494095 reg_l1 85.01867 reg_l2 26.310452
loss 0.99995965
cutoff 0.004818485 network size 734
STEP 96 ================================
prereg loss 2.1280293 reg_l1 85.02817 reg_l2 26.31681
loss 2.2130575
cutoff 0.0047743777 network size 733
STEP 97 ================================
prereg loss 3.2650542 reg_l1 85.0398 reg_l2 26.322824
loss 3.350094
cutoff 0.004975941 network size 732
STEP 98 ================================
prereg loss 2.0688703 reg_l1 85.0334 reg_l2 26.32622
loss 2.1539037
cutoff 0.00476279 network size 731
STEP 99 ================================
prereg loss 2.7381775 reg_l1 85.00601 reg_l2 26.323008
loss 2.8231835
cutoff 0.0045854477 network size 730
STEP 100 ================================
prereg loss 0.8298083 reg_l1 84.96443 reg_l2 26.313553
loss 0.91477275
cutoff 0.0048103905 network size 729
2022-07-19T21:18:03.282
```

100 more steps in this mode (the learning curve looks more and more distrurbing), then 100 steps
in the standard non-sparsifying mode (converges nicely):

```
julia> # less perfect; nonetheless, let's continue

julia> sparsifying_steps!(100)
2022-07-19T21:43:48.736
STEP 1 ================================
prereg loss 1.3238568 reg_l1 84.92072 reg_l2 26.303574
loss 1.4087776
cutoff 0.004502036 network size 728
STEP 2 ================================
prereg loss 1.0125985 reg_l1 84.88349 reg_l2 26.297583
loss 1.097482
cutoff 0.0041425396 network size 727
STEP 3 ================================
prereg loss 1.7455069 reg_l1 84.86181 reg_l2 26.297342
loss 1.8303688
cutoff 0.004412448 network size 726
STEP 4 ================================
prereg loss 1.4903362 reg_l1 84.85856 reg_l2 26.302206
loss 1.5751947
cutoff 0.0047265277 network size 725
STEP 5 ================================
prereg loss 0.785334 reg_l1 84.866455 reg_l2 26.310162
loss 0.87020046
cutoff 0.0052404897 network size 724
STEP 6 ================================
prereg loss 1.1402144 reg_l1 84.87596 reg_l2 26.319677
loss 1.2250904
cutoff 0.0055558663 network size 723
STEP 7 ================================
prereg loss 0.5491067 reg_l1 84.87937 reg_l2 26.328997
loss 0.6339861
cutoff 0.00583809 network size 722
STEP 8 ================================
prereg loss 0.76570255 reg_l1 84.876114 reg_l2 26.337011
loss 0.85057867
cutoff 0.0057939026 network size 721
STEP 9 ================================
prereg loss 0.5456712 reg_l1 84.869484 reg_l2 26.344088
loss 0.6305407
cutoff 0.006087044 network size 719
STEP 10 ================================
prereg loss 0.65967613 reg_l1 84.85852 reg_l2 26.352205
loss 0.7445347
cutoff 0.0064086844 network size 717
STEP 11 ================================
prereg loss 0.72353345 reg_l1 84.85049 reg_l2 26.362873
loss 0.80838394
cutoff 0.006479937 network size 716
STEP 12 ================================
prereg loss 0.47562984 reg_l1 84.852684 reg_l2 26.375574
loss 0.5604825
cutoff 0.0065920684 network size 715
STEP 13 ================================
prereg loss 0.57993907 reg_l1 84.85951 reg_l2 26.388739
loss 0.66479856
cutoff 0.006601302 network size 712
STEP 14 ================================
prereg loss 0.25649974 reg_l1 84.85522 reg_l2 26.4007
loss 0.34135497
cutoff 0.0067359996 network size 711
STEP 15 ================================
prereg loss 5.380694 reg_l1 84.8629 reg_l2 26.41165
loss 5.4655566
cutoff 0.0071010734 network size 710
STEP 16 ================================
prereg loss 2.5207906 reg_l1 84.86268 reg_l2 26.424593
loss 2.6056533
cutoff 0.006886258 network size 709
STEP 17 ================================
prereg loss 1.6872268 reg_l1 84.85438 reg_l2 26.43736
loss 1.7720811
cutoff 0.006673925 network size 708
STEP 18 ================================
prereg loss 3.1259038 reg_l1 84.82978 reg_l2 26.44336
loss 3.2107337
cutoff 0.0066487845 network size 706
STEP 19 ================================
prereg loss 2.4168484 reg_l1 84.781 reg_l2 26.43907
loss 2.5016294
cutoff 0.0066370377 network size 705
STEP 20 ================================
prereg loss 1.4851454 reg_l1 84.73887 reg_l2 26.431164
loss 1.5698843
cutoff 0.0065208697 network size 704
STEP 21 ================================
prereg loss 2.5889406 reg_l1 84.708916 reg_l2 26.424715
loss 2.6736495
cutoff 0.0073394156 network size 701
STEP 22 ================================
prereg loss 4.848761 reg_l1 84.68007 reg_l2 26.424074
loss 4.933441
cutoff 0.007856742 network size 699
STEP 23 ================================
prereg loss 11.233214 reg_l1 84.67445 reg_l2 26.431677
loss 11.317889
cutoff 0.0074883937 network size 698
STEP 24 ================================
prereg loss 2.1567721 reg_l1 84.713905 reg_l2 26.448421
loss 2.241486
cutoff 0.006023793 network size 697
STEP 25 ================================
prereg loss 6.025082 reg_l1 84.75574 reg_l2 26.46744
loss 6.109838
cutoff 0.007822853 network size 696
STEP 26 ================================
prereg loss 29.079815 reg_l1 84.77064 reg_l2 26.4836
loss 29.164585
cutoff 0.0076879687 network size 695
STEP 27 ================================
prereg loss 2.961008 reg_l1 84.74986 reg_l2 26.499813
loss 3.045758
cutoff 0.0074930335 network size 694
STEP 28 ================================
prereg loss 41.604206 reg_l1 84.719795 reg_l2 26.511114
loss 41.688927
cutoff 0.007752091 network size 693
STEP 29 ================================
prereg loss 18.255215 reg_l1 84.68295 reg_l2 26.508236
loss 18.339897
cutoff 0.008379387 network size 691
STEP 30 ================================
prereg loss 6.0045485 reg_l1 84.633156 reg_l2 26.493855
loss 6.089182
cutoff 0.008322812 network size 690
STEP 31 ================================
prereg loss 24.210213 reg_l1 84.58088 reg_l2 26.477308
loss 24.294794
cutoff 0.007730808 network size 689
STEP 32 ================================
prereg loss 16.663677 reg_l1 84.513916 reg_l2 26.463312
loss 16.748192
cutoff 0.006172981 network size 688
STEP 33 ================================
prereg loss 4.625311 reg_l1 84.45087 reg_l2 26.457237
loss 4.7097616
cutoff 0.0064528016 network size 687
STEP 34 ================================
prereg loss 11.226353 reg_l1 84.41741 reg_l2 26.463886
loss 11.31077
cutoff 0.0075094993 network size 686
STEP 35 ================================
prereg loss 10.31868 reg_l1 84.409935 reg_l2 26.478844
loss 10.40309
cutoff 0.007961104 network size 685
STEP 36 ================================
prereg loss 2.4666488 reg_l1 84.41529 reg_l2 26.49712
loss 2.551064
cutoff 0.008114212 network size 684
STEP 37 ================================
prereg loss 5.350068 reg_l1 84.41171 reg_l2 26.511044
loss 5.4344797
cutoff 0.00795771 network size 683
STEP 38 ================================
prereg loss 9.28425 reg_l1 84.38174 reg_l2 26.51705
loss 9.368632
cutoff 0.0070946557 network size 682
STEP 39 ================================
prereg loss 3.9415312 reg_l1 84.32522 reg_l2 26.516087
loss 4.0258565
cutoff 0.007829363 network size 681
STEP 40 ================================
prereg loss 1.1216063 reg_l1 84.26059 reg_l2 26.512077
loss 1.2058669
cutoff 0.007686356 network size 680
STEP 41 ================================
prereg loss 5.0042796 reg_l1 84.210785 reg_l2 26.51092
loss 5.0884905
cutoff 0.0077711972 network size 679
STEP 42 ================================
prereg loss 6.2743945 reg_l1 84.195015 reg_l2 26.517715
loss 6.3585896
cutoff 0.008047216 network size 678
STEP 43 ================================
prereg loss 3.139936 reg_l1 84.21491 reg_l2 26.53307
loss 3.224151
cutoff 0.008141286 network size 677
STEP 44 ================================
prereg loss 2.9363694 reg_l1 84.259315 reg_l2 26.55566
loss 3.0206287
cutoff 0.008126239 network size 676
STEP 45 ================================
prereg loss 5.6170096 reg_l1 84.31147 reg_l2 26.582092
loss 5.701321
cutoff 0.008657921 network size 675
STEP 46 ================================
prereg loss 3.5950177 reg_l1 84.35504 reg_l2 26.608055
loss 3.6793728
cutoff 0.008790768 network size 673
STEP 47 ================================
prereg loss 1.411385 reg_l1 84.37513 reg_l2 26.629635
loss 1.4957602
cutoff 0.0092203785 network size 672
STEP 48 ================================
prereg loss 2.5882485 reg_l1 84.38778 reg_l2 26.644798
loss 2.6726363
cutoff 0.009251199 network size 671
STEP 49 ================================
prereg loss 4.639847 reg_l1 84.386314 reg_l2 26.653187
loss 4.724233
cutoff 0.00911571 network size 670
STEP 50 ================================
prereg loss 2.8710296 reg_l1 84.37488 reg_l2 26.656517
loss 2.9554045
cutoff 0.009151944 network size 669
STEP 51 ================================
prereg loss 0.713449 reg_l1 84.3567 reg_l2 26.657024
loss 0.7978057
cutoff 0.00869783 network size 668
STEP 52 ================================
prereg loss 1.6345861 reg_l1 84.33339 reg_l2 26.656729
loss 1.7189195
cutoff 0.008737881 network size 667
STEP 53 ================================
prereg loss 3.042404 reg_l1 84.3072 reg_l2 26.657345
loss 3.1267111
cutoff 0.008862402 network size 666
STEP 54 ================================
prereg loss 1.9265444 reg_l1 84.28143 reg_l2 26.660648
loss 2.0108259
cutoff 0.009222781 network size 665
STEP 55 ================================
prereg loss 0.74866414 reg_l1 84.262276 reg_l2 26.667952
loss 0.8329264
cutoff 0.009356476 network size 664
STEP 56 ================================
prereg loss 1.2770364 reg_l1 84.25475 reg_l2 26.679268
loss 1.3612912
cutoff 0.009759719 network size 662
STEP 57 ================================
prereg loss 1.9021696 reg_l1 84.2509 reg_l2 26.693504
loss 1.9864205
cutoff 0.010053938 network size 660
STEP 58 ================================
prereg loss 1.5720205 reg_l1 84.25685 reg_l2 26.708542
loss 1.6562774
cutoff 0.010795343 network size 658
STEP 59 ================================
prereg loss 0.82279456 reg_l1 84.269356 reg_l2 26.723223
loss 0.9070639
cutoff 0.010861499 network size 655
STEP 60 ================================
prereg loss 1.1351247 reg_l1 84.26981 reg_l2 26.735407
loss 1.2193944
cutoff 0.010948544 network size 654
STEP 61 ================================
prereg loss 1.4549795 reg_l1 84.284294 reg_l2 26.744299
loss 1.5392638
cutoff 0.011219713 network size 652
STEP 62 ================================
prereg loss 3.5799363 reg_l1 84.27725 reg_l2 26.749866
loss 3.6642134
cutoff 0.01129755 network size 649
STEP 63 ================================
prereg loss 1.0676484 reg_l1 84.24301 reg_l2 26.75201
loss 1.1518915
cutoff 0.011321812 network size 648
STEP 64 ================================
prereg loss 4.796991 reg_l1 84.230736 reg_l2 26.75515
loss 4.881222
cutoff 0.011728255 network size 646
STEP 65 ================================
prereg loss 5.0986047 reg_l1 84.23835 reg_l2 26.766695
loss 5.182843
cutoff 0.011743087 network size 644
STEP 66 ================================
prereg loss 77.31263 reg_l1 84.25971 reg_l2 26.780415
loss 77.39689
cutoff 0.012082117 network size 642
STEP 67 ================================
prereg loss 43.340614 reg_l1 84.22919 reg_l2 26.79128
loss 43.424843
cutoff 0.011636102 network size 641
STEP 68 ================================
prereg loss 10.112212 reg_l1 84.19392 reg_l2 26.805222
loss 10.196406
cutoff 0.010280846 network size 639
STEP 69 ================================
prereg loss 6.7878575 reg_l1 84.16328 reg_l2 26.829025
loss 6.8720207
cutoff 0.010565689 network size 638
STEP 70 ================================
prereg loss 55.774208 reg_l1 84.174164 reg_l2 26.862951
loss 55.858383
cutoff 0.010886071 network size 637
STEP 71 ================================
prereg loss 52.35575 reg_l1 84.19655 reg_l2 26.892643
loss 52.43995
cutoff 0.009854665 network size 636
STEP 72 ================================
prereg loss 30.087414 reg_l1 84.226425 reg_l2 26.9152
loss 30.17164
cutoff 0.010079638 network size 635
STEP 73 ================================
prereg loss 15.014013 reg_l1 84.24431 reg_l2 26.923473
loss 15.098258
cutoff 0.010231682 network size 634
STEP 74 ================================
prereg loss 15.05908 reg_l1 84.22115 reg_l2 26.912006
loss 15.143301
cutoff 0.009361928 network size 633
STEP 75 ================================
prereg loss 14.52784 reg_l1 84.14517 reg_l2 26.881842
loss 14.611985
cutoff 0.008549178 network size 632
STEP 76 ================================
prereg loss 11.110826 reg_l1 84.03468 reg_l2 26.843641
loss 11.19486
cutoff 0.008458806 network size 631
STEP 77 ================================
prereg loss 13.990156 reg_l1 83.924385 reg_l2 26.810022
loss 14.07408
cutoff 0.010321202 network size 629
STEP 78 ================================
prereg loss 19.024862 reg_l1 83.84116 reg_l2 26.794338
loss 19.108704
cutoff 0.01004081 network size 628
STEP 79 ================================
prereg loss 19.299738 reg_l1 83.82115 reg_l2 26.801157
loss 19.383558
cutoff 0.010428563 network size 627
STEP 80 ================================
prereg loss 14.440952 reg_l1 83.84491 reg_l2 26.826782
loss 14.524797
cutoff 0.009995036 network size 626
STEP 81 ================================
prereg loss 7.315097 reg_l1 83.90245 reg_l2 26.866905
loss 7.398999
cutoff 0.0095927445 network size 625
STEP 82 ================================
prereg loss 2.2940626 reg_l1 83.98282 reg_l2 26.91607
loss 2.3780456
cutoff 0.00937419 network size 624
STEP 83 ================================
prereg loss 1.7211083 reg_l1 84.06658 reg_l2 26.965637
loss 1.805175
cutoff 0.009835479 network size 622
STEP 84 ================================
prereg loss 4.3699365 reg_l1 84.12829 reg_l2 27.009825
loss 4.454065
cutoff 0.010342192 network size 621
STEP 85 ================================
prereg loss 7.5892034 reg_l1 84.17774 reg_l2 27.0439
loss 7.6733813
cutoff 0.010063666 network size 620
STEP 86 ================================
prereg loss 8.941956 reg_l1 84.20152 reg_l2 27.065702
loss 9.026157
cutoff 0.0111593325 network size 619
STEP 87 ================================
prereg loss 7.5100117 reg_l1 84.20215 reg_l2 27.076084
loss 7.594214
cutoff 0.011789513 network size 618
STEP 88 ================================
prereg loss 5.157414 reg_l1 84.18672 reg_l2 27.078674
loss 5.2416005
cutoff 0.011936472 network size 616
STEP 89 ================================
prereg loss 3.2715576 reg_l1 84.149376 reg_l2 27.076454
loss 3.355707
cutoff 0.0121727185 network size 615
STEP 90 ================================
prereg loss 2.3680453 reg_l1 84.11982 reg_l2 27.072987
loss 2.4521651
cutoff 0.012658913 network size 613
STEP 91 ================================
prereg loss 5.876122 reg_l1 84.07632 reg_l2 27.07031
loss 5.9601984
cutoff 0.0129340235 network size 611
STEP 92 ================================
prereg loss 38.49058 reg_l1 84.0407 reg_l2 27.07131
loss 38.574623
cutoff 0.012283648 network size 610
STEP 93 ================================
prereg loss 26.12524 reg_l1 84.046646 reg_l2 27.08367
loss 26.209288
cutoff 0.011063907 network size 609
STEP 94 ================================
prereg loss 16.709984 reg_l1 84.079254 reg_l2 27.10787
loss 16.794064
cutoff 0.012311485 network size 608
STEP 95 ================================
prereg loss 11.700401 reg_l1 84.13152 reg_l2 27.142412
loss 11.784533
cutoff 0.012474586 network size 607
STEP 96 ================================
prereg loss 10.646523 reg_l1 84.19395 reg_l2 27.183659
loss 10.730718
cutoff 0.012733607 network size 605
STEP 97 ================================
prereg loss 11.427266 reg_l1 84.24707 reg_l2 27.228403
loss 11.511513
cutoff 0.012003754 network size 604
STEP 98 ================================
prereg loss 12.635269 reg_l1 84.308945 reg_l2 27.272835
loss 12.719578
cutoff 0.011547254 network size 603
STEP 99 ================================
prereg loss 12.4149885 reg_l1 84.35804 reg_l2 27.311472
loss 12.499347
cutoff 0.011078414 network size 602
STEP 100 ================================
prereg loss 8.94181 reg_l1 84.38817 reg_l2 27.3418
loss 9.026197
cutoff 0.009401592 network size 601
2022-07-19T22:21:16.776

julia> steps!(100)
2022-07-19T22:56:12.761
STEP 1 ================================
prereg loss 4.411227 reg_l1 84.40459 reg_l2 27.364614
loss 4.4956317
STEP 2 ================================
prereg loss 3.021094 reg_l1 84.42093 reg_l2 27.381817
loss 3.105515
STEP 3 ================================
prereg loss 3.9788992 reg_l1 84.431854 reg_l2 27.39579
loss 4.063331
STEP 4 ================================
prereg loss 5.553472 reg_l1 84.44333 reg_l2 27.408312
loss 5.6379156
STEP 5 ================================
prereg loss 5.90002 reg_l1 84.4594 reg_l2 27.42122
loss 5.9844794
STEP 6 ================================
prereg loss 4.6677723 reg_l1 84.48533 reg_l2 27.43686
loss 4.752258
STEP 7 ================================
prereg loss 2.7736323 reg_l1 84.52129 reg_l2 27.455786
loss 2.8581536
STEP 8 ================================
prereg loss 1.5731573 reg_l1 84.5632 reg_l2 27.477182
loss 1.6577206
STEP 9 ================================
prereg loss 1.686191 reg_l1 84.604546 reg_l2 27.49852
loss 1.7707955
STEP 10 ================================
prereg loss 2.6200266 reg_l1 84.63802 reg_l2 27.51683
loss 2.7046647
STEP 11 ================================
prereg loss 3.4610817 reg_l1 84.656425 reg_l2 27.530493
loss 3.5457382
STEP 12 ================================
prereg loss 3.5710328 reg_l1 84.65748 reg_l2 27.53865
loss 3.6556902
STEP 13 ================================
prereg loss 2.8723652 reg_l1 84.641685 reg_l2 27.541357
loss 2.957007
STEP 14 ================================
prereg loss 1.859936 reg_l1 84.61345 reg_l2 27.53988
loss 1.9445494
STEP 15 ================================
prereg loss 1.150704 reg_l1 84.57918 reg_l2 27.536016
loss 1.2352833
STEP 16 ================================
prereg loss 1.0521086 reg_l1 84.54428 reg_l2 27.531143
loss 1.136653
STEP 17 ================================
prereg loss 1.4407002 reg_l1 84.51408 reg_l2 27.52681
loss 1.5252142
STEP 18 ================================
prereg loss 1.9184306 reg_l1 84.49132 reg_l2 27.523804
loss 2.0029218
STEP 19 ================================
prereg loss 2.1128228 reg_l1 84.47749 reg_l2 27.522608
loss 2.1973002
STEP 20 ================================
prereg loss 1.8799322 reg_l1 84.47318 reg_l2 27.523262
loss 1.9644053
STEP 21 ================================
prereg loss 1.390845 reg_l1 84.47606 reg_l2 27.525074
loss 1.475321
STEP 22 ================================
prereg loss 0.9587962 reg_l1 84.48243 reg_l2 27.527128
loss 1.0432787
STEP 23 ================================
prereg loss 0.7998577 reg_l1 84.48847 reg_l2 27.52834
loss 0.8843461
STEP 24 ================================
prereg loss 0.9134323 reg_l1 84.49051 reg_l2 27.527792
loss 0.99792284
STEP 25 ================================
prereg loss 1.134279 reg_l1 84.486046 reg_l2 27.52482
loss 1.218765
STEP 26 ================================
prereg loss 1.2775453 reg_l1 84.47386 reg_l2 27.519335
loss 1.3620192
STEP 27 ================================
prereg loss 1.2534759 reg_l1 84.45473 reg_l2 27.51169
loss 1.3379307
STEP 28 ================================
prereg loss 1.0955006 reg_l1 84.430275 reg_l2 27.502577
loss 1.1799309
STEP 29 ================================
prereg loss 0.9073343 reg_l1 84.40345 reg_l2 27.492899
loss 0.9917378
STEP 30 ================================
prereg loss 0.7867593 reg_l1 84.377045 reg_l2 27.483568
loss 0.87113637
STEP 31 ================================
prereg loss 0.7622628 reg_l1 84.353226 reg_l2 27.47508
loss 0.846616
STEP 32 ================================
prereg loss 0.79835397 reg_l1 84.33335 reg_l2 27.467781
loss 0.88268733
STEP 33 ================================
prereg loss 0.83503044 reg_l1 84.31767 reg_l2 27.46163
loss 0.9193481
STEP 34 ================================
prereg loss 0.83097404 reg_l1 84.3056 reg_l2 27.456392
loss 0.9152796
STEP 35 ================================
prereg loss 0.78393346 reg_l1 84.29579 reg_l2 27.451517
loss 0.86822927
STEP 36 ================================
prereg loss 0.7186779 reg_l1 84.28692 reg_l2 27.446516
loss 0.8029648
STEP 37 ================================
prereg loss 0.6641516 reg_l1 84.27753 reg_l2 27.440987
loss 0.7484291
STEP 38 ================================
prereg loss 0.63459474 reg_l1 84.266335 reg_l2 27.43465
loss 0.7188611
STEP 39 ================================
prereg loss 0.62729 reg_l1 84.25315 reg_l2 27.427622
loss 0.71154314
STEP 40 ================================
prereg loss 0.63154185 reg_l1 84.2381 reg_l2 27.420073
loss 0.71577996
STEP 41 ================================
prereg loss 0.63509023 reg_l1 84.222534 reg_l2 27.412357
loss 0.7193128
STEP 42 ================================
prereg loss 0.6305567 reg_l1 84.20802 reg_l2 27.404858
loss 0.7147647
STEP 43 ================================
prereg loss 0.6177598 reg_l1 84.194496 reg_l2 27.397976
loss 0.7019543
STEP 44 ================================
prereg loss 0.6003991 reg_l1 84.183044 reg_l2 27.392038
loss 0.6845821
STEP 45 ================================
prereg loss 0.5839275 reg_l1 84.174194 reg_l2 27.387226
loss 0.6681017
STEP 46 ================================
prereg loss 0.57123333 reg_l1 84.16815 reg_l2 27.383621
loss 0.65540147
STEP 47 ================================
prereg loss 0.5618919 reg_l1 84.16487 reg_l2 27.381128
loss 0.6460568
STEP 48 ================================
prereg loss 0.55454624 reg_l1 84.16382 reg_l2 27.379496
loss 0.6387101
STEP 49 ================================
prereg loss 0.5480192 reg_l1 84.16435 reg_l2 27.378464
loss 0.63218355
STEP 50 ================================
prereg loss 0.54147077 reg_l1 84.165726 reg_l2 27.377724
loss 0.6256365
STEP 51 ================================
prereg loss 0.534474 reg_l1 84.16716 reg_l2 27.37692
loss 0.6186412
STEP 52 ================================
prereg loss 0.52693 reg_l1 84.168015 reg_l2 27.375875
loss 0.611098
STEP 53 ================================
prereg loss 0.5190293 reg_l1 84.16786 reg_l2 27.374468
loss 0.6031972
STEP 54 ================================
prereg loss 0.5113969 reg_l1 84.16653 reg_l2 27.37266
loss 0.5955634
STEP 55 ================================
prereg loss 0.5050159 reg_l1 84.16416 reg_l2 27.370594
loss 0.58918005
STEP 56 ================================
prereg loss 0.50046813 reg_l1 84.16114 reg_l2 27.36837
loss 0.5846293
STEP 57 ================================
prereg loss 0.4977604 reg_l1 84.15792 reg_l2 27.36621
loss 0.5819183
STEP 58 ================================
prereg loss 0.49637136 reg_l1 84.15489 reg_l2 27.364342
loss 0.58052623
STEP 59 ================================
prereg loss 0.49404287 reg_l1 84.15272 reg_l2 27.362946
loss 0.5781956
STEP 60 ================================
prereg loss 0.48940092 reg_l1 84.15132 reg_l2 27.36212
loss 0.57355225
STEP 61 ================================
prereg loss 0.48236763 reg_l1 84.15078 reg_l2 27.361845
loss 0.5665184
STEP 62 ================================
prereg loss 0.47425568 reg_l1 84.150955 reg_l2 27.361925
loss 0.55840665
STEP 63 ================================
prereg loss 0.46692836 reg_l1 84.151596 reg_l2 27.362387
loss 0.55108
STEP 64 ================================
prereg loss 0.461644 reg_l1 84.152466 reg_l2 27.362902
loss 0.54579645
STEP 65 ================================
prereg loss 0.45806175 reg_l1 84.153145 reg_l2 27.363375
loss 0.5422149
STEP 66 ================================
prereg loss 0.45525375 reg_l1 84.15323 reg_l2 27.363676
loss 0.539407
STEP 67 ================================
prereg loss 0.45208508 reg_l1 84.15285 reg_l2 27.363745
loss 0.53623796
STEP 68 ================================
prereg loss 0.44831577 reg_l1 84.15188 reg_l2 27.363583
loss 0.53246766
STEP 69 ================================
prereg loss 0.4442586 reg_l1 84.15041 reg_l2 27.363268
loss 0.528409
STEP 70 ================================
prereg loss 0.44055426 reg_l1 84.14869 reg_l2 27.362911
loss 0.52470297
STEP 71 ================================
prereg loss 0.43762136 reg_l1 84.14686 reg_l2 27.362576
loss 0.5217682
STEP 72 ================================
prereg loss 0.43545827 reg_l1 84.14518 reg_l2 27.362375
loss 0.51960343
STEP 73 ================================
prereg loss 0.43365687 reg_l1 84.14388 reg_l2 27.362389
loss 0.51780075
STEP 74 ================================
prereg loss 0.43167326 reg_l1 84.14315 reg_l2 27.362616
loss 0.5158164
STEP 75 ================================
prereg loss 0.42919978 reg_l1 84.14301 reg_l2 27.36313
loss 0.5133428
STEP 76 ================================
prereg loss 0.4262997 reg_l1 84.14336 reg_l2 27.363832
loss 0.51044303
STEP 77 ================================
prereg loss 0.42316133 reg_l1 84.14402 reg_l2 27.364668
loss 0.5073054
STEP 78 ================================
prereg loss 0.42016456 reg_l1 84.14496 reg_l2 27.365602
loss 0.50430954
STEP 79 ================================
prereg loss 0.41753832 reg_l1 84.14605 reg_l2 27.366592
loss 0.50168437
STEP 80 ================================
prereg loss 0.41532716 reg_l1 84.14694 reg_l2 27.367544
loss 0.4994741
STEP 81 ================================
prereg loss 0.41334665 reg_l1 84.14776 reg_l2 27.368443
loss 0.4974944
STEP 82 ================================
prereg loss 0.41139153 reg_l1 84.14825 reg_l2 27.369234
loss 0.49553978
STEP 83 ================================
prereg loss 0.4093791 reg_l1 84.148506 reg_l2 27.369947
loss 0.4935276
STEP 84 ================================
prereg loss 0.40733275 reg_l1 84.14868 reg_l2 27.370531
loss 0.49148142
STEP 85 ================================
prereg loss 0.4053297 reg_l1 84.14869 reg_l2 27.371103
loss 0.4894784
STEP 86 ================================
prereg loss 0.40346375 reg_l1 84.14896 reg_l2 27.371626
loss 0.48761272
STEP 87 ================================
prereg loss 0.4017424 reg_l1 84.14931 reg_l2 27.372242
loss 0.4858917
STEP 88 ================================
prereg loss 0.40011364 reg_l1 84.149864 reg_l2 27.372835
loss 0.4842635
STEP 89 ================================
prereg loss 0.398498 reg_l1 84.15054 reg_l2 27.373508
loss 0.48264855
STEP 90 ================================
prereg loss 0.39683822 reg_l1 84.15144 reg_l2 27.374249
loss 0.48098966
STEP 91 ================================
prereg loss 0.395135 reg_l1 84.15239 reg_l2 27.37503
loss 0.4792874
STEP 92 ================================
prereg loss 0.3933689 reg_l1 84.153404 reg_l2 27.375864
loss 0.4775223
STEP 93 ================================
prereg loss 0.39157817 reg_l1 84.15436 reg_l2 27.37673
loss 0.47573254
STEP 94 ================================
prereg loss 0.38981476 reg_l1 84.15536 reg_l2 27.377584
loss 0.47397012
STEP 95 ================================
prereg loss 0.38811266 reg_l1 84.15618 reg_l2 27.378372
loss 0.47226885
STEP 96 ================================
prereg loss 0.3864668 reg_l1 84.15688 reg_l2 27.37912
loss 0.47062367
STEP 97 ================================
prereg loss 0.38486868 reg_l1 84.15754 reg_l2 27.379858
loss 0.4690262
STEP 98 ================================
prereg loss 0.383303 reg_l1 84.158165 reg_l2 27.380518
loss 0.46746117
STEP 99 ================================
prereg loss 0.38176808 reg_l1 84.15873 reg_l2 27.38117
loss 0.46592683
STEP 100 ================================
prereg loss 0.3802408 reg_l1 84.15929 reg_l2 27.381807
loss 0.46440008
2022-07-19T23:28:43.864
```

repeating this pattern: 100 more steps in this mode (the learning curve looks more and more distrurbing), then 100 steps in the standard non-sparsifying mode (converges nicely):

```
julia> sparsifying_steps!(100)
2022-07-19T23:31:30.135
STEP 1 ================================
prereg loss 0.37872556 reg_l1 84.15998 reg_l2 27.382462
loss 0.46288556
cutoff 2.756146e-6 network size 600
STEP 2 ================================
prereg loss 0.37723312 reg_l1 84.16082 reg_l2 27.383131
loss 0.46139395
cutoff 0.0017449145 network size 599
STEP 3 ================================
prereg loss 0.3758833 reg_l1 84.16015 reg_l2 27.383911
loss 0.46004346
cutoff 0.0025773358 network size 598
STEP 4 ================================
prereg loss 0.374529 reg_l1 84.15861 reg_l2 27.384691
loss 0.4586876
cutoff 0.0037799985 network size 597
STEP 5 ================================
prereg loss 0.37377182 reg_l1 84.156136 reg_l2 27.385515
loss 0.45792794
cutoff 0.00451851 network size 596
STEP 6 ================================
prereg loss 0.37227 reg_l1 84.1528 reg_l2 27.38635
loss 0.4564228
cutoff 0.006395363 network size 595
STEP 7 ================================
prereg loss 0.3712183 reg_l1 84.14767 reg_l2 27.387182
loss 0.45536596
cutoff 0.0071471687 network size 594
STEP 8 ================================
prereg loss 0.3694238 reg_l1 84.14161 reg_l2 27.38807
loss 0.45356542
cutoff 0.007663896 network size 593
STEP 9 ================================
prereg loss 0.36799255 reg_l1 84.13512 reg_l2 27.388922
loss 0.45212767
cutoff 0.008852502 network size 591
STEP 10 ================================
prereg loss 0.36620715 reg_l1 84.11846 reg_l2 27.389668
loss 0.4503256
cutoff 0.008879455 network size 590
STEP 11 ================================
prereg loss 0.36472902 reg_l1 84.110756 reg_l2 27.390507
loss 0.44883978
cutoff 0.00951667 network size 589
STEP 12 ================================
prereg loss 0.36340266 reg_l1 84.10273 reg_l2 27.39133
loss 0.4475054
cutoff 0.009666807 network size 588
STEP 13 ================================
prereg loss 0.42230642 reg_l1 84.09486 reg_l2 27.392212
loss 0.5064013
cutoff 0.009834481 network size 587
STEP 14 ================================
prereg loss 0.41150647 reg_l1 84.08319 reg_l2 27.391905
loss 0.49558967
cutoff 0.011112256 network size 586
STEP 15 ================================
prereg loss 0.4441415 reg_l1 84.06813 reg_l2 27.390684
loss 0.5282096
cutoff 0.010981987 network size 585
STEP 16 ================================
prereg loss 0.42667606 reg_l1 84.05287 reg_l2 27.388966
loss 0.51072896
cutoff 0.011298681 network size 583
STEP 17 ================================
prereg loss 0.5182033 reg_l1 84.02743 reg_l2 27.387037
loss 0.6022307
cutoff 0.011631582 network size 581
STEP 18 ================================
prereg loss 0.5447775 reg_l1 84.0028 reg_l2 27.385128
loss 0.6287803
cutoff 0.011784052 network size 580
STEP 19 ================================
prereg loss 0.5044942 reg_l1 83.989365 reg_l2 27.382671
loss 0.5884836
cutoff 0.012562712 network size 579
STEP 20 ================================
prereg loss 0.60946584 reg_l1 83.975 reg_l2 27.379929
loss 0.69344085
cutoff 0.013074711 network size 578
STEP 21 ================================
prereg loss 0.55506194 reg_l1 83.958916 reg_l2 27.37645
loss 0.63902086
cutoff 0.014210907 network size 576
STEP 22 ================================
prereg loss 0.5123387 reg_l1 83.92689 reg_l2 27.37247
loss 0.5962656
cutoff 0.014645452 network size 574
STEP 23 ================================
prereg loss 0.4930152 reg_l1 83.89437 reg_l2 27.368681
loss 0.57690954
cutoff 0.01492245 network size 573
STEP 24 ================================
prereg loss 0.48889533 reg_l1 83.87745 reg_l2 27.365839
loss 0.5727728
cutoff 0.015967503 network size 571
STEP 25 ================================
prereg loss 0.4825408 reg_l1 83.84573 reg_l2 27.363739
loss 0.5663865
cutoff 0.015826216 network size 570
STEP 26 ================================
prereg loss 0.4785745 reg_l1 83.83153 reg_l2 27.362768
loss 0.56240606
cutoff 0.016265705 network size 569
STEP 27 ================================
prereg loss 0.4736346 reg_l1 83.81926 reg_l2 27.362797
loss 0.5574539
cutoff 0.016537907 network size 568
STEP 28 ================================
prereg loss 0.5877116 reg_l1 83.80836 reg_l2 27.363697
loss 0.67151994
cutoff 0.01679381 network size 567
STEP 29 ================================
prereg loss 0.54286844 reg_l1 83.80137 reg_l2 27.366335
loss 0.6266698
cutoff 0.017681051 network size 565
STEP 30 ================================
prereg loss 0.6172119 reg_l1 83.77827 reg_l2 27.369734
loss 0.70099014
cutoff 0.018147623 network size 564
STEP 31 ================================
prereg loss 0.55009466 reg_l1 83.77425 reg_l2 27.374495
loss 0.63386893
cutoff 0.018609755 network size 562
STEP 32 ================================
prereg loss 3.084665 reg_l1 83.751366 reg_l2 27.3793
loss 3.1684165
cutoff 0.019021722 network size 560
STEP 33 ================================
prereg loss 4.0018225 reg_l1 83.71822 reg_l2 27.38294
loss 4.085541
cutoff 0.019245982 network size 558
STEP 34 ================================
prereg loss 2.990972 reg_l1 83.6774 reg_l2 27.385277
loss 3.0746493
cutoff 0.01971328 network size 556
STEP 35 ================================
prereg loss 2.0737662 reg_l1 83.63328 reg_l2 27.38695
loss 2.1573994
cutoff 0.020184355 network size 555
STEP 36 ================================
prereg loss 2.0787826 reg_l1 83.61514 reg_l2 27.390215
loss 2.1623976
cutoff 0.020650499 network size 554
STEP 37 ================================
prereg loss 10.81021 reg_l1 83.60642 reg_l2 27.395496
loss 10.893817
cutoff 0.020894822 network size 553
STEP 38 ================================
prereg loss 8.692363 reg_l1 83.62961 reg_l2 27.408718
loss 8.775992
cutoff 0.021160113 network size 552
STEP 39 ================================
prereg loss 4.575205 reg_l1 83.67268 reg_l2 27.428354
loss 4.6588774
cutoff 0.0216927 network size 551
STEP 40 ================================
prereg loss 2.7366025 reg_l1 83.72432 reg_l2 27.451506
loss 2.8203268
cutoff 0.021964747 network size 549
STEP 41 ================================
prereg loss 3.9327044 reg_l1 83.754974 reg_l2 27.47553
loss 4.0164595
cutoff 0.021843046 network size 548
STEP 42 ================================
prereg loss 5.797743 reg_l1 83.78729 reg_l2 27.491735
loss 5.8815303
cutoff 0.02245583 network size 547
STEP 43 ================================
prereg loss 5.666261 reg_l1 83.7894 reg_l2 27.496939
loss 5.7500505
cutoff 0.021899445 network size 546
STEP 44 ================================
prereg loss 4.5615335 reg_l1 83.76378 reg_l2 27.492088
loss 4.645297
cutoff 0.022893032 network size 544
STEP 45 ================================
prereg loss 5.5121365 reg_l1 83.69327 reg_l2 27.479431
loss 5.59583
cutoff 0.022949496 network size 543
STEP 46 ================================
prereg loss 3.1872778 reg_l1 83.60508 reg_l2 27.447674
loss 3.2708828
cutoff 0.023072628 network size 542
STEP 47 ================================
prereg loss 3.8762572 reg_l1 83.49806 reg_l2 27.406473
loss 3.9597552
cutoff 0.023588113 network size 540
STEP 48 ================================
prereg loss 5.941941 reg_l1 83.38645 reg_l2 27.37447
loss 6.025327
cutoff 0.023732664 network size 537
STEP 49 ================================
prereg loss 7.5783577 reg_l1 83.277374 reg_l2 27.355553
loss 7.661635
cutoff 0.024014166 network size 536
STEP 50 ================================
prereg loss 6.7773833 reg_l1 83.24486 reg_l2 27.352596
loss 6.860628
cutoff 0.024051478 network size 535
STEP 51 ================================
prereg loss 4.8478255 reg_l1 83.235535 reg_l2 27.361523
loss 4.9310613
cutoff 0.024525251 network size 531
STEP 52 ================================
prereg loss 2.9813244 reg_l1 83.167625 reg_l2 27.376719
loss 3.064492
cutoff 0.024849897 network size 529
STEP 53 ================================
prereg loss 2.0059037 reg_l1 83.155045 reg_l2 27.396727
loss 2.0890589
cutoff 0.02548071 network size 526
STEP 54 ================================
prereg loss 5.735187 reg_l1 83.11143 reg_l2 27.414036
loss 5.8182983
cutoff 0.025805678 network size 525
STEP 55 ================================
prereg loss 6.3429 reg_l1 83.08386 reg_l2 27.415634
loss 6.4259834
cutoff 0.02574539 network size 524
STEP 56 ================================
prereg loss 4.5360713 reg_l1 83.024605 reg_l2 27.401556
loss 4.619096
cutoff 0.025306828 network size 523
STEP 57 ================================
prereg loss 3.5413718 reg_l1 82.94025 reg_l2 27.374466
loss 3.6243122
cutoff 0.026565023 network size 521
STEP 58 ================================
prereg loss 2.6548643 reg_l1 82.80916 reg_l2 27.336828
loss 2.7376735
cutoff 0.026257088 network size 520
STEP 59 ================================
prereg loss 1.8244134 reg_l1 82.69365 reg_l2 27.293926
loss 1.9071071
cutoff 0.027280942 network size 519
STEP 60 ================================
prereg loss 1.5636135 reg_l1 82.57223 reg_l2 27.24823
loss 1.6461858
cutoff 0.02746586 network size 516
STEP 61 ================================
prereg loss 2.124473 reg_l1 82.400795 reg_l2 27.20308
loss 2.206874
cutoff 0.027806731 network size 513
STEP 62 ================================
prereg loss 4.0595813 reg_l1 82.24189 reg_l2 27.1644
loss 4.1418233
cutoff 0.027400665 network size 512
STEP 63 ================================
prereg loss 4.4054794 reg_l1 82.15372 reg_l2 27.136223
loss 4.487633
cutoff 0.028133484 network size 508
STEP 64 ================================
prereg loss 4.8040004 reg_l1 81.998436 reg_l2 27.115402
loss 4.8859987
cutoff 0.028601151 network size 507
STEP 65 ================================
prereg loss 4.9587317 reg_l1 81.94425 reg_l2 27.106457
loss 5.040676
cutoff 0.029253796 network size 502
STEP 66 ================================
prereg loss 2.601575 reg_l1 81.80009 reg_l2 27.107828
loss 2.683375
cutoff 0.029500272 network size 501
STEP 67 ================================
prereg loss 2.1155028 reg_l1 81.78848 reg_l2 27.120413
loss 2.1972914
cutoff 0.029902143 network size 500
STEP 68 ================================
prereg loss 1.9033602 reg_l1 81.78682 reg_l2 27.138079
loss 1.9851471
cutoff 0.029723212 network size 499
STEP 69 ================================
prereg loss 1.9921725 reg_l1 81.78897 reg_l2 27.157242
loss 2.0739615
cutoff 0.030102864 network size 498
STEP 70 ================================
prereg loss 2.8955805 reg_l1 81.78875 reg_l2 27.174608
loss 2.9773693
cutoff 0.030410744 network size 497
STEP 71 ================================
prereg loss 3.1882014 reg_l1 81.777794 reg_l2 27.185822
loss 3.2699792
cutoff 0.030793576 network size 495
STEP 72 ================================
prereg loss 2.7970307 reg_l1 81.72293 reg_l2 27.188803
loss 2.8787537
cutoff 0.031293847 network size 494
STEP 73 ================================
prereg loss 2.9303935 reg_l1 81.68747 reg_l2 27.186346
loss 3.012081
cutoff 0.031779267 network size 492
STEP 74 ================================
prereg loss 2.5175185 reg_l1 81.61144 reg_l2 27.177534
loss 2.59913
cutoff 0.032142863 network size 489
STEP 75 ================================
prereg loss 2.5899944 reg_l1 81.49595 reg_l2 27.163494
loss 2.6714904
cutoff 0.03247024 network size 486
STEP 76 ================================
prereg loss 2.068355 reg_l1 81.37046 reg_l2 27.143976
loss 2.1497254
cutoff 0.03226919 network size 483
STEP 77 ================================
prereg loss 5.430624 reg_l1 81.2376 reg_l2 27.120352
loss 5.511862
cutoff 0.033240724 network size 479
STEP 78 ================================
prereg loss 8.408665 reg_l1 81.087814 reg_l2 27.107447
loss 8.489753
cutoff 0.033877563 network size 476
STEP 79 ================================
prereg loss 24.997736 reg_l1 80.99166 reg_l2 27.109459
loss 25.078728
cutoff 0.03437693 network size 473
STEP 80 ================================
prereg loss 20.668703 reg_l1 80.92713 reg_l2 27.133102
loss 20.74963
cutoff 0.03345673 network size 472
STEP 81 ================================
prereg loss 13.884712 reg_l1 80.95347 reg_l2 27.174757
loss 13.965666
cutoff 0.03324695 network size 471
STEP 82 ================================
prereg loss 8.589132 reg_l1 80.991615 reg_l2 27.226173
loss 8.670124
cutoff 0.031344406 network size 470
STEP 83 ================================
prereg loss 6.797218 reg_l1 81.04078 reg_l2 27.284546
loss 6.8782587
cutoff 0.030922012 network size 469
STEP 84 ================================
prereg loss 6.647761 reg_l1 81.092125 reg_l2 27.344458
loss 6.728853
cutoff 0.032050185 network size 468
STEP 85 ================================
prereg loss 6.8137317 reg_l1 81.13386 reg_l2 27.400295
loss 6.8948655
cutoff 0.031991363 network size 467
STEP 86 ================================
prereg loss 7.015216 reg_l1 81.16122 reg_l2 27.448488
loss 7.096377
cutoff 0.03219734 network size 465
STEP 87 ================================
prereg loss 9.484608 reg_l1 81.13171 reg_l2 27.482967
loss 9.56574
cutoff 0.035203367 network size 463
STEP 88 ================================
prereg loss 9.547832 reg_l1 81.06968 reg_l2 27.502718
loss 9.6289015
cutoff 0.034968525 network size 462
STEP 89 ================================
prereg loss 9.316504 reg_l1 81.01131 reg_l2 27.507288
loss 9.397515
cutoff 0.034856364 network size 461
STEP 90 ================================
prereg loss 8.73401 reg_l1 80.924034 reg_l2 27.495922
loss 8.814934
cutoff 0.035572343 network size 460
STEP 91 ================================
prereg loss 7.767374 reg_l1 80.810005 reg_l2 27.470036
loss 7.848184
cutoff 0.036438935 network size 456
STEP 92 ================================
prereg loss 6.632642 reg_l1 80.56687 reg_l2 27.429037
loss 6.7132087
cutoff 0.036669515 network size 455
STEP 93 ================================
prereg loss 5.5094976 reg_l1 80.42042 reg_l2 27.384754
loss 5.589918
cutoff 0.036380928 network size 454
STEP 94 ================================
prereg loss 4.700447 reg_l1 80.272095 reg_l2 27.338127
loss 4.7807193
cutoff 0.036203884 network size 453
STEP 95 ================================
prereg loss 4.195406 reg_l1 80.12996 reg_l2 27.293133
loss 4.275536
cutoff 0.035231106 network size 452
STEP 96 ================================
prereg loss 4.5730996 reg_l1 79.99977 reg_l2 27.252445
loss 4.6530995
cutoff 0.035753615 network size 451
STEP 97 ================================
prereg loss 4.778434 reg_l1 79.89789 reg_l2 27.224308
loss 4.8583317
cutoff 0.03704326 network size 449
STEP 98 ================================
prereg loss 4.8787856 reg_l1 79.78552 reg_l2 27.20741
loss 4.958571
cutoff 0.037879787 network size 447
STEP 99 ================================
prereg loss 4.8740315 reg_l1 79.69743 reg_l2 27.202757
loss 4.953729
cutoff 0.038247645 network size 446
STEP 100 ================================
prereg loss 4.551737 reg_l1 79.67048 reg_l2 27.210894
loss 4.6314073
cutoff 0.038734242 network size 443
2022-07-19T23:58:03.788

julia> steps!(100)
2022-07-19T23:58:53.090
STEP 1 ================================
prereg loss 4.1092296 reg_l1 79.58636 reg_l2 27.225906
loss 4.188816
STEP 2 ================================
prereg loss 3.6112127 reg_l1 79.63347 reg_l2 27.25347
loss 3.6908462
STEP 3 ================================
prereg loss 3.1443937 reg_l1 79.69227 reg_l2 27.287127
loss 3.224086
STEP 4 ================================
prereg loss 2.7871716 reg_l1 79.75816 reg_l2 27.324625
loss 2.8669298
STEP 5 ================================
prereg loss 2.5922592 reg_l1 79.826744 reg_l2 27.363361
loss 2.672086
STEP 6 ================================
prereg loss 2.560537 reg_l1 79.8935 reg_l2 27.400734
loss 2.6404307
STEP 7 ================================
prereg loss 2.6441085 reg_l1 79.9545 reg_l2 27.434319
loss 2.724063
STEP 8 ================================
prereg loss 2.769119 reg_l1 80.00646 reg_l2 27.46228
loss 2.8491254
STEP 9 ================================
prereg loss 2.8625042 reg_l1 80.045654 reg_l2 27.483046
loss 2.94255
STEP 10 ================================
prereg loss 2.866326 reg_l1 80.0711 reg_l2 27.496273
loss 2.9463973
STEP 11 ================================
prereg loss 2.7747533 reg_l1 80.083435 reg_l2 27.502405
loss 2.8548367
STEP 12 ================================
prereg loss 2.5926242 reg_l1 80.08373 reg_l2 27.502213
loss 2.672708
STEP 13 ================================
prereg loss 2.3662946 reg_l1 80.07425 reg_l2 27.496918
loss 2.446369
STEP 14 ================================
prereg loss 2.1490963 reg_l1 80.057976 reg_l2 27.48808
loss 2.229154
STEP 15 ================================
prereg loss 1.9848112 reg_l1 80.038124 reg_l2 27.477478
loss 2.0648494
STEP 16 ================================
prereg loss 1.8934739 reg_l1 80.01749 reg_l2 27.466614
loss 1.9734913
STEP 17 ================================
prereg loss 1.8700013 reg_l1 79.99859 reg_l2 27.456915
loss 1.9499999
STEP 18 ================================
prereg loss 1.8974726 reg_l1 79.98384 reg_l2 27.449678
loss 1.9774565
STEP 19 ================================
prereg loss 1.9378296 reg_l1 79.97542 reg_l2 27.445929
loss 2.017805
STEP 20 ================================
prereg loss 1.95564 reg_l1 79.97458 reg_l2 27.446291
loss 2.0356145
STEP 21 ================================
prereg loss 1.9286382 reg_l1 79.98144 reg_l2 27.450878
loss 2.0086195
STEP 22 ================================
prereg loss 1.8545543 reg_l1 79.99572 reg_l2 27.459444
loss 1.93455
STEP 23 ================================
prereg loss 1.7467706 reg_l1 80.01612 reg_l2 27.4714
loss 1.8267868
STEP 24 ================================
prereg loss 1.6305276 reg_l1 80.04114 reg_l2 27.48581
loss 1.7105688
STEP 25 ================================
prereg loss 1.5300093 reg_l1 80.06852 reg_l2 27.501596
loss 1.6100777
STEP 26 ================================
prereg loss 1.4601111 reg_l1 80.09633 reg_l2 27.517712
loss 1.5402075
STEP 27 ================================
prereg loss 1.4209449 reg_l1 80.122665 reg_l2 27.533154
loss 1.5010676
STEP 28 ================================
prereg loss 1.4084017 reg_l1 80.14613 reg_l2 27.547102
loss 1.4885478
STEP 29 ================================
prereg loss 1.4132936 reg_l1 80.16562 reg_l2 27.558975
loss 1.4934592
STEP 30 ================================
prereg loss 1.4205981 reg_l1 80.18047 reg_l2 27.56837
loss 1.5007787
STEP 31 ================================
prereg loss 1.4177467 reg_l1 80.19043 reg_l2 27.57513
loss 1.4979371
STEP 32 ================================
prereg loss 1.3978463 reg_l1 80.1957 reg_l2 27.579308
loss 1.478042
STEP 33 ================================
prereg loss 1.3613812 reg_l1 80.19675 reg_l2 27.581158
loss 1.4415779
STEP 34 ================================
prereg loss 1.3160558 reg_l1 80.19439 reg_l2 27.581188
loss 1.3962501
STEP 35 ================================
prereg loss 1.2707922 reg_l1 80.189705 reg_l2 27.57998
loss 1.350982
STEP 36 ================================
prereg loss 1.233413 reg_l1 80.184 reg_l2 27.578161
loss 1.313597
STEP 37 ================================
prereg loss 1.209616 reg_l1 80.17831 reg_l2 27.57636
loss 1.2897942
STEP 38 ================================
prereg loss 1.1983122 reg_l1 80.17369 reg_l2 27.575218
loss 1.2784859
STEP 39 ================================
prereg loss 1.1945792 reg_l1 80.17134 reg_l2 27.575294
loss 1.2747506
STEP 40 ================================
prereg loss 1.1933117 reg_l1 80.17189 reg_l2 27.576897
loss 1.2734836
STEP 41 ================================
prereg loss 1.1878752 reg_l1 80.175865 reg_l2 27.580359
loss 1.268051
STEP 42 ================================
prereg loss 1.1758987 reg_l1 80.183304 reg_l2 27.585651
loss 1.2560819
STEP 43 ================================
prereg loss 1.1585252 reg_l1 80.19355 reg_l2 27.592415
loss 1.2387187
STEP 44 ================================
prereg loss 1.1385586 reg_l1 80.20602 reg_l2 27.600353
loss 1.2187647
STEP 45 ================================
prereg loss 1.1193435 reg_l1 80.2199 reg_l2 27.609041
loss 1.1995634
STEP 46 ================================
prereg loss 1.1033131 reg_l1 80.23433 reg_l2 27.617937
loss 1.1835474
STEP 47 ================================
prereg loss 1.0908967 reg_l1 80.24856 reg_l2 27.62671
loss 1.1711453
STEP 48 ================================
prereg loss 1.0817878 reg_l1 80.26212 reg_l2 27.635035
loss 1.16205
STEP 49 ================================
prereg loss 1.0747049 reg_l1 80.27453 reg_l2 27.64267
loss 1.1549795
STEP 50 ================================
prereg loss 1.068112 reg_l1 80.285416 reg_l2 27.64944
loss 1.1483974
STEP 51 ================================
prereg loss 1.060533 reg_l1 80.29455 reg_l2 27.655283
loss 1.1408277
STEP 52 ================================
prereg loss 1.0516126 reg_l1 80.30202 reg_l2 27.660112
loss 1.1319146
STEP 53 ================================
prereg loss 1.0415592 reg_l1 80.3077 reg_l2 27.664032
loss 1.121867
STEP 54 ================================
prereg loss 1.0309315 reg_l1 80.31204 reg_l2 27.6672
loss 1.1112435
STEP 55 ================================
prereg loss 1.0205324 reg_l1 80.31509 reg_l2 27.669758
loss 1.1008475
STEP 56 ================================
prereg loss 1.0109757 reg_l1 80.317444 reg_l2 27.67193
loss 1.0912932
STEP 57 ================================
prereg loss 1.0025259 reg_l1 80.31917 reg_l2 27.673885
loss 1.0828451
STEP 58 ================================
prereg loss 0.9950067 reg_l1 80.32079 reg_l2 27.67583
loss 1.0753275
STEP 59 ================================
prereg loss 0.98806113 reg_l1 80.32253 reg_l2 27.677895
loss 1.0683837
STEP 60 ================================
prereg loss 0.98119843 reg_l1 80.32471 reg_l2 27.680254
loss 1.0615232
STEP 61 ================================
prereg loss 0.9739335 reg_l1 80.32738 reg_l2 27.682896
loss 1.0542608
STEP 62 ================================
prereg loss 0.96613514 reg_l1 80.33058 reg_l2 27.685894
loss 1.0464658
STEP 63 ================================
prereg loss 0.9579226 reg_l1 80.33425 reg_l2 27.689148
loss 1.0382569
STEP 64 ================================
prereg loss 0.9496546 reg_l1 80.33838 reg_l2 27.692669
loss 1.0299929
STEP 65 ================================
prereg loss 0.94165987 reg_l1 80.342766 reg_l2 27.696356
loss 1.0220027
STEP 66 ================================
prereg loss 0.9342977 reg_l1 80.34745 reg_l2 27.700113
loss 1.0146451
STEP 67 ================================
prereg loss 0.927482 reg_l1 80.352234 reg_l2 27.7039
loss 1.0078342
STEP 68 ================================
prereg loss 0.92107666 reg_l1 80.356834 reg_l2 27.707596
loss 1.0014335
STEP 69 ================================
prereg loss 0.91491127 reg_l1 80.361404 reg_l2 27.711193
loss 0.9952727
STEP 70 ================================
prereg loss 0.90881497 reg_l1 80.36572 reg_l2 27.714655
loss 0.9891807
STEP 71 ================================
prereg loss 0.9026575 reg_l1 80.36985 reg_l2 27.717947
loss 0.98302734
STEP 72 ================================
prereg loss 0.8964468 reg_l1 80.3738 reg_l2 27.721127
loss 0.97682065
STEP 73 ================================
prereg loss 0.89028734 reg_l1 80.3777 reg_l2 27.724195
loss 0.97066504
STEP 74 ================================
prereg loss 0.88426024 reg_l1 80.38132 reg_l2 27.72717
loss 0.9646416
STEP 75 ================================
prereg loss 0.8784476 reg_l1 80.38499 reg_l2 27.730125
loss 0.95883256
STEP 76 ================================
prereg loss 0.872901 reg_l1 80.388794 reg_l2 27.733118
loss 0.9532898
STEP 77 ================================
prereg loss 0.86759335 reg_l1 80.39261 reg_l2 27.736168
loss 0.94798595
STEP 78 ================================
prereg loss 0.8625635 reg_l1 80.39674 reg_l2 27.73937
loss 0.94296026
STEP 79 ================================
prereg loss 0.8575425 reg_l1 80.40114 reg_l2 27.742733
loss 0.93794364
STEP 80 ================================
prereg loss 0.8524614 reg_l1 80.40579 reg_l2 27.746273
loss 0.93286717
STEP 81 ================================
prereg loss 0.84732217 reg_l1 80.410675 reg_l2 27.749973
loss 0.9277328
STEP 82 ================================
prereg loss 0.84216475 reg_l1 80.41588 reg_l2 27.75379
loss 0.92258066
STEP 83 ================================
prereg loss 0.83697796 reg_l1 80.42112 reg_l2 27.757698
loss 0.91739905
STEP 84 ================================
prereg loss 0.83188975 reg_l1 80.42657 reg_l2 27.761658
loss 0.9123163
STEP 85 ================================
prereg loss 0.8269066 reg_l1 80.43194 reg_l2 27.765623
loss 0.90733856
STEP 86 ================================
prereg loss 0.82205194 reg_l1 80.43722 reg_l2 27.76957
loss 0.9024892
STEP 87 ================================
prereg loss 0.8173153 reg_l1 80.442345 reg_l2 27.773434
loss 0.89775765
STEP 88 ================================
prereg loss 0.8127478 reg_l1 80.447365 reg_l2 27.777172
loss 0.89319515
STEP 89 ================================
prereg loss 0.80825907 reg_l1 80.45211 reg_l2 27.78085
loss 0.8887112
STEP 90 ================================
prereg loss 0.8038148 reg_l1 80.45675 reg_l2 27.784409
loss 0.88427156
STEP 91 ================================
prereg loss 0.7994153 reg_l1 80.46119 reg_l2 27.787897
loss 0.8798765
STEP 92 ================================
prereg loss 0.79508585 reg_l1 80.46545 reg_l2 27.791279
loss 0.8755513
STEP 93 ================================
prereg loss 0.790832 reg_l1 80.469666 reg_l2 27.794617
loss 0.87130165
STEP 94 ================================
prereg loss 0.78669095 reg_l1 80.47382 reg_l2 27.797947
loss 0.8671648
STEP 95 ================================
prereg loss 0.782638 reg_l1 80.478 reg_l2 27.801304
loss 0.863116
STEP 96 ================================
prereg loss 0.77867055 reg_l1 80.48222 reg_l2 27.80466
loss 0.8591528
STEP 97 ================================
prereg loss 0.77475387 reg_l1 80.48651 reg_l2 27.808088
loss 0.8552404
STEP 98 ================================
prereg loss 0.7708649 reg_l1 80.491005 reg_l2 27.811617
loss 0.8513559
STEP 99 ================================
prereg loss 0.7669963 reg_l1 80.49566 reg_l2 27.815199
loss 0.847492
STEP 100 ================================
prereg loss 0.7631518 reg_l1 80.50035 reg_l2 27.818836
loss 0.8436522
2022-07-20T00:19:26.414
```

---

50 more sparsifying steps, then 100 steps in the standard non-sparsifying mode (converges nicely):

```
julia> # we are at 443 weights (out of initial 942)

julia> # let's do 50 sparsifying steps

julia>

julia> sparsifying_steps!(50)
2022-07-20T05:26:00.410
STEP 1 ================================
prereg loss 0.75933784 reg_l1 80.50514 reg_l2 27.822514
loss 0.839843
cutoff 0.017645923 network size 442
STEP 2 ================================
prereg loss 0.7539121 reg_l1 80.4925 reg_l2 27.825945
loss 0.8344046
cutoff 0.02873149 network size 440
STEP 3 ================================
prereg loss 0.91010165 reg_l1 80.44079 reg_l2 27.828135
loss 0.9905424
cutoff 0.032981873 network size 439
STEP 4 ================================
prereg loss 0.8976051 reg_l1 80.41619 reg_l2 27.830812
loss 0.9780213
cutoff 0.0360404 network size 437
STEP 5 ================================
prereg loss 1.8339795 reg_l1 80.35534 reg_l2 27.831944
loss 1.9143348
cutoff 0.037377458 network size 435
STEP 6 ================================
prereg loss 1.8642079 reg_l1 80.291145 reg_l2 27.832952
loss 1.944499
cutoff 0.038458697 network size 433
STEP 7 ================================
prereg loss 1.6906319 reg_l1 80.22456 reg_l2 27.833822
loss 1.7708564
cutoff 0.039277725 network size 431
STEP 8 ================================
prereg loss 1.4726993 reg_l1 80.15617 reg_l2 27.834587
loss 1.5528555
cutoff 0.040059097 network size 430
STEP 9 ================================
prereg loss 1.257406 reg_l1 80.12574 reg_l2 27.836683
loss 1.3375317
cutoff 0.040109254 network size 429
STEP 10 ================================
prereg loss 1.0753728 reg_l1 80.095314 reg_l2 27.838608
loss 1.1554681
cutoff 0.04087469 network size 427
STEP 11 ================================
prereg loss 0.9515977 reg_l1 80.0242 reg_l2 27.838562
loss 1.0316219
cutoff 0.04211138 network size 425
STEP 12 ================================
prereg loss 0.904952 reg_l1 79.951454 reg_l2 27.83809
loss 0.98490345
cutoff 0.04256707 network size 422
STEP 13 ================================
prereg loss 1.320137 reg_l1 79.83625 reg_l2 27.83553
loss 1.3999733
cutoff 0.043653637 network size 420
STEP 14 ================================
prereg loss 1.9978487 reg_l1 79.76002 reg_l2 27.833195
loss 2.0776088
cutoff 0.044372853 network size 417
STEP 15 ================================
prereg loss 1.7727865 reg_l1 79.633865 reg_l2 27.827251
loss 1.8524203
cutoff 0.044936493 network size 416
STEP 16 ================================
prereg loss 1.5661926 reg_l1 79.59219 reg_l2 27.82418
loss 1.6457849
cutoff 0.045439597 network size 415
STEP 17 ================================
prereg loss 1.4071772 reg_l1 79.54825 reg_l2 27.82057
loss 1.4867254
cutoff 0.046464667 network size 412
STEP 18 ================================
prereg loss 1.315146 reg_l1 79.40991 reg_l2 27.812376
loss 1.3945559
cutoff 0.047160912 network size 411
STEP 19 ================================
prereg loss 1.2540131 reg_l1 79.36404 reg_l2 27.808807
loss 1.3333771
cutoff 0.04782965 network size 409
STEP 20 ================================
prereg loss 1.2304304 reg_l1 79.27161 reg_l2 27.803524
loss 1.3097019
cutoff 0.048304576 network size 408
STEP 21 ================================
prereg loss 1.2245101 reg_l1 79.22875 reg_l2 27.801268
loss 1.3037388
cutoff 0.04960675 network size 407
STEP 22 ================================
prereg loss 1.215845 reg_l1 79.18712 reg_l2 27.799744
loss 1.2950321
cutoff 0.050303817 network size 403
STEP 23 ================================
prereg loss 1.2041801 reg_l1 78.99881 reg_l2 27.791878
loss 1.2831789
cutoff 0.050822966 network size 400
STEP 24 ================================
prereg loss 1.0983623 reg_l1 78.8615 reg_l2 27.787588
loss 1.1772238
cutoff 0.05122489 network size 399
STEP 25 ================================
prereg loss 1.0621675 reg_l1 78.82944 reg_l2 27.790094
loss 1.1409969
cutoff 0.05201127 network size 395
STEP 26 ================================
prereg loss 40.138737 reg_l1 78.64499 reg_l2 27.785967
loss 40.21738
cutoff 0.052532233 network size 392
STEP 27 ================================
prereg loss 37.672947 reg_l1 78.5578 reg_l2 27.811619
loss 37.751503
cutoff 0.053048912 network size 390
STEP 28 ================================
prereg loss 33.938984 reg_l1 78.55832 reg_l2 27.861431
loss 34.017544
cutoff 0.05428071 network size 387
STEP 29 ================================
prereg loss 29.82376 reg_l1 78.52338 reg_l2 27.922081
loss 29.902283
cutoff 0.054851916 network size 386
STEP 30 ================================
prereg loss 25.787048 reg_l1 78.609985 reg_l2 27.99941
loss 25.865658
cutoff 0.05307073 network size 385
STEP 31 ================================
prereg loss 22.173584 reg_l1 78.70496 reg_l2 28.084114
loss 22.252289
cutoff 0.05567501 network size 382
STEP 32 ================================
prereg loss 19.419546 reg_l1 78.69073 reg_l2 28.16643
loss 19.498238
cutoff 0.0526844 network size 381
STEP 33 ================================
prereg loss 17.050732 reg_l1 78.78491 reg_l2 28.255598
loss 17.129517
cutoff 0.050094355 network size 380
STEP 34 ================================
prereg loss 15.003143 reg_l1 78.87412 reg_l2 28.342093
loss 15.082018
cutoff 0.056296635 network size 377
STEP 35 ================================
prereg loss 14.116916 reg_l1 78.83003 reg_l2 28.414045
loss 14.195745
cutoff 0.05685302 network size 376
STEP 36 ================================
prereg loss 13.550176 reg_l1 78.875046 reg_l2 28.480755
loss 13.62905
cutoff 0.05805416 network size 373
STEP 37 ================================
prereg loss 13.100994 reg_l1 78.77756 reg_l2 28.52771
loss 13.179771
cutoff 0.057826295 network size 372
STEP 38 ================================
prereg loss 12.3798485 reg_l1 78.77294 reg_l2 28.56925
loss 12.458621
cutoff 0.05692234 network size 371
STEP 39 ================================
prereg loss 11.273335 reg_l1 78.74921 reg_l2 28.599878
loss 11.352085
cutoff 0.058354232 network size 369
STEP 40 ================================
prereg loss 10.193247 reg_l1 78.645966 reg_l2 28.614494
loss 10.271893
cutoff 0.058362003 network size 366
STEP 41 ================================
prereg loss 15.658984 reg_l1 78.46624 reg_l2 28.613358
loss 15.737451
cutoff 0.05900657 network size 365
STEP 42 ================================
prereg loss 15.436278 reg_l1 78.38836 reg_l2 28.60872
loss 15.514667
cutoff 0.060208403 network size 360
STEP 43 ================================
prereg loss 14.853868 reg_l1 78.05664 reg_l2 28.581163
loss 14.931924
cutoff 0.061226472 network size 358
STEP 44 ================================
prereg loss 26.099504 reg_l1 77.89515 reg_l2 28.557762
loss 26.177399
cutoff 0.0619468 network size 355
STEP 45 ================================
prereg loss 23.219496 reg_l1 77.657166 reg_l2 28.523111
loss 23.297153
cutoff 0.06264687 network size 354
STEP 46 ================================
prereg loss 18.016317 reg_l1 77.54005 reg_l2 28.494555
loss 18.093857
cutoff 0.06342425 network size 352
STEP 47 ================================
prereg loss 12.230963 reg_l1 77.344734 reg_l2 28.456865
loss 12.308308
cutoff 0.062393762 network size 351
STEP 48 ================================
prereg loss 10.019116 reg_l1 77.20348 reg_l2 28.421001
loss 10.09632
cutoff 0.06312763 network size 349
STEP 49 ================================
prereg loss 9.29136 reg_l1 77.00576 reg_l2 28.384453
loss 9.368365
cutoff 0.06425684 network size 347
STEP 50 ================================
prereg loss 8.84513 reg_l1 76.80996 reg_l2 28.351328
loss 8.92194
cutoff 0.064929076 network size 346
2022-07-20T05:35:15.362

julia> steps!(100)
2022-07-20T05:35:22.469
STEP 1 ================================
prereg loss 8.304662 reg_l1 76.681114 reg_l2 28.325808
loss 8.381343
STEP 2 ================================
prereg loss 7.712193 reg_l1 76.62085 reg_l2 28.307817
loss 7.788814
STEP 3 ================================
prereg loss 7.0591364 reg_l1 76.56417 reg_l2 28.292696
loss 7.1357007
STEP 4 ================================
prereg loss 6.400145 reg_l1 76.510666 reg_l2 28.279985
loss 6.476656
STEP 5 ================================
prereg loss 5.8118887 reg_l1 76.46016 reg_l2 28.26909
loss 5.888349
STEP 6 ================================
prereg loss 5.3284855 reg_l1 76.41229 reg_l2 28.259563
loss 5.4048977
STEP 7 ================================
prereg loss 4.964253 reg_l1 76.36688 reg_l2 28.25078
loss 5.04062
STEP 8 ================================
prereg loss 4.7286677 reg_l1 76.32448 reg_l2 28.242556
loss 4.804992
STEP 9 ================================
prereg loss 4.6090055 reg_l1 76.285065 reg_l2 28.234558
loss 4.6852903
STEP 10 ================================
prereg loss 4.562223 reg_l1 76.24853 reg_l2 28.226618
loss 4.6384716
STEP 11 ================================
prereg loss 4.542633 reg_l1 76.215065 reg_l2 28.218622
loss 4.6188483
STEP 12 ================================
prereg loss 4.5101604 reg_l1 76.184784 reg_l2 28.210546
loss 4.586345
STEP 13 ================================
prereg loss 4.438387 reg_l1 76.15741 reg_l2 28.20235
loss 4.5145445
STEP 14 ================================
prereg loss 4.3235455 reg_l1 76.13365 reg_l2 28.194504
loss 4.399679
STEP 15 ================================
prereg loss 4.1868286 reg_l1 76.113846 reg_l2 28.187435
loss 4.2629423
STEP 16 ================================
prereg loss 4.057777 reg_l1 76.0981 reg_l2 28.181559
loss 4.133875
STEP 17 ================================
prereg loss 3.961461 reg_l1 76.08699 reg_l2 28.177446
loss 4.037548
STEP 18 ================================
prereg loss 3.9103425 reg_l1 76.08098 reg_l2 28.175728
loss 3.9864235
STEP 19 ================================
prereg loss 3.8978014 reg_l1 76.08091 reg_l2 28.177135
loss 3.9738822
STEP 20 ================================
prereg loss 3.8990767 reg_l1 76.08682 reg_l2 28.18188
loss 3.9751635
STEP 21 ================================
prereg loss 3.8885276 reg_l1 76.098495 reg_l2 28.190125
loss 3.964626
STEP 22 ================================
prereg loss 3.849469 reg_l1 76.11611 reg_l2 28.201946
loss 3.925585
STEP 23 ================================
prereg loss 3.7787855 reg_l1 76.13889 reg_l2 28.217018
loss 3.8549244
STEP 24 ================================
prereg loss 3.6817348 reg_l1 76.16589 reg_l2 28.234869
loss 3.7579007
STEP 25 ================================
prereg loss 3.5709472 reg_l1 76.19634 reg_l2 28.254967
loss 3.6471436
STEP 26 ================================
prereg loss 3.4601796 reg_l1 76.2294 reg_l2 28.276867
loss 3.536409
STEP 27 ================================
prereg loss 3.364557 reg_l1 76.26428 reg_l2 28.29989
loss 3.4408214
STEP 28 ================================
prereg loss 3.2921693 reg_l1 76.29975 reg_l2 28.32341
loss 3.368469
STEP 29 ================================
prereg loss 3.2397048 reg_l1 76.335106 reg_l2 28.346834
loss 3.31604
STEP 30 ================================
prereg loss 3.2006772 reg_l1 76.36968 reg_l2 28.369726
loss 3.277047
STEP 31 ================================
prereg loss 3.1679754 reg_l1 76.40299 reg_l2 28.39181
loss 3.2443783
STEP 32 ================================
prereg loss 3.1358879 reg_l1 76.43494 reg_l2 28.412931
loss 3.2123227
STEP 33 ================================
prereg loss 3.1003158 reg_l1 76.46492 reg_l2 28.43292
loss 3.1767807
STEP 34 ================================
prereg loss 3.0601301 reg_l1 76.493126 reg_l2 28.451857
loss 3.1366231
STEP 35 ================================
prereg loss 3.0179913 reg_l1 76.51951 reg_l2 28.469776
loss 3.0945108
STEP 36 ================================
prereg loss 2.9775236 reg_l1 76.5441 reg_l2 28.48672
loss 3.0540676
STEP 37 ================================
prereg loss 2.9404967 reg_l1 76.566986 reg_l2 28.502825
loss 3.0170636
STEP 38 ================================
prereg loss 2.9085588 reg_l1 76.58824 reg_l2 28.518223
loss 2.985147
STEP 39 ================================
prereg loss 2.882224 reg_l1 76.60799 reg_l2 28.53307
loss 2.958832
STEP 40 ================================
prereg loss 2.85896 reg_l1 76.62646 reg_l2 28.547497
loss 2.9355865
STEP 41 ================================
prereg loss 2.836495 reg_l1 76.64383 reg_l2 28.561604
loss 2.9131389
STEP 42 ================================
prereg loss 2.8140106 reg_l1 76.6601 reg_l2 28.575485
loss 2.8906708
STEP 43 ================================
prereg loss 2.7892718 reg_l1 76.675575 reg_l2 28.589256
loss 2.8659475
STEP 44 ================================
prereg loss 2.761443 reg_l1 76.69035 reg_l2 28.602945
loss 2.8381333
STEP 45 ================================
prereg loss 2.7310925 reg_l1 76.70448 reg_l2 28.6166
loss 2.807797
STEP 46 ================================
prereg loss 2.6995707 reg_l1 76.71811 reg_l2 28.630194
loss 2.7762887
STEP 47 ================================
prereg loss 2.6681616 reg_l1 76.73122 reg_l2 28.643707
loss 2.7448928
STEP 48 ================================
prereg loss 2.638028 reg_l1 76.743965 reg_l2 28.657131
loss 2.714772
STEP 49 ================================
prereg loss 2.6097357 reg_l1 76.756386 reg_l2 28.670393
loss 2.6864922
STEP 50 ================================
prereg loss 2.583032 reg_l1 76.768486 reg_l2 28.683498
loss 2.6598003
STEP 51 ================================
prereg loss 2.5579004 reg_l1 76.780426 reg_l2 28.696463
loss 2.6346807
STEP 52 ================================
prereg loss 2.5334337 reg_l1 76.79224 reg_l2 28.709225
loss 2.610226
STEP 53 ================================
prereg loss 2.509271 reg_l1 76.80404 reg_l2 28.721823
loss 2.5860748
STEP 54 ================================
prereg loss 2.4851172 reg_l1 76.81578 reg_l2 28.73417
loss 2.561933
STEP 55 ================================
prereg loss 2.4610019 reg_l1 76.82746 reg_l2 28.746283
loss 2.5378294
STEP 56 ================================
prereg loss 2.4371808 reg_l1 76.83906 reg_l2 28.758156
loss 2.5140197
STEP 57 ================================
prereg loss 2.4139962 reg_l1 76.85066 reg_l2 28.76983
loss 2.4908469
STEP 58 ================================
prereg loss 2.3916984 reg_l1 76.86226 reg_l2 28.781357
loss 2.4685607
STEP 59 ================================
prereg loss 2.370385 reg_l1 76.87387 reg_l2 28.79277
loss 2.4472587
STEP 60 ================================
prereg loss 2.3501024 reg_l1 76.88556 reg_l2 28.80412
loss 2.426988
STEP 61 ================================
prereg loss 2.3304617 reg_l1 76.89748 reg_l2 28.81551
loss 2.4073591
STEP 62 ================================
prereg loss 2.3110945 reg_l1 76.90957 reg_l2 28.826992
loss 2.388004
STEP 63 ================================
prereg loss 2.291721 reg_l1 76.921974 reg_l2 28.83858
loss 2.368643
STEP 64 ================================
prereg loss 2.2721827 reg_l1 76.93451 reg_l2 28.85029
loss 2.3491173
STEP 65 ================================
prereg loss 2.252473 reg_l1 76.9475 reg_l2 28.862173
loss 2.3294206
STEP 66 ================================
prereg loss 2.2327023 reg_l1 76.96072 reg_l2 28.874208
loss 2.309663
STEP 67 ================================
prereg loss 2.2130208 reg_l1 76.974266 reg_l2 28.886404
loss 2.2899952
STEP 68 ================================
prereg loss 2.193588 reg_l1 76.988045 reg_l2 28.898733
loss 2.270576
STEP 69 ================================
prereg loss 2.1745098 reg_l1 77.002106 reg_l2 28.91114
loss 2.2515118
STEP 70 ================================
prereg loss 2.1558123 reg_l1 77.016396 reg_l2 28.923653
loss 2.2328286
STEP 71 ================================
prereg loss 2.1374452 reg_l1 77.03073 reg_l2 28.936182
loss 2.2144759
STEP 72 ================================
prereg loss 2.1193357 reg_l1 77.045204 reg_l2 28.948702
loss 2.1963809
STEP 73 ================================
prereg loss 2.101621 reg_l1 77.059654 reg_l2 28.961199
loss 2.1786807
STEP 74 ================================
prereg loss 2.0836213 reg_l1 77.07034 reg_l2 28.96916
loss 2.1606915
STEP 75 ================================
prereg loss 2.0659275 reg_l1 77.081215 reg_l2 28.97733
loss 2.1430087
STEP 76 ================================
prereg loss 2.0483732 reg_l1 77.09214 reg_l2 28.985708
loss 2.1254654
STEP 77 ================================
prereg loss 2.0310168 reg_l1 77.10302 reg_l2 28.994228
loss 2.10812
STEP 78 ================================
prereg loss 2.0139225 reg_l1 77.1139 reg_l2 29.002796
loss 2.0910363
STEP 79 ================================
prereg loss 1.9972692 reg_l1 77.124626 reg_l2 29.011402
loss 2.0743937
STEP 80 ================================
prereg loss 1.9812595 reg_l1 77.135216 reg_l2 29.020119
loss 2.0583947
STEP 81 ================================
prereg loss 1.9656126 reg_l1 77.14559 reg_l2 29.028868
loss 2.0427582
STEP 82 ================================
prereg loss 1.9501638 reg_l1 77.15572 reg_l2 29.037699
loss 2.0273197
STEP 83 ================================
prereg loss 1.9348682 reg_l1 77.16565 reg_l2 29.04652
loss 2.012034
STEP 84 ================================
prereg loss 1.9196967 reg_l1 77.175354 reg_l2 29.055412
loss 1.9968721
STEP 85 ================================
prereg loss 1.9046398 reg_l1 77.18476 reg_l2 29.064297
loss 1.9818246
STEP 86 ================================
prereg loss 1.8897077 reg_l1 77.19405 reg_l2 29.073215
loss 1.9669018
STEP 87 ================================
prereg loss 1.8749243 reg_l1 77.203186 reg_l2 29.08214
loss 1.9521275
STEP 88 ================================
prereg loss 1.8603058 reg_l1 77.21219 reg_l2 29.091091
loss 1.937518
STEP 89 ================================
prereg loss 1.8458799 reg_l1 77.22111 reg_l2 29.10005
loss 1.9231011
STEP 90 ================================
prereg loss 1.8316481 reg_l1 77.22987 reg_l2 29.109041
loss 1.908878
STEP 91 ================================
prereg loss 1.8176092 reg_l1 77.23853 reg_l2 29.11799
loss 1.8948478
STEP 92 ================================
prereg loss 1.8037452 reg_l1 77.247154 reg_l2 29.12696
loss 1.8809923
STEP 93 ================================
prereg loss 1.7900475 reg_l1 77.25577 reg_l2 29.135908
loss 1.8673033
STEP 94 ================================
prereg loss 1.7765104 reg_l1 77.26423 reg_l2 29.144848
loss 1.8537745
STEP 95 ================================
prereg loss 1.7631333 reg_l1 77.27271 reg_l2 29.15375
loss 1.840406
STEP 96 ================================
prereg loss 1.749916 reg_l1 77.28113 reg_l2 29.162617
loss 1.8271971
STEP 97 ================================
prereg loss 1.7368661 reg_l1 77.28949 reg_l2 29.17147
loss 1.8141556
STEP 98 ================================
prereg loss 1.723989 reg_l1 77.29781 reg_l2 29.180286
loss 1.8012868
STEP 99 ================================
prereg loss 1.7112823 reg_l1 77.30602 reg_l2 29.189058
loss 1.7885883
STEP 100 ================================
prereg loss 1.6987361 reg_l1 77.31417 reg_l2 29.197786
loss 1.7760502
2022-07-20T05:51:54.296
```

Time to reduce the situations when we drop multiple weights at once:

```
julia> # let's reduce the situations when we cut multiple weights at once

julia> function sparsifying_steps!(n_steps)
           printlog_v(io, now())
           for i in 1:n_steps
               printlog_v(io, "STEP ", i, " ================================")
               training_step!()
               lim = 1.0001f0*min_abs_dict(trainable["network_matrix"])
               trim_network(trainable, opt, lim)
               printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
           end
           printlog_v(io, now())
       end
sparsifying_steps! (generic function with 1 method)
```

Eventually, I should just cut those situations altogether.

Let's do 20 sparsifying steps:

```
julia> sparsifying_steps!(20)
2022-07-20T08:04:07.674
STEP 1 ================================
prereg loss 1.686342 reg_l1 77.322296 reg_l2 29.206442
loss 1.7636642
cutoff 0.032668527 network size 345
STEP 2 ================================
prereg loss 1.706832 reg_l1 77.29765 reg_l2 29.21402
loss 1.7841297
cutoff 0.039154977 network size 344
STEP 3 ================================
prereg loss 1.6889707 reg_l1 77.26572 reg_l2 29.220148
loss 1.7662364
cutoff 0.043744996 network size 343
STEP 4 ================================
prereg loss 1.786321 reg_l1 77.22845 reg_l2 29.22516
loss 1.8635495
cutoff 0.04397694 network size 342
STEP 5 ================================
prereg loss 1.7463385 reg_l1 77.18926 reg_l2 29.228682
loss 1.8235278
cutoff 0.04843623 network size 341
STEP 6 ================================
prereg loss 1.7001631 reg_l1 77.14461 reg_l2 29.230768
loss 1.7773077
cutoff 0.0507353 network size 340
STEP 7 ================================
prereg loss 1.6741017 reg_l1 77.097466 reg_l2 29.232168
loss 1.7511991
cutoff 0.052056964 network size 339
STEP 8 ================================
prereg loss 1.6325121 reg_l1 77.04977 reg_l2 29.233438
loss 1.7095618
cutoff 0.0541421 network size 338
STEP 9 ================================
prereg loss 1.6249648 reg_l1 77.00141 reg_l2 29.234974
loss 1.7019663
cutoff 0.05407267 network size 337
STEP 10 ================================
prereg loss 1.6375675 reg_l1 76.953865 reg_l2 29.236834
loss 1.7145214
cutoff 0.054038305 network size 336
STEP 11 ================================
prereg loss 1.6099769 reg_l1 76.90793 reg_l2 29.239511
loss 1.6868849
cutoff 0.056838874 network size 335
STEP 12 ================================
prereg loss 1.5876215 reg_l1 76.8609 reg_l2 29.242851
loss 1.6644824
cutoff 0.057547685 network size 334
STEP 13 ================================
prereg loss 1.5546942 reg_l1 76.81475 reg_l2 29.247068
loss 1.631509
cutoff 0.0596611 network size 333
STEP 14 ================================
prereg loss 1.527912 reg_l1 76.76796 reg_l2 29.252047
loss 1.60468
cutoff 0.061858095 network size 332
STEP 15 ================================
prereg loss 1.5052534 reg_l1 76.720085 reg_l2 29.257555
loss 1.5819736
cutoff 0.062345184 network size 331
STEP 16 ================================
prereg loss 1.4789643 reg_l1 76.672264 reg_l2 29.26358
loss 1.5556366
cutoff 0.06397073 network size 330
STEP 17 ================================
prereg loss 1.4642652 reg_l1 76.62326 reg_l2 29.26982
loss 1.5408885
cutoff 0.065625794 network size 328
STEP 18 ================================
prereg loss 7.616159 reg_l1 76.507 reg_l2 29.271715
loss 7.692666
cutoff 0.065763526 network size 327
STEP 19 ================================
prereg loss 6.8809915 reg_l1 76.4582 reg_l2 29.278725
loss 6.9574494
cutoff 0.0658644 network size 326
STEP 20 ================================
prereg loss 5.7308645 reg_l1 76.41015 reg_l2 29.286074
loss 5.807275
cutoff 0.066518284 network size 325
2022-07-20T08:07:19.665

julia> # still, multiple weights dropped at once is what yields instabilities
```

100 standard steps

```
julia> steps!(100)
2022-07-20T08:20:49.341
STEP 1 ================================
prereg loss 13.449715 reg_l1 76.36128 reg_l2 29.293219
loss 13.526076
STEP 2 ================================
prereg loss 11.720385 reg_l1 76.396935 reg_l2 29.318676
loss 11.796782
STEP 3 ================================
prereg loss 9.906363 reg_l1 76.4457 reg_l2 29.354885
loss 9.982808
STEP 4 ================================
prereg loss 8.258975 reg_l1 76.50303 reg_l2 29.398674
loss 8.335478
STEP 5 ================================
prereg loss 6.9030766 reg_l1 76.56437 reg_l2 29.446934
loss 6.979641
STEP 6 ================================
prereg loss 5.8338532 reg_l1 76.626045 reg_l2 29.496695
loss 5.910479
STEP 7 ================================
prereg loss 5.0730553 reg_l1 76.684845 reg_l2 29.545612
loss 5.14974
STEP 8 ================================
prereg loss 4.609315 reg_l1 76.73737 reg_l2 29.591059
loss 4.6860523
STEP 9 ================================
prereg loss 4.3729672 reg_l1 76.78084 reg_l2 29.630869
loss 4.449748
STEP 10 ================================
prereg loss 4.331225 reg_l1 76.81337 reg_l2 29.663439
loss 4.408038
STEP 11 ================================
prereg loss 4.4220247 reg_l1 76.83263 reg_l2 29.686964
loss 4.4988575
STEP 12 ================================
prereg loss 4.5670586 reg_l1 76.83742 reg_l2 29.700232
loss 4.643896
STEP 13 ================================
prereg loss 4.6922555 reg_l1 76.82708 reg_l2 29.702642
loss 4.7690825
STEP 14 ================================
prereg loss 4.7404985 reg_l1 76.80183 reg_l2 29.694223
loss 4.8173003
STEP 15 ================================
prereg loss 4.6819386 reg_l1 76.76275 reg_l2 29.675703
loss 4.7587013
STEP 16 ================================
prereg loss 4.51213 reg_l1 76.71151 reg_l2 29.64833
loss 4.5888414
STEP 17 ================================
prereg loss 4.245661 reg_l1 76.65025 reg_l2 29.613525
loss 4.322311
STEP 18 ================================
prereg loss 3.9172366 reg_l1 76.58123 reg_l2 29.573008
loss 3.9938178
STEP 19 ================================
prereg loss 3.5665188 reg_l1 76.50685 reg_l2 29.528528
loss 3.6430256
STEP 20 ================================
prereg loss 3.232433 reg_l1 76.4296 reg_l2 29.481796
loss 3.3088627
STEP 21 ================================
prereg loss 2.9490197 reg_l1 76.35164 reg_l2 29.43442
loss 3.0253713
STEP 22 ================================
prereg loss 2.7377834 reg_l1 76.2752 reg_l2 29.387934
loss 2.8140585
STEP 23 ================================
prereg loss 2.6055667 reg_l1 76.20213 reg_l2 29.343613
loss 2.681769
STEP 24 ================================
prereg loss 2.5490801 reg_l1 76.13396 reg_l2 29.302588
loss 2.625214
STEP 25 ================================
prereg loss 2.554843 reg_l1 76.07217 reg_l2 29.265852
loss 2.6309152
STEP 26 ================================
prereg loss 2.6028402 reg_l1 76.01772 reg_l2 29.234045
loss 2.6788578
STEP 27 ================================
prereg loss 2.671563 reg_l1 75.97154 reg_l2 29.207748
loss 2.7475345
STEP 28 ================================
prereg loss 2.7343597 reg_l1 75.93382 reg_l2 29.187126
loss 2.8102937
STEP 29 ================================
prereg loss 2.776475 reg_l1 75.90463 reg_l2 29.172178
loss 2.8523796
STEP 30 ================================
prereg loss 2.7878177 reg_l1 75.88358 reg_l2 29.16264
loss 2.8637013
STEP 31 ================================
prereg loss 2.7656107 reg_l1 75.87021 reg_l2 29.158148
loss 2.841481
STEP 32 ================================
prereg loss 2.7130318 reg_l1 75.86364 reg_l2 29.15821
loss 2.7888954
STEP 33 ================================
prereg loss 2.6373582 reg_l1 75.86317 reg_l2 29.16229
loss 2.7132213
STEP 34 ================================
prereg loss 2.5485704 reg_l1 75.86776 reg_l2 29.16973
loss 2.624438
STEP 35 ================================
prereg loss 2.4570498 reg_l1 75.876495 reg_l2 29.17985
loss 2.5329263
STEP 36 ================================
prereg loss 2.3724928 reg_l1 75.888306 reg_l2 29.192003
loss 2.4483812
STEP 37 ================================
prereg loss 2.3041596 reg_l1 75.90216 reg_l2 29.205454
loss 2.3800619
STEP 38 ================================
prereg loss 2.2532198 reg_l1 75.91686 reg_l2 29.219435
loss 2.3291366
STEP 39 ================================
prereg loss 2.218349 reg_l1 75.931564 reg_l2 29.233402
loss 2.2942805
STEP 40 ================================
prereg loss 2.198627 reg_l1 75.945496 reg_l2 29.246838
loss 2.2745724
STEP 41 ================================
prereg loss 2.189522 reg_l1 75.957985 reg_l2 29.25927
loss 2.26548
STEP 42 ================================
prereg loss 2.1875007 reg_l1 75.968575 reg_l2 29.270348
loss 2.2634692
STEP 43 ================================
prereg loss 2.187872 reg_l1 75.976944 reg_l2 29.279903
loss 2.2638488
STEP 44 ================================
prereg loss 2.1873477 reg_l1 75.98288 reg_l2 29.287868
loss 2.2633305
STEP 45 ================================
prereg loss 2.1839128 reg_l1 75.98641 reg_l2 29.294193
loss 2.2598991
STEP 46 ================================
prereg loss 2.1763055 reg_l1 75.987526 reg_l2 29.2989
loss 2.252293
STEP 47 ================================
prereg loss 2.1643796 reg_l1 75.98629 reg_l2 29.302128
loss 2.240366
STEP 48 ================================
prereg loss 2.148712 reg_l1 75.98308 reg_l2 29.304024
loss 2.224695
STEP 49 ================================
prereg loss 2.1298761 reg_l1 75.97807 reg_l2 29.304775
loss 2.2058542
STEP 50 ================================
prereg loss 2.1094275 reg_l1 75.97159 reg_l2 29.30454
loss 2.185399
STEP 51 ================================
prereg loss 2.0886807 reg_l1 75.963974 reg_l2 29.303589
loss 2.1646447
STEP 52 ================================
prereg loss 2.069165 reg_l1 75.95551 reg_l2 29.302141
loss 2.1451206
STEP 53 ================================
prereg loss 2.0513964 reg_l1 75.946655 reg_l2 29.300503
loss 2.127343
STEP 54 ================================
prereg loss 2.0361063 reg_l1 75.937775 reg_l2 29.29889
loss 2.112044
STEP 55 ================================
prereg loss 2.023445 reg_l1 75.929115 reg_l2 29.297464
loss 2.099374
STEP 56 ================================
prereg loss 2.0132434 reg_l1 75.92093 reg_l2 29.296398
loss 2.0891643
STEP 57 ================================
prereg loss 2.005409 reg_l1 75.913345 reg_l2 29.295792
loss 2.0813224
STEP 58 ================================
prereg loss 1.9993378 reg_l1 75.906654 reg_l2 29.295784
loss 2.0752444
STEP 59 ================================
prereg loss 1.9940445 reg_l1 75.900894 reg_l2 29.296385
loss 2.0699453
STEP 60 ================================
prereg loss 1.9887599 reg_l1 75.89617 reg_l2 29.29767
loss 2.064656
STEP 61 ================================
prereg loss 1.9832146 reg_l1 75.8925 reg_l2 29.299643
loss 2.059107
STEP 62 ================================
prereg loss 1.9768765 reg_l1 75.88977 reg_l2 29.302256
loss 2.0527663
STEP 63 ================================
prereg loss 1.9693915 reg_l1 75.88802 reg_l2 29.305513
loss 2.0452795
STEP 64 ================================
prereg loss 1.9607731 reg_l1 75.8871 reg_l2 29.309345
loss 2.0366602
STEP 65 ================================
prereg loss 1.9512844 reg_l1 75.886925 reg_l2 29.313618
loss 2.0271714
STEP 66 ================================
prereg loss 1.9412606 reg_l1 75.8873 reg_l2 29.318274
loss 2.0171478
STEP 67 ================================
prereg loss 1.9312781 reg_l1 75.88819 reg_l2 29.323177
loss 2.0071664
STEP 68 ================================
prereg loss 1.921646 reg_l1 75.88914 reg_l2 29.328152
loss 1.9975351
STEP 69 ================================
prereg loss 1.9126047 reg_l1 75.8901 reg_l2 29.333113
loss 1.9884948
STEP 70 ================================
prereg loss 1.9043638 reg_l1 75.890945 reg_l2 29.337957
loss 1.9802547
STEP 71 ================================
prereg loss 1.8970315 reg_l1 75.8915 reg_l2 29.342598
loss 1.972923
STEP 72 ================================
prereg loss 1.8905593 reg_l1 75.89167 reg_l2 29.34696
loss 1.9664509
STEP 73 ================================
prereg loss 1.8848091 reg_l1 75.89135 reg_l2 29.350952
loss 1.9607005
STEP 74 ================================
prereg loss 1.8795952 reg_l1 75.890434 reg_l2 29.354527
loss 1.9554856
STEP 75 ================================
prereg loss 1.8747025 reg_l1 75.88898 reg_l2 29.357723
loss 1.9505914
STEP 76 ================================
prereg loss 1.8699476 reg_l1 75.88683 reg_l2 29.36048
loss 1.9458344
STEP 77 ================================
prereg loss 1.8651768 reg_l1 75.88409 reg_l2 29.362835
loss 1.9410609
STEP 78 ================================
prereg loss 1.8603121 reg_l1 75.8808 reg_l2 29.364784
loss 1.9361929
STEP 79 ================================
prereg loss 1.8553323 reg_l1 75.87697 reg_l2 29.366405
loss 1.9312092
STEP 80 ================================
prereg loss 1.8502626 reg_l1 75.87281 reg_l2 29.367746
loss 1.9261354
STEP 81 ================================
prereg loss 1.8451536 reg_l1 75.86826 reg_l2 29.368824
loss 1.9210218
STEP 82 ================================
prereg loss 1.8400432 reg_l1 75.86349 reg_l2 29.369728
loss 1.9159067
STEP 83 ================================
prereg loss 1.8350261 reg_l1 75.85847 reg_l2 29.37046
loss 1.9108846
STEP 84 ================================
prereg loss 1.8302168 reg_l1 75.853325 reg_l2 29.371115
loss 1.9060701
STEP 85 ================================
prereg loss 1.8256625 reg_l1 75.84817 reg_l2 29.371733
loss 1.9015107
STEP 86 ================================
prereg loss 1.8213557 reg_l1 75.84304 reg_l2 29.372334
loss 1.8971988
STEP 87 ================================
prereg loss 1.8172722 reg_l1 75.83786 reg_l2 29.372955
loss 1.89311
STEP 88 ================================
prereg loss 1.8133222 reg_l1 75.832825 reg_l2 29.373661
loss 1.889155
STEP 89 ================================
prereg loss 1.8094788 reg_l1 75.82791 reg_l2 29.374426
loss 1.8853067
STEP 90 ================================
prereg loss 1.8057029 reg_l1 75.82317 reg_l2 29.375315
loss 1.8815261
STEP 91 ================================
prereg loss 1.8019282 reg_l1 75.818596 reg_l2 29.37629
loss 1.8777468
STEP 92 ================================
prereg loss 1.7981185 reg_l1 75.81424 reg_l2 29.377392
loss 1.8739327
STEP 93 ================================
prereg loss 1.7942623 reg_l1 75.810005 reg_l2 29.378622
loss 1.8700722
STEP 94 ================================
prereg loss 1.7903639 reg_l1 75.80609 reg_l2 29.379967
loss 1.86617
STEP 95 ================================
prereg loss 1.7864388 reg_l1 75.80216 reg_l2 29.381369
loss 1.862241
STEP 96 ================================
prereg loss 1.782497 reg_l1 75.79848 reg_l2 29.38288
loss 1.8582956
STEP 97 ================================
prereg loss 1.7785772 reg_l1 75.794945 reg_l2 29.384445
loss 1.8543721
STEP 98 ================================
prereg loss 1.7747114 reg_l1 75.79151 reg_l2 29.386068
loss 1.8505028
STEP 99 ================================
prereg loss 1.7709194 reg_l1 75.788086 reg_l2 29.387716
loss 1.8467076
STEP 100 ================================
prereg loss 1.7672101 reg_l1 75.78461 reg_l2 29.389334
loss 1.8429947
2022-07-20T08:36:26.806
```

**Let's really put an end to cutting multiple weights at once:**

```
julia> # true sparsification

julia> function sparsecopy(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, lim::Float32)
           y = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()
           for i in keys(x)
               for j in keys(x[i])
                   for m in keys(x[i][j])
                       for n in keys(x[i][j][m])
                           if abs(x[i][j][m][n]) > lim
                               link!(y, i, j, m, n, x[i][j][m][n])
           end end end end end
           y
       end
sparsecopy (generic function with 2 methods)

julia> # loaded sparsification

julia> function sparsecopy(x::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}, lim::Float32,
                           load1::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
                           load2::Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}})
           y = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()
           next1 = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()
           next2 = Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}}()
           for i in keys(x)
               for j in keys(x[i])
                   for m in keys(x[i][j])
                       for n in keys(x[i][j][m])
                           if abs(x[i][j][m][n]) > lim
                               link!(y, i, j, m, n, x[i][j][m][n])
                               link!(next1, i, j, m, n, load1[i][j][m][n])
                               link!(next2, i, j, m, n, load2[i][j][m][n])
           end end end end end
           (y, next1, next2)
       end
sparsecopy (generic function with 2 methods)

julia> function trim_network(trainable::Dict{String, Dict{String}}, opt::TreeADAM, lim::Float32)
           (y, next1, next2) = sparsecopy(trainable["network_matrix"], lim, opt.mt, opt.vt)
           trainable["network_matrix"] = y
           opt.mt = next1
           opt.vt = next2
       end
trim_network (generic function with 1 method)

julia> function sparsifying_steps!(n_steps)
           printlog_v(io, now())
           for i in 1:n_steps
               printlog_v(io, "STEP ", i, " ================================")
               training_step!()
               lim = min_abs_dict(trainable["network_matrix"])
               trim_network(trainable, opt, lim)
               printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
           end
           printlog_v(io, now())
       end
sparsifying_steps! (generic function with 1 method)

julia> sparsifying_steps!(20)
2022-07-20T08:39:59.712
STEP 1 ================================
prereg loss 1.7635819 reg_l1 75.781105 reg_l2 29.390911
loss 1.839363
cutoff 0.016467331 network size 324
STEP 2 ================================
prereg loss 1.7551043 reg_l1 75.76102 reg_l2 29.392155
loss 1.8308654
cutoff 0.037715834 network size 323
STEP 3 ================================
prereg loss 1.7477497 reg_l1 75.7201 reg_l2 29.392216
loss 1.8234698
cutoff 0.046766404 network size 322
STEP 4 ================================
prereg loss 1.7444887 reg_l1 75.67041 reg_l2 29.391516
loss 1.8201591
cutoff 0.053467195 network size 321
STEP 5 ================================
prereg loss 1.7377717 reg_l1 75.61423 reg_l2 29.390156
loss 1.813386
cutoff 0.055798408 network size 320
STEP 6 ================================
prereg loss 1.733686 reg_l1 75.55642 reg_l2 29.388876
loss 1.8092424
cutoff 0.057773408 network size 319
STEP 7 ================================
prereg loss 1.8127204 reg_l1 75.49738 reg_l2 29.387674
loss 1.8882178
cutoff 0.06216917 network size 318
STEP 8 ================================
prereg loss 2.6717322 reg_l1 75.43468 reg_l2 29.38649
loss 2.7471669
cutoff 0.065267995 network size 317
STEP 9 ================================
prereg loss 2.627143 reg_l1 75.36397 reg_l2 29.38167
loss 2.7025068
cutoff 0.066536985 network size 316
STEP 10 ================================
prereg loss 2.549914 reg_l1 75.28755 reg_l2 29.373886
loss 2.6252015
cutoff 0.06698828 network size 315
STEP 11 ================================
prereg loss 2.4494731 reg_l1 75.207085 reg_l2 29.363758
loss 2.5246801
cutoff 0.066986166 network size 314
STEP 12 ================================
prereg loss 2.3371096 reg_l1 75.123795 reg_l2 29.35187
loss 2.4122334
cutoff 0.06699563 network size 313
STEP 13 ================================
prereg loss 2.220445 reg_l1 75.03871 reg_l2 29.338915
loss 2.2954836
cutoff 0.06735825 network size 312
STEP 14 ================================
prereg loss 2.1494799 reg_l1 74.95489 reg_l2 29.326876
loss 2.2244349
cutoff 0.06775914 network size 311
STEP 15 ================================
prereg loss 2.080501 reg_l1 74.87229 reg_l2 29.31602
loss 2.1553733
cutoff 0.06841715 network size 310
STEP 16 ================================
prereg loss 2.0171735 reg_l1 74.792015 reg_l2 29.306711
loss 2.0919654
cutoff 0.06844359 network size 309
STEP 17 ================================
prereg loss 1.959438 reg_l1 74.714554 reg_l2 29.299234
loss 2.0341525
cutoff 0.06852482 network size 308
STEP 18 ================================
prereg loss 1.9076036 reg_l1 74.640015 reg_l2 29.293604
loss 1.9822437
cutoff 0.06861703 network size 307
STEP 19 ================================
prereg loss 2.345999 reg_l1 74.568344 reg_l2 29.289839
loss 2.4205673
cutoff 0.07008104 network size 306
STEP 20 ================================
prereg loss 2.3012013 reg_l1 74.49952 reg_l2 29.288446
loss 2.375701
cutoff 0.070306405 network size 305
2022-07-20T08:43:04.881
```

Let's do more

```
julia> steps!(20)
2022-07-20T08:44:50.085
STEP 1 ================================
prereg loss 2.0837212 reg_l1 74.43405 reg_l2 29.289228
loss 2.1581552
STEP 2 ================================
prereg loss 2.0214012 reg_l1 74.442245 reg_l2 29.297035
loss 2.0958433
STEP 3 ================================
prereg loss 1.9584934 reg_l1 74.452965 reg_l2 29.306412
loss 2.0329463
STEP 4 ================================
prereg loss 1.8990332 reg_l1 74.4661 reg_l2 29.317326
loss 1.9734993
STEP 5 ================================
prereg loss 1.8492192 reg_l1 74.48078 reg_l2 29.329168
loss 1.9237
STEP 6 ================================
prereg loss 1.8106781 reg_l1 74.49601 reg_l2 29.3413
loss 1.8851742
STEP 7 ================================
prereg loss 1.7831631 reg_l1 74.51099 reg_l2 29.353117
loss 1.8576741
STEP 8 ================================
prereg loss 1.7642167 reg_l1 74.5251 reg_l2 29.36423
loss 1.8387418
STEP 9 ================================
prereg loss 1.7498384 reg_l1 74.537796 reg_l2 29.374348
loss 1.8243761
STEP 10 ================================
prereg loss 1.7388436 reg_l1 74.54867 reg_l2 29.383083
loss 1.8133923
STEP 11 ================================
prereg loss 1.728132 reg_l1 74.55729 reg_l2 29.390247
loss 1.8026893
STEP 12 ================================
prereg loss 1.7157981 reg_l1 74.56356 reg_l2 29.395744
loss 1.7903616
STEP 13 ================================
prereg loss 1.7013848 reg_l1 74.56743 reg_l2 29.399624
loss 1.7759522
STEP 14 ================================
prereg loss 1.6860598 reg_l1 74.56921 reg_l2 29.402042
loss 1.760629
STEP 15 ================================
prereg loss 1.671137 reg_l1 74.56924 reg_l2 29.403315
loss 1.7457062
STEP 16 ================================
prereg loss 1.6583371 reg_l1 74.567924 reg_l2 29.403742
loss 1.732905
STEP 17 ================================
prereg loss 1.6488851 reg_l1 74.56579 reg_l2 29.403631
loss 1.7234509
STEP 18 ================================
prereg loss 1.6429946 reg_l1 74.56317 reg_l2 29.40331
loss 1.7175578
STEP 19 ================================
prereg loss 1.6401529 reg_l1 74.5606 reg_l2 29.403133
loss 1.7147136
STEP 20 ================================
prereg loss 1.639411 reg_l1 74.55837 reg_l2 29.4033
loss 1.7139693
2022-07-20T08:47:45.967

julia> sparsifying_steps!(20)
2022-07-20T08:48:06.014
STEP 1 ================================
prereg loss 1.6395383 reg_l1 74.55677 reg_l2 29.403996
loss 1.7140951
cutoff 0.070295 network size 304
STEP 2 ================================
prereg loss 1.6394331 reg_l1 74.48568 reg_l2 29.400404
loss 1.7139188
cutoff 0.07040647 network size 303
STEP 3 ================================
prereg loss 1.6384901 reg_l1 74.41535 reg_l2 29.397455
loss 1.7129054
cutoff 0.07046135 network size 302
STEP 4 ================================
prereg loss 1.6362823 reg_l1 74.34591 reg_l2 29.395147
loss 1.7106283
cutoff 0.072634004 network size 301
STEP 5 ================================
prereg loss 1.6329238 reg_l1 74.275 reg_l2 29.393112
loss 1.7071989
cutoff 0.072749466 network size 300
STEP 6 ================================
prereg loss 1.6285573 reg_l1 74.20451 reg_l2 29.391464
loss 1.7027619
cutoff 0.07290156 network size 299
STEP 7 ================================
prereg loss 1.6233858 reg_l1 74.13437 reg_l2 29.390102
loss 1.6975201
cutoff 0.07376908 network size 298
STEP 8 ================================
prereg loss 1.6177086 reg_l1 74.06352 reg_l2 29.388737
loss 1.6917721
cutoff 0.07492769 network size 297
STEP 9 ================================
prereg loss 1.6117622 reg_l1 73.99158 reg_l2 29.38722
loss 1.6857537
cutoff 0.075025514 network size 296
STEP 10 ================================
prereg loss 1.6057272 reg_l1 73.91935 reg_l2 29.385544
loss 1.6796465
cutoff 0.07596628 network size 295
STEP 11 ================================
prereg loss 1.5998256 reg_l1 73.84578 reg_l2 29.383448
loss 1.6736714
cutoff 0.0760644 network size 294
STEP 12 ================================
prereg loss 1.5940636 reg_l1 73.771614 reg_l2 29.380968
loss 1.6678352
cutoff 0.07702034 network size 293
STEP 13 ================================
prereg loss 1.5884794 reg_l1 73.69591 reg_l2 29.377924
loss 1.6621753
cutoff 0.077726156 network size 292
STEP 14 ================================
prereg loss 1.584656 reg_l1 73.61878 reg_l2 29.37427
loss 1.6582748
cutoff 0.078883715 network size 291
STEP 15 ================================
prereg loss 1.5799046 reg_l1 73.540215 reg_l2 29.370222
loss 1.6534448
cutoff 0.07923853 network size 290
STEP 16 ================================
prereg loss 8.575257 reg_l1 73.46094 reg_l2 29.36591
loss 8.648718
cutoff 0.079969645 network size 289
STEP 17 ================================
prereg loss 8.422117 reg_l1 73.38937 reg_l2 29.36711
loss 8.495506
cutoff 0.07994215 network size 288
STEP 18 ================================
prereg loss 9.392753 reg_l1 73.32609 reg_l2 29.374136
loss 9.466079
cutoff 0.080212235 network size 287
STEP 19 ================================
prereg loss 9.091561 reg_l1 73.26395 reg_l2 29.381802
loss 9.164825
cutoff 0.080778785 network size 286
STEP 20 ================================
prereg loss 8.751788 reg_l1 73.20153 reg_l2 29.389303
loss 8.824989
cutoff 0.08092181 network size 285
2022-07-20T08:50:56.993

julia> steps!(20)
2022-07-20T08:51:04.194
STEP 1 ================================
prereg loss 8.365945 reg_l1 73.13816 reg_l2 29.396257
loss 8.439083
STEP 2 ================================
prereg loss 7.960313 reg_l1 73.15518 reg_l2 29.409061
loss 8.033468
STEP 3 ================================
prereg loss 7.5486784 reg_l1 73.17143 reg_l2 29.421091
loss 7.62185
STEP 4 ================================
prereg loss 7.138423 reg_l1 73.18665 reg_l2 29.432081
loss 7.21161
STEP 5 ================================
prereg loss 6.734838 reg_l1 73.20105 reg_l2 29.442156
loss 6.808039
STEP 6 ================================
prereg loss 6.3402195 reg_l1 73.21483 reg_l2 29.451483
loss 6.4134345
STEP 7 ================================
prereg loss 5.9494686 reg_l1 73.22826 reg_l2 29.460295
loss 6.022697
STEP 8 ================================
prereg loss 5.5740232 reg_l1 73.24252 reg_l2 29.46953
loss 5.647266
STEP 9 ================================
prereg loss 5.2162757 reg_l1 73.25782 reg_l2 29.479395
loss 5.2895336
STEP 10 ================================
prereg loss 4.858698 reg_l1 73.27482 reg_l2 29.490253
loss 4.9319725
STEP 11 ================================
prereg loss 4.495397 reg_l1 73.294556 reg_l2 29.502985
loss 4.5686917
STEP 12 ================================
prereg loss 4.123797 reg_l1 73.31796 reg_l2 29.518276
loss 4.197115
STEP 13 ================================
prereg loss 3.7677727 reg_l1 73.345665 reg_l2 29.536568
loss 3.8411183
STEP 14 ================================
prereg loss 3.450452 reg_l1 73.37666 reg_l2 29.557257
loss 3.5238287
STEP 15 ================================
prereg loss 3.1788437 reg_l1 73.410034 reg_l2 29.579737
loss 3.2522538
STEP 16 ================================
prereg loss 2.9560306 reg_l1 73.4447 reg_l2 29.603294
loss 3.0294752
STEP 17 ================================
prereg loss 2.7830536 reg_l1 73.479515 reg_l2 29.627203
loss 2.856533
STEP 18 ================================
prereg loss 2.6567433 reg_l1 73.51365 reg_l2 29.65077
loss 2.730257
STEP 19 ================================
prereg loss 2.572198 reg_l1 73.54621 reg_l2 29.673443
loss 2.645744
STEP 20 ================================
prereg loss 2.5212736 reg_l1 73.57649 reg_l2 29.6947
loss 2.59485
2022-07-20T08:53:50.699

julia> steps!(20)
2022-07-20T08:53:57.298
STEP 1 ================================
prereg loss 2.4954717 reg_l1 73.60386 reg_l2 29.714048
loss 2.5690756
STEP 2 ================================
prereg loss 2.4866629 reg_l1 73.627846 reg_l2 29.73117
loss 2.5602908
STEP 3 ================================
prereg loss 2.485555 reg_l1 73.648 reg_l2 29.74565
loss 2.559203
STEP 4 ================================
prereg loss 2.4842415 reg_l1 73.6642 reg_l2 29.7574
loss 2.5579057
STEP 5 ================================
prereg loss 2.4781713 reg_l1 73.67655 reg_l2 29.7665
loss 2.551848
STEP 6 ================================
prereg loss 2.4646437 reg_l1 73.685326 reg_l2 29.773067
loss 2.5383291
STEP 7 ================================
prereg loss 2.4426534 reg_l1 73.69089 reg_l2 29.777315
loss 2.5163443
STEP 8 ================================
prereg loss 2.4144301 reg_l1 73.69362 reg_l2 29.779552
loss 2.4881237
STEP 9 ================================
prereg loss 2.3926408 reg_l1 73.69435 reg_l2 29.780226
loss 2.4663353
STEP 10 ================================
prereg loss 2.3647077 reg_l1 73.6939 reg_l2 29.779799
loss 2.4384017
STEP 11 ================================
prereg loss 2.3292859 reg_l1 73.692764 reg_l2 29.77861
loss 2.4029787
STEP 12 ================================
prereg loss 2.2865508 reg_l1 73.69118 reg_l2 29.77686
loss 2.360242
STEP 13 ================================
prereg loss 2.2384057 reg_l1 73.689476 reg_l2 29.77479
loss 2.3120952
STEP 14 ================================
prereg loss 2.1868758 reg_l1 73.687775 reg_l2 29.772598
loss 2.2605636
STEP 15 ================================
prereg loss 2.1340241 reg_l1 73.68637 reg_l2 29.770422
loss 2.2077105
STEP 16 ================================
prereg loss 2.081839 reg_l1 73.685234 reg_l2 29.768358
loss 2.1555243
STEP 17 ================================
prereg loss 2.0321352 reg_l1 73.68446 reg_l2 29.766428
loss 2.1058197
STEP 18 ================================
prereg loss 1.9861923 reg_l1 73.683945 reg_l2 29.764688
loss 2.0598762
STEP 19 ================================
prereg loss 1.9450731 reg_l1 73.68373 reg_l2 29.763107
loss 2.0187569
STEP 20 ================================
prereg loss 1.9100645 reg_l1 73.683685 reg_l2 29.761707
loss 1.9837482
2022-07-20T08:56:48.402

julia> steps!(20)
2022-07-20T08:57:20.922
STEP 1 ================================
prereg loss 1.8812013 reg_l1 73.684105 reg_l2 29.760605
loss 1.9548854
STEP 2 ================================
prereg loss 1.8579265 reg_l1 73.68471 reg_l2 29.759739
loss 1.9316112
STEP 3 ================================
prereg loss 1.8397388 reg_l1 73.68543 reg_l2 29.759033
loss 1.9134243
STEP 4 ================================
prereg loss 1.8259783 reg_l1 73.686264 reg_l2 29.758465
loss 1.8996645
STEP 5 ================================
prereg loss 1.8154775 reg_l1 73.68704 reg_l2 29.757921
loss 1.8891646
STEP 6 ================================
prereg loss 1.8072323 reg_l1 73.68767 reg_l2 29.757324
loss 1.8809199
STEP 7 ================================
prereg loss 1.8002962 reg_l1 73.688 reg_l2 29.75664
loss 1.8739842
STEP 8 ================================
prereg loss 1.793854 reg_l1 73.68797 reg_l2 29.7558
loss 1.867542
STEP 9 ================================
prereg loss 1.7871869 reg_l1 73.6876 reg_l2 29.75481
loss 1.8608744
STEP 10 ================================
prereg loss 1.7800058 reg_l1 73.686935 reg_l2 29.753717
loss 1.8536928
STEP 11 ================================
prereg loss 1.7719938 reg_l1 73.68609 reg_l2 29.752598
loss 1.8456799
STEP 12 ================================
prereg loss 1.763107 reg_l1 73.68511 reg_l2 29.751436
loss 1.8367921
STEP 13 ================================
prereg loss 1.7534362 reg_l1 73.68404 reg_l2 29.750298
loss 1.8271203
STEP 14 ================================
prereg loss 1.7431592 reg_l1 73.68294 reg_l2 29.7492
loss 1.8168421
STEP 15 ================================
prereg loss 1.7324976 reg_l1 73.68187 reg_l2 29.748228
loss 1.8061794
STEP 16 ================================
prereg loss 1.7216845 reg_l1 73.68095 reg_l2 29.747402
loss 1.7953655
STEP 17 ================================
prereg loss 1.7109402 reg_l1 73.68021 reg_l2 29.74674
loss 1.7846204
STEP 18 ================================
prereg loss 1.7004368 reg_l1 73.67977 reg_l2 29.74633
loss 1.7741166
STEP 19 ================================
prereg loss 1.6903061 reg_l1 73.67971 reg_l2 29.746191
loss 1.7639858
STEP 20 ================================
prereg loss 1.6805657 reg_l1 73.679955 reg_l2 29.746283
loss 1.7542456
2022-07-20T09:00:07.634
```

Still some instability, so let's add an option of even more rarely applied sparsification:

```
julia> # one sparsifying step out of N

julia> function interleaving_steps!(n_steps, N=2)
           printlog_v(io, now())
           for i in 1:n_steps
               printlog_v(io, "STEP ", i, " ================================")
               training_step!()
               if i%N == 1
                   lim = min_abs_dict(trainable["network_matrix"])
                   trim_network(trainable, opt, lim)
                   printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
               end
           end
           printlog_v(io, now())
       end
interleaving_steps! (generic function with 2 methods)

julia> interleaving_steps!(20)
2022-07-20T09:07:54.802
STEP 1 ================================
prereg loss 1.6713611 reg_l1 73.68058 reg_l2 29.746647
loss 1.7450417
cutoff 0.072006315 network size 284
STEP 2 ================================
prereg loss 1.6494412 reg_l1 73.60955 reg_l2 29.742046
loss 1.7230508
STEP 3 ================================
prereg loss 1.6424904 reg_l1 73.61057 reg_l2 29.742643
loss 1.7161009
cutoff 0.07772665 network size 283
STEP 4 ================================
prereg loss 1.6463403 reg_l1 73.53378 reg_l2 29.737156
loss 1.719874
STEP 5 ================================
prereg loss 1.641266 reg_l1 73.534424 reg_l2 29.73749
loss 1.7148004
cutoff 0.078943044 network size 282
STEP 6 ================================
prereg loss 1.6620328 reg_l1 73.45588 reg_l2 29.73143
loss 1.7354888
STEP 7 ================================
prereg loss 1.6566789 reg_l1 73.45459 reg_l2 29.730345
loss 1.7301335
cutoff 0.08140979 network size 281
STEP 8 ================================
prereg loss 1.7100089 reg_l1 73.37045 reg_l2 29.721582
loss 1.7833793
STEP 9 ================================
prereg loss 1.7040238 reg_l1 73.37148 reg_l2 29.722095
loss 1.7773954
cutoff 0.08237441 network size 280
STEP 10 ================================
prereg loss 1.6928575 reg_l1 73.29341 reg_l2 29.718151
loss 1.766151
STEP 11 ================================
prereg loss 1.6780068 reg_l1 73.30039 reg_l2 29.723005
loss 1.7513071
cutoff 0.08448705 network size 279
STEP 12 ================================
prereg loss 4.582714 reg_l1 73.22512 reg_l2 29.72235
loss 4.655939
STEP 13 ================================
prereg loss 4.253177 reg_l1 73.2359 reg_l2 29.73155
loss 4.326413
cutoff 0.084564954 network size 278
STEP 14 ================================
prereg loss 3.72297 reg_l1 73.163124 reg_l2 29.735607
loss 3.796133
STEP 15 ================================
prereg loss 3.1127393 reg_l1 73.17526 reg_l2 29.748056
loss 3.1859145
cutoff 0.08514429 network size 277
STEP 16 ================================
prereg loss 2.537181 reg_l1 73.10181 reg_l2 29.753672
loss 2.6102827
STEP 17 ================================
prereg loss 2.0842948 reg_l1 73.11235 reg_l2 29.766247
loss 2.157407
cutoff 0.085654125 network size 276
STEP 18 ================================
prereg loss 1.8032515 reg_l1 73.035515 reg_l2 29.770355
loss 1.876287
STEP 19 ================================
prereg loss 1.6990224 reg_l1 73.042076 reg_l2 29.780048
loss 1.7720644
cutoff 0.08619813 network size 275
STEP 20 ================================
prereg loss 1.7392155 reg_l1 72.959915 reg_l2 29.780014
loss 1.8121754
2022-07-20T09:10:41.457

julia> interleaving_steps!(30, 3)
2022-07-20T09:11:04.281
STEP 1 ================================
prereg loss 1.86624 reg_l1 72.961334 reg_l2 29.784851
loss 1.9392014
cutoff 0.08734821 network size 274
STEP 2 ================================
prereg loss 2.0146036 reg_l1 72.87306 reg_l2 29.779406
loss 2.0874767
STEP 3 ================================
prereg loss 2.133538 reg_l1 72.870285 reg_l2 29.779238
loss 2.2064083
STEP 4 ================================
prereg loss 2.1885428 reg_l1 72.86626 reg_l2 29.777096
loss 2.261409
cutoff 0.08728116 network size 273
STEP 5 ================================
prereg loss 2.1711829 reg_l1 72.77452 reg_l2 29.766031
loss 2.2439573
STEP 6 ================================
prereg loss 2.0941212 reg_l1 72.77047 reg_l2 29.761974
loss 2.1668916
STEP 7 ================================
prereg loss 1.9825376 reg_l1 72.76785 reg_l2 29.758186
loss 2.0553055
cutoff 0.08685043 network size 272
STEP 8 ================================
prereg loss 143.81482 reg_l1 72.6803 reg_l2 29.747875
loss 143.8875
STEP 9 ================================
prereg loss 140.32797 reg_l1 72.69734 reg_l2 29.760647
loss 140.40067
STEP 10 ================================
prereg loss 133.90611 reg_l1 72.72671 reg_l2 29.785427
loss 133.97884
cutoff 0.08801637 network size 271
STEP 11 ================================
prereg loss 124.8544 reg_l1 72.680084 reg_l2 29.814245
loss 124.92708
STEP 12 ================================
prereg loss 113.992096 reg_l1 72.73422 reg_l2 29.86319
loss 114.06483
STEP 13 ================================
prereg loss 101.98855 reg_l1 72.801346 reg_l2 29.924583
loss 102.06135
cutoff 0.08614151 network size 270
STEP 14 ================================
prereg loss 80.75532 reg_l1 72.79598 reg_l2 29.991144
loss 80.82812
STEP 15 ================================
prereg loss 71.33318 reg_l1 72.89321 reg_l2 30.076303
loss 71.406075
STEP 16 ================================
prereg loss 64.42593 reg_l1 73.000656 reg_l2 30.169065
loss 64.498924
cutoff 0.088147886 network size 269
STEP 17 ================================
prereg loss 60.364517 reg_l1 73.02384 reg_l2 30.256252
loss 60.437542
STEP 18 ================================
prereg loss 58.94922 reg_l1 73.12904 reg_l2 30.34506
loss 59.022346
STEP 19 ================================
prereg loss 59.237892 reg_l1 73.21355 reg_l2 30.416452
loss 59.311104
cutoff 0.07840801 network size 268
STEP 20 ================================
prereg loss 61.407436 reg_l1 73.18448 reg_l2 30.453058
loss 61.48062
STEP 21 ================================
prereg loss 61.00072 reg_l1 73.182205 reg_l2 30.453558
loss 61.073902
STEP 22 ================================
prereg loss 58.50825 reg_l1 73.12931 reg_l2 30.415487
loss 58.58138
cutoff 0.08619932 network size 267
STEP 23 ================================
prereg loss 54.708576 reg_l1 72.95043 reg_l2 30.340673
loss 54.78153
STEP 24 ================================
prereg loss 51.16217 reg_l1 72.831955 reg_l2 30.254576
loss 51.235
STEP 25 ================================
prereg loss 48.567913 reg_l1 72.697815 reg_l2 30.158682
loss 48.64061
cutoff 0.08095026 network size 266
STEP 26 ================================
prereg loss 47.15755 reg_l1 72.477196 reg_l2 30.05423
loss 47.230026
STEP 27 ================================
prereg loss 46.844578 reg_l1 72.34548 reg_l2 29.961403
loss 46.916924
STEP 28 ================================
prereg loss 47.05481 reg_l1 72.22295 reg_l2 29.877186
loss 47.127033
cutoff 0.08729636 network size 265
STEP 29 ================================
prereg loss 47.280994 reg_l1 72.02652 reg_l2 29.796688
loss 47.35302
STEP 30 ================================
prereg loss 47.136894 reg_l1 71.93272 reg_l2 29.73632
loss 47.208828
2022-07-20T09:15:05.683
```

This is a reminder that we really need better checkpointing; sometimes one does want to go back.

```
julia> interleaving_steps!(100, 10)
2022-07-20T09:17:41.493
STEP 1 ================================
prereg loss 46.449177 reg_l1 71.85508 reg_l2 29.688604
loss 46.52103
cutoff 0.08537797 network size 264
STEP 2 ================================
prereg loss 48.497883 reg_l1 71.707535 reg_l2 29.64546
loss 48.56959
STEP 3 ================================
prereg loss 46.682964 reg_l1 71.661606 reg_l2 29.621466
loss 46.754627
STEP 4 ================================
prereg loss 44.600124 reg_l1 71.628876 reg_l2 29.607338
loss 44.671753
STEP 5 ================================
prereg loss 42.529022 reg_l1 71.60681 reg_l2 29.601192
loss 42.600628
STEP 6 ================================
prereg loss 40.757896 reg_l1 71.592865 reg_l2 29.60103
loss 40.82949
STEP 7 ================================
prereg loss 39.37407 reg_l1 71.58444 reg_l2 29.604797
loss 39.445652
STEP 8 ================================
prereg loss 38.418053 reg_l1 71.57915 reg_l2 29.610592
loss 38.48963
STEP 9 ================================
prereg loss 37.83701 reg_l1 71.574844 reg_l2 29.616709
loss 37.908585
STEP 10 ================================
prereg loss 37.500748 reg_l1 71.56969 reg_l2 29.621666
loss 37.57232
STEP 11 ================================
prereg loss 37.24546 reg_l1 71.562355 reg_l2 29.624378
loss 37.317024
cutoff 0.08248634 network size 263
STEP 12 ================================
prereg loss 36.848507 reg_l1 71.46957 reg_l2 29.61737
loss 36.919975
STEP 13 ================================
prereg loss 36.37062 reg_l1 71.45706 reg_l2 29.614532
loss 36.442078
STEP 14 ================================
prereg loss 35.71546 reg_l1 71.4422 reg_l2 29.609226
loss 35.786903
STEP 15 ================================
prereg loss 34.914406 reg_l1 71.425705 reg_l2 29.602015
loss 34.985832
STEP 16 ================================
prereg loss 34.0366 reg_l1 71.40857 reg_l2 29.593756
loss 34.108006
STEP 17 ================================
prereg loss 33.16235 reg_l1 71.391884 reg_l2 29.585411
loss 33.23374
STEP 18 ================================
prereg loss 32.355087 reg_l1 71.37695 reg_l2 29.578047
loss 32.426464
STEP 19 ================================
prereg loss 31.652958 reg_l1 71.364944 reg_l2 29.572605
loss 31.724323
STEP 20 ================================
prereg loss 31.060339 reg_l1 71.35669 reg_l2 29.56988
loss 31.131695
STEP 21 ================================
prereg loss 30.55694 reg_l1 71.35308 reg_l2 29.57053
loss 30.628294
cutoff 0.06682566 network size 262
STEP 22 ================================
prereg loss 30.018234 reg_l1 71.28777 reg_l2 29.570515
loss 30.089521
STEP 23 ================================
prereg loss 29.607805 reg_l1 71.29732 reg_l2 29.580074
loss 29.679102
STEP 24 ================================
prereg loss 29.145706 reg_l1 71.31311 reg_l2 29.594448
loss 29.21702
STEP 25 ================================
prereg loss 28.616898 reg_l1 71.334724 reg_l2 29.613358
loss 28.688232
STEP 26 ================================
prereg loss 28.023384 reg_l1 71.36162 reg_l2 29.636435
loss 28.094746
STEP 27 ================================
prereg loss 27.380684 reg_l1 71.39309 reg_l2 29.6632
loss 27.452078
STEP 28 ================================
prereg loss 26.712053 reg_l1 71.42839 reg_l2 29.69317
loss 26.783482
STEP 29 ================================
prereg loss 26.042747 reg_l1 71.46671 reg_l2 29.72567
loss 26.114214
STEP 30 ================================
prereg loss 25.394096 reg_l1 71.50714 reg_l2 29.760101
loss 25.465603
STEP 31 ================================
prereg loss 24.779633 reg_l1 71.54887 reg_l2 29.795914
loss 24.851181
cutoff 0.08619014 network size 261
STEP 32 ================================
prereg loss 24.15189 reg_l1 71.50502 reg_l2 29.825056
loss 24.223394
STEP 33 ================================
prereg loss 23.606209 reg_l1 71.54887 reg_l2 29.862417
loss 23.677757
STEP 34 ================================
prereg loss 23.08581 reg_l1 71.592514 reg_l2 29.899948
loss 23.157402
STEP 35 ================================
prereg loss 22.578312 reg_l1 71.63565 reg_l2 29.93736
loss 22.649948
STEP 36 ================================
prereg loss 22.073212 reg_l1 71.67805 reg_l2 29.97447
loss 22.14489
STEP 37 ================================
prereg loss 21.562008 reg_l1 71.71984 reg_l2 30.011242
loss 21.633728
STEP 38 ================================
prereg loss 21.04082 reg_l1 71.76107 reg_l2 30.04782
loss 21.11258
STEP 39 ================================
prereg loss 20.50959 reg_l1 71.80213 reg_l2 30.084332
loss 20.581392
STEP 40 ================================
prereg loss 19.972387 reg_l1 71.84335 reg_l2 30.121075
loss 20.044231
STEP 41 ================================
prereg loss 19.432457 reg_l1 71.88525 reg_l2 30.15838
loss 19.504343
cutoff 0.08907908 network size 260
STEP 42 ================================
prereg loss 18.892933 reg_l1 71.83918 reg_l2 30.188648
loss 18.964771
STEP 43 ================================
prereg loss 18.355064 reg_l1 71.88382 reg_l2 30.228102
loss 18.426949
STEP 44 ================================
prereg loss 17.816832 reg_l1 71.93047 reg_l2 30.269106
loss 17.888762
STEP 45 ================================
prereg loss 17.27637 reg_l1 71.979485 reg_l2 30.311949
loss 17.348349
STEP 46 ================================
prereg loss 16.729158 reg_l1 72.03105 reg_l2 30.3568
loss 16.80119
STEP 47 ================================
prereg loss 16.172241 reg_l1 72.08537 reg_l2 30.403793
loss 16.244326
STEP 48 ================================
prereg loss 15.604539 reg_l1 72.14249 reg_l2 30.453003
loss 15.6766815
STEP 49 ================================
prereg loss 15.023807 reg_l1 72.202156 reg_l2 30.504368
loss 15.096008
STEP 50 ================================
prereg loss 14.432046 reg_l1 72.26432 reg_l2 30.557734
loss 14.504311
STEP 51 ================================
prereg loss 13.834895 reg_l1 72.328606 reg_l2 30.61295
loss 13.907224
cutoff 0.09014694 network size 259
STEP 52 ================================
prereg loss 13.235478 reg_l1 72.304474 reg_l2 30.661585
loss 13.307783
STEP 53 ================================
prereg loss 12.639759 reg_l1 72.371735 reg_l2 30.719566
loss 12.712131
STEP 54 ================================
prereg loss 12.053472 reg_l1 72.43961 reg_l2 30.77831
loss 12.125911
STEP 55 ================================
prereg loss 11.480338 reg_l1 72.50761 reg_l2 30.837412
loss 11.552846
STEP 56 ================================
prereg loss 10.922264 reg_l1 72.57519 reg_l2 30.896517
loss 10.99484
STEP 57 ================================
prereg loss 10.380303 reg_l1 72.64184 reg_l2 30.95521
loss 10.452945
STEP 58 ================================
prereg loss 9.855762 reg_l1 72.70721 reg_l2 31.013226
loss 9.928469
STEP 59 ================================
prereg loss 9.346587 reg_l1 72.7709 reg_l2 31.070261
loss 9.419358
STEP 60 ================================
prereg loss 8.853369 reg_l1 72.83261 reg_l2 31.126099
loss 8.926202
STEP 61 ================================
prereg loss 8.373449 reg_l1 72.89247 reg_l2 31.180641
loss 8.4463415
cutoff 0.090496995 network size 258
STEP 62 ================================
prereg loss 7.7988424 reg_l1 72.85977 reg_l2 31.225657
loss 7.871702
STEP 63 ================================
prereg loss 7.356555 reg_l1 72.91863 reg_l2 31.278976
loss 7.4294734
STEP 64 ================================
prereg loss 6.9322014 reg_l1 72.97729 reg_l2 31.332018
loss 7.0051785
STEP 65 ================================
prereg loss 6.530748 reg_l1 73.03535 reg_l2 31.384542
loss 6.603783
STEP 66 ================================
prereg loss 6.1547694 reg_l1 73.09259 reg_l2 31.436317
loss 6.227862
STEP 67 ================================
prereg loss 5.8068075 reg_l1 73.14867 reg_l2 31.487074
loss 5.8799562
STEP 68 ================================
prereg loss 5.4869847 reg_l1 73.203064 reg_l2 31.536469
loss 5.560188
STEP 69 ================================
prereg loss 5.196132 reg_l1 73.2554 reg_l2 31.584097
loss 5.2693877
STEP 70 ================================
prereg loss 4.9343944 reg_l1 73.30527 reg_l2 31.629702
loss 5.0076995
STEP 71 ================================
prereg loss 4.6996408 reg_l1 73.35226 reg_l2 31.672947
loss 4.772993
cutoff 0.082156844 network size 257
STEP 72 ================================
prereg loss 4.495488 reg_l1 73.31394 reg_l2 31.706854
loss 4.568802
STEP 73 ================================
prereg loss 4.321054 reg_l1 73.35656 reg_l2 31.745808
loss 4.3944106
STEP 74 ================================
prereg loss 4.1671395 reg_l1 73.39691 reg_l2 31.782747
loss 4.240536
STEP 75 ================================
prereg loss 4.0369244 reg_l1 73.43478 reg_l2 31.817455
loss 4.110359
STEP 76 ================================
prereg loss 3.926223 reg_l1 73.46906 reg_l2 31.849146
loss 3.9996922
STEP 77 ================================
prereg loss 3.8301413 reg_l1 73.499344 reg_l2 31.877413
loss 3.9036407
STEP 78 ================================
prereg loss 3.7403579 reg_l1 73.525986 reg_l2 31.90265
loss 3.8138838
STEP 79 ================================
prereg loss 3.6556923 reg_l1 73.54936 reg_l2 31.925116
loss 3.7292416
STEP 80 ================================
prereg loss 3.57617 reg_l1 73.56985 reg_l2 31.945127
loss 3.6497397
STEP 81 ================================
prereg loss 3.5011158 reg_l1 73.58784 reg_l2 31.963005
loss 3.5747037
cutoff 0.0818437 network size 256
STEP 82 ================================
prereg loss 3.4727614 reg_l1 73.521774 reg_l2 31.972307
loss 3.5462832
STEP 83 ================================
prereg loss 3.3770733 reg_l1 73.54353 reg_l2 31.992361
loss 3.4506168
STEP 84 ================================
prereg loss 3.2711258 reg_l1 73.56857 reg_l2 32.01483
loss 3.3446944
STEP 85 ================================
prereg loss 3.167529 reg_l1 73.595345 reg_l2 32.038467
loss 3.2411244
STEP 86 ================================
prereg loss 3.0761185 reg_l1 73.62278 reg_l2 32.06246
loss 3.1497412
STEP 87 ================================
prereg loss 3.007259 reg_l1 73.649635 reg_l2 32.08591
loss 3.0809085
STEP 88 ================================
prereg loss 2.9606242 reg_l1 73.674805 reg_l2 32.107925
loss 3.0342991
STEP 89 ================================
prereg loss 2.9321058 reg_l1 73.69735 reg_l2 32.127796
loss 3.005803
STEP 90 ================================
prereg loss 2.9139862 reg_l1 73.71641 reg_l2 32.144894
loss 2.9877026
STEP 91 ================================
prereg loss 2.897224 reg_l1 73.73153 reg_l2 32.1588
loss 2.9709554
cutoff 0.08301113 network size 255
STEP 92 ================================
prereg loss 2.7792974 reg_l1 73.65947 reg_l2 32.162468
loss 2.8529568
STEP 93 ================================
prereg loss 2.7496834 reg_l1 73.66709 reg_l2 32.17013
loss 2.8233504
STEP 94 ================================
prereg loss 2.7123616 reg_l1 73.671074 reg_l2 32.17499
loss 2.7860327
STEP 95 ================================
prereg loss 2.6689274 reg_l1 73.67207 reg_l2 32.17734
loss 2.7425995
STEP 96 ================================
prereg loss 2.622397 reg_l1 73.67057 reg_l2 32.1776
loss 2.6960676
STEP 97 ================================
prereg loss 2.5761375 reg_l1 73.667076 reg_l2 32.176254
loss 2.6498046
STEP 98 ================================
prereg loss 2.532975 reg_l1 73.66235 reg_l2 32.173794
loss 2.6066372
STEP 99 ================================
prereg loss 2.4943087 reg_l1 73.65711 reg_l2 32.170773
loss 2.5679657
STEP 100 ================================
prereg loss 2.4602308 reg_l1 73.65185 reg_l2 32.16763
loss 2.5338826
2022-07-20T09:30:40.069

julia> serialize("cf-255-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-255-parameters-opt.ser", opt)
```

Let's continue:

```
julia> interleaving_steps!(100, 10)
2022-07-20T09:36:16.203
STEP 1 ================================
prereg loss 2.4301913 reg_l1 73.64715 reg_l2 32.164734
loss 2.5038385
cutoff 0.08423886 network size 254
STEP 2 ================================
prereg loss 2.4048598 reg_l1 73.559204 reg_l2 32.15537
loss 2.478419
STEP 3 ================================
prereg loss 2.3808255 reg_l1 73.55771 reg_l2 32.15456
loss 2.4543831
STEP 4 ================================
prereg loss 2.3545434 reg_l1 73.55818 reg_l2 32.155163
loss 2.4281015
STEP 5 ================================
prereg loss 2.3261943 reg_l1 73.56041 reg_l2 32.157097
loss 2.3997548
STEP 6 ================================
prereg loss 2.2967005 reg_l1 73.56413 reg_l2 32.160206
loss 2.3702645
STEP 7 ================================
prereg loss 2.2677636 reg_l1 73.56914 reg_l2 32.16426
loss 2.3413327
STEP 8 ================================
prereg loss 2.240691 reg_l1 73.57487 reg_l2 32.168953
loss 2.3142657
STEP 9 ================================
prereg loss 2.2161338 reg_l1 73.580956 reg_l2 32.17397
loss 2.2897148
STEP 10 ================================
prereg loss 2.1943736 reg_l1 73.587166 reg_l2 32.179035
loss 2.2679608
STEP 11 ================================
prereg loss 2.174721 reg_l1 73.592964 reg_l2 32.183918
loss 2.248314
cutoff 0.093751125 network size 253
STEP 12 ================================
prereg loss 2.156627 reg_l1 73.50437 reg_l2 32.179573
loss 2.2301314
STEP 13 ================================
prereg loss 2.139083 reg_l1 73.5087 reg_l2 32.183422
loss 2.2125916
STEP 14 ================================
prereg loss 2.121158 reg_l1 73.51203 reg_l2 32.186615
loss 2.19467
STEP 15 ================================
prereg loss 2.1023073 reg_l1 73.51443 reg_l2 32.189075
loss 2.1758218
STEP 16 ================================
prereg loss 2.082336 reg_l1 73.51582 reg_l2 32.19082
loss 2.1558518
STEP 17 ================================
prereg loss 2.0617235 reg_l1 73.51642 reg_l2 32.19198
loss 2.1352398
STEP 18 ================================
prereg loss 2.0411108 reg_l1 73.516495 reg_l2 32.192707
loss 2.1146274
STEP 19 ================================
prereg loss 2.0211074 reg_l1 73.51617 reg_l2 32.19319
loss 2.0946236
STEP 20 ================================
prereg loss 2.0020616 reg_l1 73.515785 reg_l2 32.193592
loss 2.0755775
STEP 21 ================================
prereg loss 1.9842604 reg_l1 73.51557 reg_l2 32.194103
loss 2.057776
cutoff 0.09038206 network size 252
STEP 22 ================================
prereg loss 5.417408 reg_l1 73.42528 reg_l2 32.186684
loss 5.4908333
STEP 23 ================================
prereg loss 5.1581273 reg_l1 73.42415 reg_l2 32.188457
loss 5.2315516
STEP 24 ================================
prereg loss 4.753116 reg_l1 73.42196 reg_l2 32.191345
loss 4.826538
STEP 25 ================================
prereg loss 4.2996426 reg_l1 73.41977 reg_l2 32.195732
loss 4.373062
STEP 26 ================================
prereg loss 3.8862085 reg_l1 73.41864 reg_l2 32.201965
loss 3.9596272
STEP 27 ================================
prereg loss 3.573907 reg_l1 73.41949 reg_l2 32.210377
loss 3.6473265
STEP 28 ================================
prereg loss 3.387123 reg_l1 73.423164 reg_l2 32.221237
loss 3.4605463
STEP 29 ================================
prereg loss 3.314587 reg_l1 73.43033 reg_l2 32.234596
loss 3.3880174
STEP 30 ================================
prereg loss 3.318279 reg_l1 73.44138 reg_l2 32.250412
loss 3.3917203
STEP 31 ================================
prereg loss 3.3485594 reg_l1 73.45639 reg_l2 32.268433
loss 3.4220157
cutoff 0.092837036 network size 251
STEP 32 ================================
prereg loss 3.3597655 reg_l1 73.38218 reg_l2 32.279636
loss 3.4331477
STEP 33 ================================
prereg loss 3.3219438 reg_l1 73.40417 reg_l2 32.300846
loss 3.3953478
STEP 34 ================================
prereg loss 3.2158117 reg_l1 73.4286 reg_l2 32.322872
loss 3.2892404
STEP 35 ================================
prereg loss 3.0285118 reg_l1 73.45468 reg_l2 32.34516
loss 3.1019664
STEP 36 ================================
prereg loss 2.8062718 reg_l1 73.4816 reg_l2 32.36727
loss 2.8797534
STEP 37 ================================
prereg loss 2.5818863 reg_l1 73.508514 reg_l2 32.388844
loss 2.6553948
STEP 38 ================================
prereg loss 2.3854482 reg_l1 73.53482 reg_l2 32.40953
loss 2.458983
STEP 39 ================================
prereg loss 2.2191734 reg_l1 73.55986 reg_l2 32.429127
loss 2.2927332
STEP 40 ================================
prereg loss 2.0953116 reg_l1 73.58306 reg_l2 32.44748
loss 2.1688948
STEP 41 ================================
prereg loss 2.027611 reg_l1 73.60424 reg_l2 32.464607
loss 2.1012154
cutoff 0.09377394 network size 250
STEP 42 ================================
prereg loss 1.9996502 reg_l1 73.52968 reg_l2 32.471943
loss 2.07318
STEP 43 ================================
prereg loss 1.9894007 reg_l1 73.547035 reg_l2 32.487244
loss 2.0629478
STEP 44 ================================
prereg loss 1.9774835 reg_l1 73.562645 reg_l2 32.501934
loss 2.0510461
STEP 45 ================================
prereg loss 1.9514583 reg_l1 73.57673 reg_l2 32.516125
loss 2.0250351
STEP 46 ================================
prereg loss 1.9047235 reg_l1 73.58948 reg_l2 32.52999
loss 1.978313
STEP 47 ================================
prereg loss 1.8408103 reg_l1 73.601105 reg_l2 32.54356
loss 1.9144114
STEP 48 ================================
prereg loss 1.7685481 reg_l1 73.61185 reg_l2 32.556942
loss 1.84216
STEP 49 ================================
prereg loss 1.6986432 reg_l1 73.62185 reg_l2 32.57012
loss 1.7722651
STEP 50 ================================
prereg loss 1.6402923 reg_l1 73.63136 reg_l2 32.583096
loss 1.7139237
STEP 51 ================================
prereg loss 1.5989693 reg_l1 73.640495 reg_l2 32.59579
loss 1.6726098
cutoff 0.09492528 network size 249
STEP 52 ================================
prereg loss 1.5753111 reg_l1 73.55437 reg_l2 32.599186
loss 1.6488655
STEP 53 ================================
prereg loss 1.5656965 reg_l1 73.56297 reg_l2 32.61123
loss 1.6392595
STEP 54 ================================
prereg loss 1.563865 reg_l1 73.57147 reg_l2 32.62288
loss 1.6374364
STEP 55 ================================
prereg loss 1.5630299 reg_l1 73.57993 reg_l2 32.634098
loss 1.6366098
STEP 56 ================================
prereg loss 1.5578429 reg_l1 73.58832 reg_l2 32.644905
loss 1.6314312
STEP 57 ================================
prereg loss 1.5455518 reg_l1 73.59674 reg_l2 32.655285
loss 1.6191485
STEP 58 ================================
prereg loss 1.5262078 reg_l1 73.605156 reg_l2 32.665314
loss 1.599813
STEP 59 ================================
prereg loss 1.5022448 reg_l1 73.613625 reg_l2 32.675034
loss 1.5758585
STEP 60 ================================
prereg loss 1.4771172 reg_l1 73.62202 reg_l2 32.684536
loss 1.5507392
STEP 61 ================================
prereg loss 1.4542301 reg_l1 73.6305 reg_l2 32.693882
loss 1.5278605
cutoff 0.09501367 network size 248
STEP 62 ================================
prereg loss 1.4360592 reg_l1 73.54394 reg_l2 32.694088
loss 1.5096031
STEP 63 ================================
prereg loss 1.4234982 reg_l1 73.5522 reg_l2 32.703243
loss 1.4970504
STEP 64 ================================
prereg loss 1.4142709 reg_l1 73.56032 reg_l2 32.712307
loss 1.4878312
STEP 65 ================================
prereg loss 1.4079752 reg_l1 73.568184 reg_l2 32.721355
loss 1.4815434
STEP 66 ================================
prereg loss 1.4037094 reg_l1 73.57588 reg_l2 32.73042
loss 1.4772853
STEP 67 ================================
prereg loss 1.3991044 reg_l1 73.58322 reg_l2 32.739452
loss 1.4726876
STEP 68 ================================
prereg loss 1.3926808 reg_l1 73.59028 reg_l2 32.74846
loss 1.466271
STEP 69 ================================
prereg loss 1.3840598 reg_l1 73.5969 reg_l2 32.757347
loss 1.4576567
STEP 70 ================================
prereg loss 1.3737856 reg_l1 73.60303 reg_l2 32.76602
loss 1.4473886
STEP 71 ================================
prereg loss 1.3630059 reg_l1 73.6086 reg_l2 32.77443
loss 1.4366145
cutoff 0.09526269 network size 247
STEP 72 ================================
prereg loss 1.3529767 reg_l1 73.51832 reg_l2 32.773373
loss 1.426495
STEP 73 ================================
prereg loss 1.344621 reg_l1 73.522736 reg_l2 32.780937
loss 1.4181436
STEP 74 ================================
prereg loss 1.3382784 reg_l1 73.52652 reg_l2 32.788025
loss 1.4118049
STEP 75 ================================
prereg loss 1.3336782 reg_l1 73.52974 reg_l2 32.794567
loss 1.407208
STEP 76 ================================
prereg loss 1.3301227 reg_l1 73.53241 reg_l2 32.800636
loss 1.4036552
STEP 77 ================================
prereg loss 1.3267623 reg_l1 73.53472 reg_l2 32.80625
loss 1.400297
STEP 78 ================================
prereg loss 1.3228942 reg_l1 73.53678 reg_l2 32.8115
loss 1.396431
STEP 79 ================================
prereg loss 1.3181367 reg_l1 73.538635 reg_l2 32.816525
loss 1.3916754
STEP 80 ================================
prereg loss 1.3124946 reg_l1 73.5406 reg_l2 32.821453
loss 1.3860352
STEP 81 ================================
prereg loss 1.3062718 reg_l1 73.54275 reg_l2 32.826385
loss 1.3798145
cutoff 0.09583336 network size 246
STEP 82 ================================
prereg loss 1.2999104 reg_l1 73.44935 reg_l2 32.822327
loss 1.3733598
STEP 83 ================================
prereg loss 1.2938113 reg_l1 73.45216 reg_l2 32.827705
loss 1.3672634
STEP 84 ================================
prereg loss 1.2882419 reg_l1 73.455505 reg_l2 32.83344
loss 1.3616974
STEP 85 ================================
prereg loss 1.2832543 reg_l1 73.459305 reg_l2 32.839584
loss 1.3567135
STEP 86 ================================
prereg loss 1.2787011 reg_l1 73.46363 reg_l2 32.846157
loss 1.3521647
STEP 87 ================================
prereg loss 1.2743531 reg_l1 73.4684 reg_l2 32.85313
loss 1.3478216
STEP 88 ================================
prereg loss 1.2700071 reg_l1 73.47355 reg_l2 32.86046
loss 1.3434807
STEP 89 ================================
prereg loss 1.2655559 reg_l1 73.47882 reg_l2 32.868023
loss 1.3390347
STEP 90 ================================
prereg loss 1.2610168 reg_l1 73.484245 reg_l2 32.875725
loss 1.3345011
STEP 91 ================================
prereg loss 1.2565913 reg_l1 73.48953 reg_l2 32.883446
loss 1.3300809
cutoff 0.09618641 network size 245
STEP 92 ================================
prereg loss 1.2524241 reg_l1 73.398415 reg_l2 32.881786
loss 1.3258226
STEP 93 ================================
prereg loss 1.2486022 reg_l1 73.40313 reg_l2 32.889137
loss 1.3220053
STEP 94 ================================
prereg loss 1.2451887 reg_l1 73.40733 reg_l2 32.896137
loss 1.318596
STEP 95 ================================
prereg loss 1.242161 reg_l1 73.41111 reg_l2 32.90272
loss 1.3155721
STEP 96 ================================
prereg loss 1.2394099 reg_l1 73.41426 reg_l2 32.908875
loss 1.3128242
STEP 97 ================================
prereg loss 1.2367922 reg_l1 73.41697 reg_l2 32.91456
loss 1.3102092
STEP 98 ================================
prereg loss 1.2341667 reg_l1 73.41915 reg_l2 32.919827
loss 1.307586
STEP 99 ================================
prereg loss 1.2314371 reg_l1 73.42103 reg_l2 32.92474
loss 1.3048581
STEP 100 ================================
prereg loss 1.2285783 reg_l1 73.422615 reg_l2 32.929382
loss 1.302001
2022-07-20T09:49:06.515

julia> serialize("cf-245-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-245-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T09:49:24.739
STEP 1 ================================
prereg loss 1.2256179 reg_l1 73.424065 reg_l2 32.93384
loss 1.299042
cutoff 0.096816435 network size 244
STEP 2 ================================
prereg loss 1.2226266 reg_l1 73.328705 reg_l2 32.928898
loss 1.2959553
STEP 3 ================================
prereg loss 1.219674 reg_l1 73.330246 reg_l2 32.933357
loss 1.2930043
STEP 4 ================================
prereg loss 1.2168082 reg_l1 73.332085 reg_l2 32.937958
loss 1.2901403
STEP 5 ================================
prereg loss 1.2140378 reg_l1 73.3342 reg_l2 32.942787
loss 1.287372
STEP 6 ================================
prereg loss 1.2113374 reg_l1 73.33665 reg_l2 32.947884
loss 1.284674
STEP 7 ================================
prereg loss 1.2086561 reg_l1 73.339516 reg_l2 32.95328
loss 1.2819955
STEP 8 ================================
prereg loss 1.2059387 reg_l1 73.34273 reg_l2 32.95898
loss 1.2792814
STEP 9 ================================
prereg loss 1.2031455 reg_l1 73.34624 reg_l2 32.964977
loss 1.2764918
STEP 10 ================================
prereg loss 1.2002953 reg_l1 73.35001 reg_l2 32.971203
loss 1.2736454
STEP 11 ================================
prereg loss 1.1974298 reg_l1 73.35397 reg_l2 32.97759
loss 1.2707838
cutoff 0.096876994 network size 243
STEP 12 ================================
prereg loss 1.1946168 reg_l1 73.261086 reg_l2 32.974674
loss 1.2678779
STEP 13 ================================
prereg loss 1.19192 reg_l1 73.26506 reg_l2 32.981144
loss 1.2651851
STEP 14 ================================
prereg loss 1.1893886 reg_l1 73.26895 reg_l2 32.987534
loss 1.2626576
STEP 15 ================================
prereg loss 1.1870421 reg_l1 73.272575 reg_l2 32.99375
loss 1.2603147
STEP 16 ================================
prereg loss 1.1848621 reg_l1 73.27591 reg_l2 32.999767
loss 1.2581381
STEP 17 ================================
prereg loss 1.1828092 reg_l1 73.27893 reg_l2 33.00551
loss 1.2560881
STEP 18 ================================
prereg loss 1.1808221 reg_l1 73.281624 reg_l2 33.010967
loss 1.2541038
STEP 19 ================================
prereg loss 1.1788547 reg_l1 73.28398 reg_l2 33.016113
loss 1.2521387
STEP 20 ================================
prereg loss 1.1768682 reg_l1 73.28607 reg_l2 33.020992
loss 1.2501543
STEP 21 ================================
prereg loss 1.1748457 reg_l1 73.28785 reg_l2 33.025635
loss 1.2481335
cutoff 0.09713705 network size 242
STEP 22 ================================
prereg loss 1.1727984 reg_l1 73.19234 reg_l2 33.02066
loss 1.2459908
STEP 23 ================================
prereg loss 1.1707455 reg_l1 73.19382 reg_l2 33.024994
loss 1.2439393
STEP 24 ================================
prereg loss 1.1687107 reg_l1 73.19529 reg_l2 33.029285
loss 1.241906
STEP 25 ================================
prereg loss 1.1667092 reg_l1 73.19682 reg_l2 33.03358
loss 1.2399061
STEP 26 ================================
prereg loss 1.1647421 reg_l1 73.19845 reg_l2 33.037952
loss 1.2379405
STEP 27 ================================
prereg loss 1.1628014 reg_l1 73.20023 reg_l2 33.042427
loss 1.2360016
STEP 28 ================================
prereg loss 1.1608672 reg_l1 73.20217 reg_l2 33.04703
loss 1.2340693
STEP 29 ================================
prereg loss 1.158918 reg_l1 73.20429 reg_l2 33.051792
loss 1.2321223
STEP 30 ================================
prereg loss 1.1569426 reg_l1 73.20664 reg_l2 33.056736
loss 1.2301493
STEP 31 ================================
prereg loss 1.1549408 reg_l1 73.20916 reg_l2 33.06182
loss 1.22815
cutoff 0.09733032 network size 241
STEP 32 ================================
prereg loss 1.1529263 reg_l1 73.11444 reg_l2 33.057526
loss 1.2260407
STEP 33 ================================
prereg loss 1.1509199 reg_l1 73.11717 reg_l2 33.062782
loss 1.224037
STEP 34 ================================
prereg loss 1.1489476 reg_l1 73.11984 reg_l2 33.068066
loss 1.2220675
STEP 35 ================================
prereg loss 1.1470361 reg_l1 73.12252 reg_l2 33.073338
loss 1.2201586
STEP 36 ================================
prereg loss 1.1452006 reg_l1 73.12513 reg_l2 33.07855
loss 1.2183257
STEP 37 ================================
prereg loss 1.1434447 reg_l1 73.12762 reg_l2 33.083656
loss 1.2165723
STEP 38 ================================
prereg loss 1.1417603 reg_l1 73.12993 reg_l2 33.08861
loss 1.2148902
STEP 39 ================================
prereg loss 1.1401346 reg_l1 73.13205 reg_l2 33.093414
loss 1.2132666
STEP 40 ================================
prereg loss 1.1385479 reg_l1 73.13393 reg_l2 33.09805
loss 1.2116818
STEP 41 ================================
prereg loss 1.136986 reg_l1 73.13568 reg_l2 33.102512
loss 1.2101218
cutoff 0.0974079 network size 240
STEP 42 ================================
prereg loss 1.1354352 reg_l1 73.039795 reg_l2 33.097305
loss 1.208475
STEP 43 ================================
prereg loss 1.1338927 reg_l1 73.04122 reg_l2 33.10146
loss 1.2069339
STEP 44 ================================
prereg loss 1.1323574 reg_l1 73.04252 reg_l2 33.105522
loss 1.2053999
STEP 45 ================================
prereg loss 1.1308357 reg_l1 73.043724 reg_l2 33.109486
loss 1.2038794
STEP 46 ================================
prereg loss 1.1293275 reg_l1 73.044914 reg_l2 33.113415
loss 1.2023724
STEP 47 ================================
prereg loss 1.1278349 reg_l1 73.04611 reg_l2 33.117348
loss 1.200881
STEP 48 ================================
prereg loss 1.1263554 reg_l1 73.04739 reg_l2 33.121284
loss 1.1994028
STEP 49 ================================
prereg loss 1.1248829 reg_l1 73.04864 reg_l2 33.125275
loss 1.1979315
STEP 50 ================================
prereg loss 1.1234106 reg_l1 73.05006 reg_l2 33.12933
loss 1.1964606
STEP 51 ================================
prereg loss 1.1219323 reg_l1 73.05147 reg_l2 33.133446
loss 1.1949837
cutoff 0.09781937 network size 239
STEP 52 ================================
prereg loss 1.1204504 reg_l1 72.95522 reg_l2 33.128067
loss 1.1934056
STEP 53 ================================
prereg loss 1.1189657 reg_l1 72.95682 reg_l2 33.132286
loss 1.1919225
STEP 54 ================================
prereg loss 1.1175029 reg_l1 72.95847 reg_l2 33.136555
loss 1.1904614
STEP 55 ================================
prereg loss 1.1160532 reg_l1 72.96013 reg_l2 33.140827
loss 1.1890134
STEP 56 ================================
prereg loss 1.1146281 reg_l1 72.96173 reg_l2 33.145065
loss 1.1875898
STEP 57 ================================
prereg loss 1.1132355 reg_l1 72.96332 reg_l2 33.149265
loss 1.1861988
STEP 58 ================================
prereg loss 1.1118866 reg_l1 72.96478 reg_l2 33.153423
loss 1.1848514
STEP 59 ================================
prereg loss 1.1105771 reg_l1 72.96618 reg_l2 33.15745
loss 1.1835433
STEP 60 ================================
prereg loss 1.1093044 reg_l1 72.96743 reg_l2 33.161415
loss 1.1822718
STEP 61 ================================
prereg loss 1.1080625 reg_l1 72.968575 reg_l2 33.165257
loss 1.1810311
cutoff 0.09766093 network size 238
STEP 62 ================================
prereg loss 1.1043948 reg_l1 72.87187 reg_l2 33.159447
loss 1.1772667
STEP 63 ================================
prereg loss 1.100592 reg_l1 72.87336 reg_l2 33.16328
loss 1.1734654
STEP 64 ================================
prereg loss 1.0950866 reg_l1 72.87524 reg_l2 33.167175
loss 1.1679618
STEP 65 ================================
prereg loss 1.0892226 reg_l1 72.877396 reg_l2 33.171143
loss 1.1621
STEP 66 ================================
prereg loss 1.0841793 reg_l1 72.879875 reg_l2 33.175262
loss 1.1570592
STEP 67 ================================
prereg loss 1.0806276 reg_l1 72.88257 reg_l2 33.179497
loss 1.1535101
STEP 68 ================================
prereg loss 1.078627 reg_l1 72.885445 reg_l2 33.18384
loss 1.1515124
STEP 69 ================================
prereg loss 1.0777448 reg_l1 72.88834 reg_l2 33.1883
loss 1.1506332
STEP 70 ================================
prereg loss 1.077325 reg_l1 72.891266 reg_l2 33.192806
loss 1.1502162
STEP 71 ================================
prereg loss 1.0767496 reg_l1 72.89405 reg_l2 33.197277
loss 1.1496437
cutoff 0.09679763 network size 237
STEP 72 ================================
prereg loss 1.0934956 reg_l1 72.79978 reg_l2 33.19229
loss 1.1662954
STEP 73 ================================
prereg loss 1.0920745 reg_l1 72.80176 reg_l2 33.19613
loss 1.1648762
STEP 74 ================================
prereg loss 1.0905062 reg_l1 72.803 reg_l2 33.19935
loss 1.1633092
STEP 75 ================================
prereg loss 1.088725 reg_l1 72.80353 reg_l2 33.20202
loss 1.1615285
STEP 76 ================================
prereg loss 1.0867574 reg_l1 72.80344 reg_l2 33.20417
loss 1.1595609
STEP 77 ================================
prereg loss 1.0846851 reg_l1 72.80278 reg_l2 33.205875
loss 1.1574879
STEP 78 ================================
prereg loss 1.0826133 reg_l1 72.801674 reg_l2 33.207237
loss 1.155415
STEP 79 ================================
prereg loss 1.0806415 reg_l1 72.800316 reg_l2 33.208363
loss 1.1534418
STEP 80 ================================
prereg loss 1.0788249 reg_l1 72.79883 reg_l2 33.20936
loss 1.1516237
STEP 81 ================================
prereg loss 1.0771681 reg_l1 72.79727 reg_l2 33.210323
loss 1.1499654
cutoff 0.0981587 network size 236
STEP 82 ================================
prereg loss 54.430218 reg_l1 72.69767 reg_l2 33.2017
loss 54.502914
STEP 83 ================================
prereg loss 50.609787 reg_l1 72.73875 reg_l2 33.246326
loss 50.682526
STEP 84 ================================
prereg loss 43.980263 reg_l1 72.8141 reg_l2 33.326538
loss 44.053078
STEP 85 ================================
prereg loss 35.38051 reg_l1 72.918686 reg_l2 33.437393
loss 35.453426
STEP 86 ================================
prereg loss 25.938564 reg_l1 73.04765 reg_l2 33.57412
loss 26.011612
STEP 87 ================================
prereg loss 16.982714 reg_l1 73.19533 reg_l2 33.731102
loss 17.055908
STEP 88 ================================
prereg loss 10.156697 reg_l1 73.354195 reg_l2 33.900875
loss 10.230051
STEP 89 ================================
prereg loss 5.3259454 reg_l1 73.515526 reg_l2 34.074615
loss 5.399461
STEP 90 ================================
prereg loss 2.9075654 reg_l1 73.67327 reg_l2 34.245846
loss 2.9812386
STEP 91 ================================
prereg loss 2.8473032 reg_l1 73.81878 reg_l2 34.405422
loss 2.9211218
cutoff 0.0857298 network size 235
STEP 92 ================================
prereg loss 4.66098 reg_l1 73.85893 reg_l2 34.537914
loss 4.734839
STEP 93 ================================
prereg loss 7.3944535 reg_l1 73.96029 reg_l2 34.650986
loss 7.468414
STEP 94 ================================
prereg loss 9.76134 reg_l1 74.0298 reg_l2 34.731197
loss 9.83537
STEP 95 ================================
prereg loss 11.383141 reg_l1 74.07252 reg_l2 34.782543
loss 11.457213
STEP 96 ================================
prereg loss 12.078404 reg_l1 74.08868 reg_l2 34.80511
loss 12.1524935
STEP 97 ================================
prereg loss 11.770033 reg_l1 74.080086 reg_l2 34.80078
loss 11.844113
STEP 98 ================================
prereg loss 10.61954 reg_l1 74.04996 reg_l2 34.773212
loss 10.69359
STEP 99 ================================
prereg loss 8.943303 reg_l1 74.00255 reg_l2 34.72699
loss 9.017305
STEP 100 ================================
prereg loss 7.0964527 reg_l1 73.94241 reg_l2 34.66711
loss 7.170395
2022-07-20T10:02:24.958
```

When something like this happens, the choice is to go to the previous checkpoint
(which is a nice model with 245 parameters), or to try to continue.

Let's continue:

```
julia> interleaving_steps!(100, 10)
2022-07-20T10:05:19.623
STEP 1 ================================
prereg loss 5.370444 reg_l1 73.87379 reg_l2 34.59831
loss 5.444318
cutoff 0.080421075 network size 234
STEP 2 ================================
prereg loss 3.3297305 reg_l1 73.71966 reg_l2 34.517776
loss 3.4034503
STEP 3 ================================
prereg loss 2.3268921 reg_l1 73.64367 reg_l2 34.440933
loss 2.4005358
STEP 4 ================================
prereg loss 1.8229178 reg_l1 73.56871 reg_l2 34.365524
loss 1.8964865
STEP 5 ================================
prereg loss 1.7432761 reg_l1 73.49808 reg_l2 34.295013
loss 1.8167742
STEP 6 ================================
prereg loss 1.9596932 reg_l1 73.43393 reg_l2 34.231487
loss 2.033127
STEP 7 ================================
prereg loss 2.3516161 reg_l1 73.378265 reg_l2 34.176796
loss 2.4249945
STEP 8 ================================
prereg loss 2.7882972 reg_l1 73.33237 reg_l2 34.13227
loss 2.8616295
STEP 9 ================================
prereg loss 3.1644974 reg_l1 73.29678 reg_l2 34.098286
loss 3.2377942
STEP 10 ================================
prereg loss 3.412127 reg_l1 73.271385 reg_l2 34.07477
loss 3.4853983
STEP 11 ================================
prereg loss 3.5027661 reg_l1 73.25579 reg_l2 34.061268
loss 3.576022
cutoff 0.09754463 network size 233
STEP 12 ================================
prereg loss 3.4286637 reg_l1 73.153 reg_l2 34.048634
loss 3.5018167
STEP 13 ================================
prereg loss 3.2111375 reg_l1 73.15765 reg_l2 34.05482
loss 3.284295
STEP 14 ================================
prereg loss 2.9109101 reg_l1 73.16903 reg_l2 34.06781
loss 2.9840791
STEP 15 ================================
prereg loss 2.571911 reg_l1 73.185974 reg_l2 34.086273
loss 2.645097
STEP 16 ================================
prereg loss 2.2404413 reg_l1 73.2072 reg_l2 34.10884
loss 2.3136485
STEP 17 ================================
prereg loss 1.956221 reg_l1 73.2312 reg_l2 34.134045
loss 2.029452
STEP 18 ================================
prereg loss 1.7449447 reg_l1 73.25664 reg_l2 34.160316
loss 1.8182013
STEP 19 ================================
prereg loss 1.6140234 reg_l1 73.28215 reg_l2 34.18635
loss 1.6873056
STEP 20 ================================
prereg loss 1.5566493 reg_l1 73.30683 reg_l2 34.211163
loss 1.6299561
STEP 21 ================================
prereg loss 1.5627122 reg_l1 73.32962 reg_l2 34.23367
loss 1.6360419
cutoff 0.09910997 network size 232
STEP 22 ================================
prereg loss 1.6111221 reg_l1 73.25048 reg_l2 34.243126
loss 1.6843727
STEP 23 ================================
prereg loss 1.6779633 reg_l1 73.26698 reg_l2 34.25843
loss 1.7512302
STEP 24 ================================
prereg loss 1.7407894 reg_l1 73.279335 reg_l2 34.26928
loss 1.8140688
STEP 25 ================================
prereg loss 1.7820761 reg_l1 73.28729 reg_l2 34.275364
loss 1.8553634
STEP 26 ================================
prereg loss 1.7920687 reg_l1 73.29077 reg_l2 34.276703
loss 1.8653595
STEP 27 ================================
prereg loss 1.7688969 reg_l1 73.289856 reg_l2 34.2735
loss 1.8421868
STEP 28 ================================
prereg loss 1.7173946 reg_l1 73.28496 reg_l2 34.26614
loss 1.7906796
STEP 29 ================================
prereg loss 1.6468986 reg_l1 73.27653 reg_l2 34.255245
loss 1.7201751
STEP 30 ================================
prereg loss 1.5686443 reg_l1 73.265236 reg_l2 34.241447
loss 1.6419095
STEP 31 ================================
prereg loss 1.4931458 reg_l1 73.2517 reg_l2 34.225555
loss 1.5663975
cutoff 0.10008942 network size 231
STEP 32 ================================
prereg loss 1.4283687 reg_l1 73.13666 reg_l2 34.198254
loss 1.5015054
STEP 33 ================================
prereg loss 1.3791151 reg_l1 73.1209 reg_l2 34.180367
loss 1.452236
STEP 34 ================================
prereg loss 1.3465571 reg_l1 73.10503 reg_l2 34.162502
loss 1.4196621
STEP 35 ================================
prereg loss 1.3289663 reg_l1 73.089584 reg_l2 34.14528
loss 1.4020559
STEP 36 ================================
prereg loss 1.3214687 reg_l1 73.07511 reg_l2 34.129204
loss 1.3945439
STEP 37 ================================
prereg loss 1.3189932 reg_l1 73.06169 reg_l2 34.114285
loss 1.3920549
STEP 38 ================================
prereg loss 1.3198402 reg_l1 73.049675 reg_l2 34.10093
loss 1.3928899
STEP 39 ================================
prereg loss 1.3201567 reg_l1 73.03937 reg_l2 34.08936
loss 1.3931961
STEP 40 ================================
prereg loss 1.3171455 reg_l1 73.03086 reg_l2 34.07971
loss 1.3901763
STEP 41 ================================
prereg loss 1.3119372 reg_l1 73.02432 reg_l2 34.072033
loss 1.3849615
cutoff 0.099786006 network size 230
STEP 42 ================================
prereg loss 1.300641 reg_l1 72.92036 reg_l2 34.0567
loss 1.3735613
STEP 43 ================================
prereg loss 1.2836396 reg_l1 72.91843 reg_l2 34.053463
loss 1.356558
STEP 44 ================================
prereg loss 1.2626414 reg_l1 72.91846 reg_l2 34.052128
loss 1.3355598
STEP 45 ================================
prereg loss 1.2398691 reg_l1 72.92023 reg_l2 34.05239
loss 1.3127893
STEP 46 ================================
prereg loss 1.2175415 reg_l1 72.923294 reg_l2 34.05388
loss 1.2904648
STEP 47 ================================
prereg loss 1.1975307 reg_l1 72.92736 reg_l2 34.056244
loss 1.2704581
STEP 48 ================================
prereg loss 1.1809605 reg_l1 72.93189 reg_l2 34.059025
loss 1.2538924
STEP 49 ================================
prereg loss 1.1682248 reg_l1 72.936584 reg_l2 34.061928
loss 1.2411613
STEP 50 ================================
prereg loss 1.1590847 reg_l1 72.94105 reg_l2 34.064564
loss 1.2320257
STEP 51 ================================
prereg loss 1.1526234 reg_l1 72.94495 reg_l2 34.06666
loss 1.2255684
cutoff 0.0996224 network size 229
STEP 52 ================================
prereg loss 1.1477373 reg_l1 72.84851 reg_l2 34.058086
loss 1.2205858
STEP 53 ================================
prereg loss 1.1434313 reg_l1 72.8509 reg_l2 34.058517
loss 1.2162822
STEP 54 ================================
prereg loss 1.1387814 reg_l1 72.85223 reg_l2 34.05794
loss 1.2116337
STEP 55 ================================
prereg loss 1.1331837 reg_l1 72.85242 reg_l2 34.05628
loss 1.2060361
STEP 56 ================================
prereg loss 1.1263372 reg_l1 72.85148 reg_l2 34.053547
loss 1.1991887
STEP 57 ================================
prereg loss 1.1183275 reg_l1 72.84945 reg_l2 34.049812
loss 1.1911769
STEP 58 ================================
prereg loss 1.109495 reg_l1 72.84648 reg_l2 34.045223
loss 1.1823416
STEP 59 ================================
prereg loss 1.1003078 reg_l1 72.84273 reg_l2 34.039925
loss 1.1731505
STEP 60 ================================
prereg loss 1.0911821 reg_l1 72.8383 reg_l2 34.03411
loss 1.1640204
STEP 61 ================================
prereg loss 1.082195 reg_l1 72.83334 reg_l2 34.027744
loss 1.1550283
cutoff 0.100612305 network size 228
STEP 62 ================================
prereg loss 1.0740589 reg_l1 72.72734 reg_l2 34.01092
loss 1.1467862
STEP 63 ================================
prereg loss 1.0669032 reg_l1 72.72176 reg_l2 34.00413
loss 1.139625
STEP 64 ================================
prereg loss 1.0606828 reg_l1 72.71623 reg_l2 33.997406
loss 1.133399
STEP 65 ================================
prereg loss 1.0552119 reg_l1 72.71089 reg_l2 33.9909
loss 1.1279228
STEP 66 ================================
prereg loss 1.0502329 reg_l1 72.705864 reg_l2 33.984764
loss 1.1229388
STEP 67 ================================
prereg loss 1.0454752 reg_l1 72.70123 reg_l2 33.979046
loss 1.1181765
STEP 68 ================================
prereg loss 1.0407145 reg_l1 72.697174 reg_l2 33.97384
loss 1.1134117
STEP 69 ================================
prereg loss 1.0358044 reg_l1 72.69353 reg_l2 33.96914
loss 1.108498
STEP 70 ================================
prereg loss 1.0306809 reg_l1 72.69042 reg_l2 33.96495
loss 1.1033714
STEP 71 ================================
prereg loss 1.0253074 reg_l1 72.68778 reg_l2 33.96122
loss 1.0979952
cutoff 0.10080544 network size 227
STEP 72 ================================
prereg loss 1.0197972 reg_l1 72.58473 reg_l2 33.947758
loss 1.092382
STEP 73 ================================
prereg loss 1.0141984 reg_l1 72.582924 reg_l2 33.944836
loss 1.0867814
STEP 74 ================================
prereg loss 1.0087801 reg_l1 72.58148 reg_l2 33.942245
loss 1.0813617
STEP 75 ================================
prereg loss 1.0035865 reg_l1 72.5803 reg_l2 33.93993
loss 1.0761669
STEP 76 ================================
prereg loss 0.99871373 reg_l1 72.57925 reg_l2 33.937725
loss 1.071293
STEP 77 ================================
prereg loss 0.9942014 reg_l1 72.57816 reg_l2 33.93556
loss 1.0667796
STEP 78 ================================
prereg loss 0.9900413 reg_l1 72.57704 reg_l2 33.933327
loss 1.0626184
STEP 79 ================================
prereg loss 0.9861765 reg_l1 72.57573 reg_l2 33.930935
loss 1.0587522
STEP 80 ================================
prereg loss 0.98247063 reg_l1 72.57421 reg_l2 33.928333
loss 1.0550449
STEP 81 ================================
prereg loss 0.97885936 reg_l1 72.57236 reg_l2 33.925453
loss 1.0514318
cutoff 0.10080587 network size 226
STEP 82 ================================
prereg loss 0.97532487 reg_l1 72.4694 reg_l2 33.91213
loss 1.0477942
STEP 83 ================================
prereg loss 0.9718286 reg_l1 72.46693 reg_l2 33.908653
loss 1.0442955
STEP 84 ================================
prereg loss 0.9683465 reg_l1 72.464134 reg_l2 33.904945
loss 1.0408106
STEP 85 ================================
prereg loss 0.9648645 reg_l1 72.461044 reg_l2 33.900944
loss 1.0373255
STEP 86 ================================
prereg loss 0.9614097 reg_l1 72.45774 reg_l2 33.896774
loss 1.0338675
STEP 87 ================================
prereg loss 0.9580087 reg_l1 72.45419 reg_l2 33.89239
loss 1.0304629
STEP 88 ================================
prereg loss 0.9546962 reg_l1 72.45058 reg_l2 33.887913
loss 1.0271468
STEP 89 ================================
prereg loss 0.9514989 reg_l1 72.44682 reg_l2 33.883366
loss 1.0239458
STEP 90 ================================
prereg loss 0.9484313 reg_l1 72.44304 reg_l2 33.87883
loss 1.0208744
STEP 91 ================================
prereg loss 0.9454978 reg_l1 72.43927 reg_l2 33.8743
loss 1.0179371
cutoff 0.10087043 network size 225
STEP 92 ================================
prereg loss 0.9426482 reg_l1 72.33468 reg_l2 33.859673
loss 1.0149828
STEP 93 ================================
prereg loss 0.9398842 reg_l1 72.33113 reg_l2 33.85535
loss 1.0122154
STEP 94 ================================
prereg loss 0.93720585 reg_l1 72.32763 reg_l2 33.851162
loss 1.0095335
STEP 95 ================================
prereg loss 0.93459815 reg_l1 72.32438 reg_l2 33.847145
loss 1.0069225
STEP 96 ================================
prereg loss 0.9320476 reg_l1 72.32119 reg_l2 33.84328
loss 1.0043688
STEP 97 ================================
prereg loss 0.92955106 reg_l1 72.31815 reg_l2 33.83955
loss 1.0018692
STEP 98 ================================
prereg loss 0.9271169 reg_l1 72.31524 reg_l2 33.835968
loss 0.9994321
STEP 99 ================================
prereg loss 0.9247461 reg_l1 72.3125 reg_l2 33.83248
loss 0.99705863
STEP 100 ================================
prereg loss 0.92245615 reg_l1 72.30972 reg_l2 33.829094
loss 0.9947659
2022-07-20T10:18:23.267

julia> serialize("cf-225-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-225-parameters-opt.ser", opt)
```

And, indeed, we get a better model here (the usual rule of the thumb here is not to backtrack)

```
julia> interleaving_steps!(100, 10)
2022-07-20T10:22:40.454
STEP 1 ================================
prereg loss 0.92029756 reg_l1 72.30701 reg_l2 33.825768
loss 0.99260455
cutoff 0.10096314 network size 224
STEP 2 ================================
prereg loss 0.9182178 reg_l1 72.2034 reg_l2 33.812267
loss 0.9904212
STEP 3 ================================
prereg loss 0.9162187 reg_l1 72.200745 reg_l2 33.808956
loss 0.9884194
STEP 4 ================================
prereg loss 0.91429377 reg_l1 72.19799 reg_l2 33.80564
loss 0.98649174
STEP 5 ================================
prereg loss 0.912437 reg_l1 72.19523 reg_l2 33.8023
loss 0.98463225
STEP 6 ================================
prereg loss 0.9106434 reg_l1 72.19247 reg_l2 33.798912
loss 0.9828359
STEP 7 ================================
prereg loss 0.908884 reg_l1 72.18957 reg_l2 33.795486
loss 0.98107356
STEP 8 ================================
prereg loss 0.90713406 reg_l1 72.18659 reg_l2 33.792007
loss 0.97932065
STEP 9 ================================
prereg loss 0.90543187 reg_l1 72.18357 reg_l2 33.788494
loss 0.9776154
STEP 10 ================================
prereg loss 0.90376925 reg_l1 72.18051 reg_l2 33.784924
loss 0.97594976
STEP 11 ================================
prereg loss 0.9021479 reg_l1 72.177414 reg_l2 33.781326
loss 0.9743253
cutoff 0.10073518 network size 223
STEP 12 ================================
prereg loss 0.9005647 reg_l1 72.07351 reg_l2 33.767536
loss 0.9726382
STEP 13 ================================
prereg loss 0.8990226 reg_l1 72.0704 reg_l2 33.76392
loss 0.971093
STEP 14 ================================
prereg loss 0.8975238 reg_l1 72.067215 reg_l2 33.760303
loss 0.969591
STEP 15 ================================
prereg loss 0.896068 reg_l1 72.06405 reg_l2 33.756657
loss 0.968132
STEP 16 ================================
prereg loss 0.89465433 reg_l1 72.06088 reg_l2 33.753067
loss 0.9667152
STEP 17 ================================
prereg loss 0.893286 reg_l1 72.05774 reg_l2 33.74949
loss 0.9653437
STEP 18 ================================
prereg loss 0.89196205 reg_l1 72.05459 reg_l2 33.745937
loss 0.9640167
STEP 19 ================================
prereg loss 0.89068145 reg_l1 72.051506 reg_l2 33.742428
loss 0.962733
STEP 20 ================================
prereg loss 0.8894431 reg_l1 72.04846 reg_l2 33.738964
loss 0.9614916
STEP 21 ================================
prereg loss 0.88824725 reg_l1 72.04542 reg_l2 33.735565
loss 0.9602927
cutoff 0.10148471 network size 222
STEP 22 ================================
prereg loss 0.88709295 reg_l1 71.94098 reg_l2 33.721897
loss 0.9590339
STEP 23 ================================
prereg loss 0.8859776 reg_l1 71.93801 reg_l2 33.718594
loss 0.95791566
STEP 24 ================================
prereg loss 0.8849028 reg_l1 71.935074 reg_l2 33.715313
loss 0.95683783
STEP 25 ================================
prereg loss 0.88386655 reg_l1 71.93218 reg_l2 33.7121
loss 0.95579875
STEP 26 ================================
prereg loss 0.88286835 reg_l1 71.92932 reg_l2 33.708897
loss 0.9547977
STEP 27 ================================
prereg loss 0.8819074 reg_l1 71.926476 reg_l2 33.705738
loss 0.9538339
STEP 28 ================================
prereg loss 0.88098294 reg_l1 71.923615 reg_l2 33.70261
loss 0.95290655
STEP 29 ================================
prereg loss 0.8801052 reg_l1 71.920746 reg_l2 33.69949
loss 0.95202595
STEP 30 ================================
prereg loss 0.87927306 reg_l1 71.917946 reg_l2 33.696407
loss 0.951191
STEP 31 ================================
prereg loss 0.87847036 reg_l1 71.91514 reg_l2 33.69335
loss 0.9503855
cutoff 0.100692675 network size 221
STEP 32 ================================
prereg loss 0.8776969 reg_l1 71.811676 reg_l2 33.6802
loss 0.94950855
STEP 33 ================================
prereg loss 0.87695086 reg_l1 71.80909 reg_l2 33.67721
loss 0.94876
STEP 34 ================================
prereg loss 0.8762287 reg_l1 71.80655 reg_l2 33.674294
loss 0.94803524
STEP 35 ================================
prereg loss 0.875531 reg_l1 71.804016 reg_l2 33.67138
loss 0.947335
STEP 36 ================================
prereg loss 0.87485754 reg_l1 71.80145 reg_l2 33.66847
loss 0.94665897
STEP 37 ================================
prereg loss 0.8742071 reg_l1 71.79892 reg_l2 33.665615
loss 0.946006
STEP 38 ================================
prereg loss 0.8735782 reg_l1 71.796455 reg_l2 33.66276
loss 0.94537467
STEP 39 ================================
prereg loss 0.87297213 reg_l1 71.79394 reg_l2 33.659946
loss 0.94476604
STEP 40 ================================
prereg loss 0.87238914 reg_l1 71.791504 reg_l2 33.657158
loss 0.94418067
STEP 41 ================================
prereg loss 0.87182754 reg_l1 71.789085 reg_l2 33.654415
loss 0.9436166
cutoff 0.10213986 network size 220
STEP 42 ================================
prereg loss 0.8712872 reg_l1 71.68449 reg_l2 33.641262
loss 0.9429717
STEP 43 ================================
prereg loss 0.8707688 reg_l1 71.68213 reg_l2 33.638573
loss 0.94245094
STEP 44 ================================
prereg loss 0.8702756 reg_l1 71.679726 reg_l2 33.635925
loss 0.9419553
STEP 45 ================================
prereg loss 0.86980623 reg_l1 71.67734 reg_l2 33.633297
loss 0.94148356
STEP 46 ================================
prereg loss 0.8693608 reg_l1 71.674965 reg_l2 33.63069
loss 0.94103575
STEP 47 ================================
prereg loss 0.86893535 reg_l1 71.67257 reg_l2 33.6281
loss 0.9406079
STEP 48 ================================
prereg loss 0.86853033 reg_l1 71.67018 reg_l2 33.62554
loss 0.9402005
STEP 49 ================================
prereg loss 0.86814517 reg_l1 71.66782 reg_l2 33.623
loss 0.939813
STEP 50 ================================
prereg loss 0.8677779 reg_l1 71.66542 reg_l2 33.620483
loss 0.9394433
STEP 51 ================================
prereg loss 0.8674286 reg_l1 71.66304 reg_l2 33.617992
loss 0.9390916
cutoff 0.10276893 network size 219
STEP 52 ================================
prereg loss 0.86709714 reg_l1 71.55787 reg_l2 33.604927
loss 0.938655
STEP 53 ================================
prereg loss 0.8667834 reg_l1 71.555466 reg_l2 33.60246
loss 0.9383389
STEP 54 ================================
prereg loss 0.8664845 reg_l1 71.553085 reg_l2 33.59999
loss 0.93803763
STEP 55 ================================
prereg loss 0.866201 reg_l1 71.55067 reg_l2 33.59755
loss 0.93775165
STEP 56 ================================
prereg loss 0.86593336 reg_l1 71.548256 reg_l2 33.595104
loss 0.93748164
STEP 57 ================================
prereg loss 0.8656778 reg_l1 71.54586 reg_l2 33.592674
loss 0.9372236
STEP 58 ================================
prereg loss 0.8654364 reg_l1 71.54347 reg_l2 33.590267
loss 0.93697983
STEP 59 ================================
prereg loss 0.8651996 reg_l1 71.54104 reg_l2 33.58786
loss 0.93674064
STEP 60 ================================
prereg loss 0.86496556 reg_l1 71.53862 reg_l2 33.58549
loss 0.9365042
STEP 61 ================================
prereg loss 0.86474335 reg_l1 71.53623 reg_l2 33.58312
loss 0.9362796
cutoff 0.103447475 network size 218
STEP 62 ================================
prereg loss 0.8645389 reg_l1 71.430374 reg_l2 33.57007
loss 0.9359693
STEP 63 ================================
prereg loss 0.8643466 reg_l1 71.42798 reg_l2 33.567726
loss 0.9357746
STEP 64 ================================
prereg loss 0.864167 reg_l1 71.425575 reg_l2 33.565395
loss 0.93559253
STEP 65 ================================
prereg loss 0.863998 reg_l1 71.42316 reg_l2 33.563095
loss 0.93542117
STEP 66 ================================
prereg loss 0.8638447 reg_l1 71.420715 reg_l2 33.560764
loss 0.9352654
STEP 67 ================================
prereg loss 0.8637037 reg_l1 71.418274 reg_l2 33.558445
loss 0.935122
STEP 68 ================================
prereg loss 0.8635767 reg_l1 71.41581 reg_l2 33.556145
loss 0.93499255
STEP 69 ================================
prereg loss 0.8634626 reg_l1 71.41337 reg_l2 33.553844
loss 0.93487597
STEP 70 ================================
prereg loss 0.86336106 reg_l1 71.41087 reg_l2 33.551544
loss 0.93477196
STEP 71 ================================
prereg loss 0.8632721 reg_l1 71.40839 reg_l2 33.54924
loss 0.93468046
cutoff 0.103945434 network size 217
STEP 72 ================================
prereg loss 0.8631959 reg_l1 71.30192 reg_l2 33.53614
loss 0.93449783
STEP 73 ================================
prereg loss 0.8631301 reg_l1 71.29941 reg_l2 33.53384
loss 0.9344295
STEP 74 ================================
prereg loss 0.8630746 reg_l1 71.29689 reg_l2 33.531567
loss 0.9343715
STEP 75 ================================
prereg loss 0.8630295 reg_l1 71.29435 reg_l2 33.529274
loss 0.93432385
STEP 76 ================================
prereg loss 0.8629927 reg_l1 71.29181 reg_l2 33.526997
loss 0.9342845
STEP 77 ================================
prereg loss 0.86296135 reg_l1 71.289276 reg_l2 33.524708
loss 0.93425065
STEP 78 ================================
prereg loss 0.86293954 reg_l1 71.28675 reg_l2 33.52247
loss 0.9342263
STEP 79 ================================
prereg loss 0.8629225 reg_l1 71.28419 reg_l2 33.52021
loss 0.93420666
STEP 80 ================================
prereg loss 0.86291087 reg_l1 71.2817 reg_l2 33.517994
loss 0.93419254
STEP 81 ================================
prereg loss 0.8629037 reg_l1 71.27919 reg_l2 33.51577
loss 0.9341829
cutoff 0.105518565 network size 216
STEP 82 ================================
prereg loss 0.8629026 reg_l1 71.17114 reg_l2 33.50243
loss 0.93407375
STEP 83 ================================
prereg loss 0.862906 reg_l1 71.168655 reg_l2 33.50026
loss 0.93407464
STEP 84 ================================
prereg loss 0.86291134 reg_l1 71.16616 reg_l2 33.4981
loss 0.9340775
STEP 85 ================================
prereg loss 0.8629221 reg_l1 71.16367 reg_l2 33.49594
loss 0.9340857
STEP 86 ================================
prereg loss 0.862939 reg_l1 71.16124 reg_l2 33.493816
loss 0.9341003
STEP 87 ================================
prereg loss 0.86295825 reg_l1 71.1588 reg_l2 33.491703
loss 0.9341171
STEP 88 ================================
prereg loss 0.8629848 reg_l1 71.15631 reg_l2 33.4896
loss 0.9341411
STEP 89 ================================
prereg loss 0.8630214 reg_l1 71.15386 reg_l2 33.487503
loss 0.93417525
STEP 90 ================================
prereg loss 0.86306256 reg_l1 71.151436 reg_l2 33.485424
loss 0.934214
STEP 91 ================================
prereg loss 0.86310977 reg_l1 71.14894 reg_l2 33.483353
loss 0.9342587
cutoff 0.105830714 network size 215
STEP 92 ================================
prereg loss 0.86316186 reg_l1 71.040726 reg_l2 33.470097
loss 0.9342026
STEP 93 ================================
prereg loss 0.8632188 reg_l1 71.03826 reg_l2 33.46806
loss 0.93425703
STEP 94 ================================
prereg loss 0.86328024 reg_l1 71.03582 reg_l2 33.46601
loss 0.93431604
STEP 95 ================================
prereg loss 0.8633485 reg_l1 71.03342 reg_l2 33.463978
loss 0.9343819
STEP 96 ================================
prereg loss 0.8634184 reg_l1 71.03098 reg_l2 33.46198
loss 0.9344494
STEP 97 ================================
prereg loss 0.8634956 reg_l1 71.02854 reg_l2 33.45996
loss 0.9345241
STEP 98 ================================
prereg loss 0.8635749 reg_l1 71.02609 reg_l2 33.45795
loss 0.934601
STEP 99 ================================
prereg loss 0.86365914 reg_l1 71.02364 reg_l2 33.45598
loss 0.9346828
STEP 100 ================================
prereg loss 0.8637451 reg_l1 71.02122 reg_l2 33.453983
loss 0.9347663
2022-07-20T10:35:17.351

julia> serialize("cf-215-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-215-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T10:35:36.897
STEP 1 ================================
prereg loss 0.8638352 reg_l1 71.01878 reg_l2 33.452003
loss 0.93485403
cutoff 0.10607506 network size 214
STEP 2 ================================
prereg loss 0.8639281 reg_l1 70.9103 reg_l2 33.43878
loss 0.9348384
STEP 3 ================================
prereg loss 0.8640217 reg_l1 70.90787 reg_l2 33.436832
loss 0.9349296
STEP 4 ================================
prereg loss 0.86412024 reg_l1 70.905464 reg_l2 33.434887
loss 0.9350257
STEP 5 ================================
prereg loss 0.86421907 reg_l1 70.90307 reg_l2 33.43295
loss 0.93512213
STEP 6 ================================
prereg loss 0.8643216 reg_l1 70.90062 reg_l2 33.43102
loss 0.9352222
STEP 7 ================================
prereg loss 0.8644252 reg_l1 70.89822 reg_l2 33.4291
loss 0.9353234
STEP 8 ================================
prereg loss 0.8645319 reg_l1 70.89581 reg_l2 33.427197
loss 0.93542767
STEP 9 ================================
prereg loss 0.8646391 reg_l1 70.89341 reg_l2 33.425297
loss 0.9355325
STEP 10 ================================
prereg loss 0.86474794 reg_l1 70.89103 reg_l2 33.423397
loss 0.93563896
STEP 11 ================================
prereg loss 0.8648599 reg_l1 70.888626 reg_l2 33.42153
loss 0.9357485
cutoff 0.10629425 network size 213
STEP 12 ================================
prereg loss 0.8649725 reg_l1 70.779945 reg_l2 33.408375
loss 0.9357524
STEP 13 ================================
prereg loss 0.8650886 reg_l1 70.777565 reg_l2 33.406525
loss 0.9358661
STEP 14 ================================
prereg loss 0.86520725 reg_l1 70.7752 reg_l2 33.404675
loss 0.93598247
STEP 15 ================================
prereg loss 0.86532766 reg_l1 70.77281 reg_l2 33.402836
loss 0.9361005
STEP 16 ================================
prereg loss 0.86545074 reg_l1 70.77046 reg_l2 33.401028
loss 0.9362212
STEP 17 ================================
prereg loss 0.86557597 reg_l1 70.76807 reg_l2 33.39918
loss 0.936344
STEP 18 ================================
prereg loss 0.8657022 reg_l1 70.765686 reg_l2 33.39737
loss 0.9364679
STEP 19 ================================
prereg loss 0.8658316 reg_l1 70.76331 reg_l2 33.39557
loss 0.93659496
STEP 20 ================================
prereg loss 0.8659628 reg_l1 70.760925 reg_l2 33.393772
loss 0.9367237
STEP 21 ================================
prereg loss 0.86609674 reg_l1 70.758545 reg_l2 33.39198
loss 0.9368553
cutoff 0.107029036 network size 212
STEP 22 ================================
prereg loss 0.86623096 reg_l1 70.64917 reg_l2 33.37874
loss 0.9368801
STEP 23 ================================
prereg loss 0.8663709 reg_l1 70.646774 reg_l2 33.376953
loss 0.9370177
STEP 24 ================================
prereg loss 0.8665148 reg_l1 70.64442 reg_l2 33.37519
loss 0.93715924
STEP 25 ================================
prereg loss 0.8666574 reg_l1 70.64205 reg_l2 33.37342
loss 0.93729943
STEP 26 ================================
prereg loss 0.8668025 reg_l1 70.639694 reg_l2 33.37166
loss 0.9374422
STEP 27 ================================
prereg loss 0.8669474 reg_l1 70.63734 reg_l2 33.36992
loss 0.93758476
STEP 28 ================================
prereg loss 0.86709094 reg_l1 70.63502 reg_l2 33.368168
loss 0.93772596
STEP 29 ================================
prereg loss 0.8672345 reg_l1 70.63266 reg_l2 33.36648
loss 0.93786716
STEP 30 ================================
prereg loss 0.8673795 reg_l1 70.63034 reg_l2 33.36477
loss 0.93800986
STEP 31 ================================
prereg loss 0.8675244 reg_l1 70.62805 reg_l2 33.363083
loss 0.93815243
cutoff 0.10932311 network size 211
STEP 32 ================================
prereg loss 0.8656321 reg_l1 70.5164 reg_l2 33.34948
loss 0.9361485
STEP 33 ================================
prereg loss 0.86580366 reg_l1 70.51407 reg_l2 33.3478
loss 0.93631774
STEP 34 ================================
prereg loss 0.8659901 reg_l1 70.51169 reg_l2 33.346153
loss 0.9365018
STEP 35 ================================
prereg loss 0.86619055 reg_l1 70.50929 reg_l2 33.34449
loss 0.93669987
STEP 36 ================================
prereg loss 0.8664033 reg_l1 70.506874 reg_l2 33.342846
loss 0.93691015
STEP 37 ================================
prereg loss 0.8666298 reg_l1 70.50443 reg_l2 33.341198
loss 0.9371342
STEP 38 ================================
prereg loss 0.86690164 reg_l1 70.50208 reg_l2 33.339607
loss 0.93740374
STEP 39 ================================
prereg loss 0.86716825 reg_l1 70.49983 reg_l2 33.33809
loss 0.9376681
STEP 40 ================================
prereg loss 0.86741877 reg_l1 70.497665 reg_l2 33.33662
loss 0.93791646
STEP 41 ================================
prereg loss 0.8676542 reg_l1 70.49553 reg_l2 33.33519
loss 0.93814975
cutoff 0.10931727 network size 210
STEP 42 ================================
prereg loss 0.8678718 reg_l1 70.38418 reg_l2 33.321873
loss 0.938256
STEP 43 ================================
prereg loss 0.8680717 reg_l1 70.382195 reg_l2 33.32053
loss 0.93845385
STEP 44 ================================
prereg loss 0.86825407 reg_l1 70.38029 reg_l2 33.319244
loss 0.93863434
STEP 45 ================================
prereg loss 0.86842424 reg_l1 70.378365 reg_l2 33.31798
loss 0.9388026
STEP 46 ================================
prereg loss 0.8685822 reg_l1 70.37655 reg_l2 33.316753
loss 0.93895876
STEP 47 ================================
prereg loss 0.86873347 reg_l1 70.3747 reg_l2 33.315548
loss 0.9391082
STEP 48 ================================
prereg loss 0.8688819 reg_l1 70.37291 reg_l2 33.31434
loss 0.93925476
STEP 49 ================================
prereg loss 0.86902875 reg_l1 70.37114 reg_l2 33.313156
loss 0.9393999
STEP 50 ================================
prereg loss 0.8691788 reg_l1 70.36935 reg_l2 33.311974
loss 0.93954813
STEP 51 ================================
prereg loss 0.869363 reg_l1 70.36759 reg_l2 33.310814
loss 0.9397306
cutoff 0.10934707 network size 209
STEP 52 ================================
prereg loss 0.86955273 reg_l1 70.25644 reg_l2 33.29766
loss 0.9398092
STEP 53 ================================
prereg loss 0.8697448 reg_l1 70.25449 reg_l2 33.29641
loss 0.9399993
STEP 54 ================================
prereg loss 0.86993873 reg_l1 70.252464 reg_l2 33.29512
loss 0.9401912
STEP 55 ================================
prereg loss 0.8701366 reg_l1 70.25042 reg_l2 33.2938
loss 0.940387
STEP 56 ================================
prereg loss 0.87033653 reg_l1 70.24828 reg_l2 33.292442
loss 0.94058484
STEP 57 ================================
prereg loss 0.8705377 reg_l1 70.24606 reg_l2 33.291065
loss 0.94078374
STEP 58 ================================
prereg loss 0.8707395 reg_l1 70.24385 reg_l2 33.28965
loss 0.94098336
STEP 59 ================================
prereg loss 0.87093985 reg_l1 70.24155 reg_l2 33.288223
loss 0.9411814
STEP 60 ================================
prereg loss 0.8711391 reg_l1 70.23921 reg_l2 33.286762
loss 0.94137836
STEP 61 ================================
prereg loss 0.8713568 reg_l1 70.236885 reg_l2 33.285305
loss 0.94159365
cutoff 0.109790005 network size 208
STEP 62 ================================
prereg loss 0.8715717 reg_l1 70.12476 reg_l2 33.271843
loss 0.94169647
STEP 63 ================================
prereg loss 0.8717793 reg_l1 70.12256 reg_l2 33.270443
loss 0.94190186
STEP 64 ================================
prereg loss 0.87197757 reg_l1 70.12038 reg_l2 33.269115
loss 0.94209796
STEP 65 ================================
prereg loss 0.8721664 reg_l1 70.11825 reg_l2 33.267807
loss 0.94228464
STEP 66 ================================
prereg loss 0.8723454 reg_l1 70.11614 reg_l2 33.266533
loss 0.94246155
STEP 67 ================================
prereg loss 0.8725194 reg_l1 70.11415 reg_l2 33.26528
loss 0.9426335
STEP 68 ================================
prereg loss 0.8726868 reg_l1 70.11217 reg_l2 33.26406
loss 0.942799
STEP 69 ================================
prereg loss 0.8728742 reg_l1 70.11021 reg_l2 33.262867
loss 0.9429844
STEP 70 ================================
prereg loss 0.873059 reg_l1 70.108 reg_l2 33.261562
loss 0.943167
STEP 71 ================================
prereg loss 0.8732272 reg_l1 70.105576 reg_l2 33.26014
loss 0.9433328
cutoff 0.109940805 network size 207
STEP 72 ================================
prereg loss 0.87338936 reg_l1 69.99299 reg_l2 33.246502
loss 0.9433824
STEP 73 ================================
prereg loss 0.8735561 reg_l1 69.99038 reg_l2 33.24499
loss 0.9435465
STEP 74 ================================
prereg loss 0.8737177 reg_l1 69.98778 reg_l2 33.243465
loss 0.9437055
STEP 75 ================================
prereg loss 0.87387633 reg_l1 69.985146 reg_l2 33.241962
loss 0.9438615
STEP 76 ================================
prereg loss 0.8740407 reg_l1 69.98257 reg_l2 33.24044
loss 0.9440233
STEP 77 ================================
prereg loss 0.8742105 reg_l1 69.98004 reg_l2 33.23901
loss 0.9441905
STEP 78 ================================
prereg loss 0.87437624 reg_l1 69.97759 reg_l2 33.237595
loss 0.9443538
STEP 79 ================================
prereg loss 0.87453806 reg_l1 69.97527 reg_l2 33.236248
loss 0.9445133
STEP 80 ================================
prereg loss 0.8746962 reg_l1 69.97299 reg_l2 33.23493
loss 0.9446692
STEP 81 ================================
prereg loss 0.8748518 reg_l1 69.97076 reg_l2 33.233677
loss 0.94482255
cutoff 0.110358864 network size 206
STEP 82 ================================
prereg loss 0.8750231 reg_l1 69.85823 reg_l2 33.220245
loss 0.9448814
STEP 83 ================================
prereg loss 0.8751892 reg_l1 69.85607 reg_l2 33.21899
loss 0.94504523
STEP 84 ================================
prereg loss 0.87535113 reg_l1 69.85381 reg_l2 33.2177
loss 0.945205
STEP 85 ================================
prereg loss 0.8755083 reg_l1 69.851524 reg_l2 33.216415
loss 0.9453598
STEP 86 ================================
prereg loss 0.87565905 reg_l1 69.84921 reg_l2 33.215084
loss 0.94550824
STEP 87 ================================
prereg loss 0.8758078 reg_l1 69.846886 reg_l2 33.21376
loss 0.9456547
STEP 88 ================================
prereg loss 0.87596244 reg_l1 69.8445 reg_l2 33.212406
loss 0.9458069
STEP 89 ================================
prereg loss 0.8761169 reg_l1 69.84198 reg_l2 33.211006
loss 0.94595885
STEP 90 ================================
prereg loss 0.87626964 reg_l1 69.83939 reg_l2 33.209564
loss 0.94610906
STEP 91 ================================
prereg loss 0.87641853 reg_l1 69.83691 reg_l2 33.208183
loss 0.94625545
cutoff 0.113082394 network size 205
STEP 92 ================================
prereg loss 0.8765594 reg_l1 69.721405 reg_l2 33.194054
loss 0.9462808
STEP 93 ================================
prereg loss 0.8767066 reg_l1 69.719086 reg_l2 33.19277
loss 0.9464257
STEP 94 ================================
prereg loss 0.87685007 reg_l1 69.71671 reg_l2 33.19147
loss 0.94656676
STEP 95 ================================
prereg loss 0.8769875 reg_l1 69.71433 reg_l2 33.190155
loss 0.9467019
STEP 96 ================================
prereg loss 0.87712115 reg_l1 69.711945 reg_l2 33.188854
loss 0.9468331
STEP 97 ================================
prereg loss 0.8772472 reg_l1 69.7095 reg_l2 33.187515
loss 0.94695675
STEP 98 ================================
prereg loss 0.8773811 reg_l1 69.70707 reg_l2 33.186184
loss 0.9470882
STEP 99 ================================
prereg loss 0.87751347 reg_l1 69.70472 reg_l2 33.18488
loss 0.9472182
STEP 100 ================================
prereg loss 0.8776415 reg_l1 69.70243 reg_l2 33.18365
loss 0.94734395
2022-07-20T10:47:51.371

julia> serialize("cf-205-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-205-parameters-opt.ser", opt)
```

Let's continue:

```
julia> interleaving_steps!(100, 10)
2022-07-20T10:50:52.986
STEP 1 ================================
prereg loss 0.8777659 reg_l1 69.70022 reg_l2 33.182434
loss 0.94746614
cutoff 0.114840396 network size 204
STEP 2 ================================
prereg loss 0.87938994 reg_l1 69.58322 reg_l2 33.168083
loss 0.9489732
STEP 3 ================================
prereg loss 0.87952584 reg_l1 69.58075 reg_l2 33.166824
loss 0.9491066
STEP 4 ================================
prereg loss 0.8796596 reg_l1 69.57831 reg_l2 33.165577
loss 0.9492379
STEP 5 ================================
prereg loss 0.87979233 reg_l1 69.57589 reg_l2 33.164345
loss 0.94936824
STEP 6 ================================
prereg loss 0.8799248 reg_l1 69.573524 reg_l2 33.16319
loss 0.9494983
STEP 7 ================================
prereg loss 0.8800546 reg_l1 69.57127 reg_l2 33.162098
loss 0.94962585
STEP 8 ================================
prereg loss 0.88018477 reg_l1 69.56885 reg_l2 33.160923
loss 0.94975364
STEP 9 ================================
prereg loss 0.8803158 reg_l1 69.56637 reg_l2 33.15974
loss 0.94988215
STEP 10 ================================
prereg loss 0.8804395 reg_l1 69.564 reg_l2 33.158646
loss 0.9500035
STEP 11 ================================
prereg loss 0.88051546 reg_l1 69.56177 reg_l2 33.15764
loss 0.95007724
cutoff 0.11763161 network size 203
STEP 12 ================================
prereg loss 0.88057774 reg_l1 69.44197 reg_l2 33.142822
loss 0.9500197
STEP 13 ================================
prereg loss 0.88062763 reg_l1 69.439835 reg_l2 33.14187
loss 0.95006746
STEP 14 ================================
prereg loss 0.88066274 reg_l1 69.4378 reg_l2 33.141014
loss 0.95010054
STEP 15 ================================
prereg loss 0.88069224 reg_l1 69.43589 reg_l2 33.14022
loss 0.95012814
STEP 16 ================================
prereg loss 0.8807218 reg_l1 69.433815 reg_l2 33.13934
loss 0.9501556
STEP 17 ================================
prereg loss 0.88075 reg_l1 69.43171 reg_l2 33.138477
loss 0.9501817
STEP 18 ================================
prereg loss 0.88077897 reg_l1 69.4297 reg_l2 33.137672
loss 0.95020866
STEP 19 ================================
prereg loss 0.88081056 reg_l1 69.427826 reg_l2 33.136948
loss 0.9502384
STEP 20 ================================
prereg loss 0.88084257 reg_l1 69.42591 reg_l2 33.136196
loss 0.9502685
STEP 21 ================================
prereg loss 0.8808803 reg_l1 69.42398 reg_l2 33.135452
loss 0.95030427
cutoff 0.12017097 network size 202
STEP 22 ================================
prereg loss 2.608559 reg_l1 69.30161 reg_l2 33.12016
loss 2.6778605
STEP 23 ================================
prereg loss 2.4531982 reg_l1 69.30942 reg_l2 33.12804
loss 2.5225077
STEP 24 ================================
prereg loss 2.1910276 reg_l1 69.32541 reg_l2 33.14322
loss 2.260353
STEP 25 ================================
prereg loss 1.8836421 reg_l1 69.347755 reg_l2 33.164112
loss 1.9529898
STEP 26 ================================
prereg loss 1.5948287 reg_l1 69.37456 reg_l2 33.189037
loss 1.6642033
STEP 27 ================================
prereg loss 1.3755063 reg_l1 69.4036 reg_l2 33.216103
loss 1.4449099
STEP 28 ================================
prereg loss 1.2559332 reg_l1 69.43278 reg_l2 33.24343
loss 1.3253659
STEP 29 ================================
prereg loss 1.2394719 reg_l1 69.45996 reg_l2 33.269203
loss 1.3089318
STEP 30 ================================
prereg loss 1.3035275 reg_l1 69.48361 reg_l2 33.29185
loss 1.3730111
STEP 31 ================================
prereg loss 1.4076314 reg_l1 69.502205 reg_l2 33.310036
loss 1.4771336
cutoff 0.119341105 network size 201
STEP 32 ================================
prereg loss 84.03834 reg_l1 69.39535 reg_l2 33.308605
loss 84.107735
STEP 33 ================================
prereg loss 76.17012 reg_l1 69.30873 reg_l2 33.240223
loss 76.239426
STEP 34 ================================
prereg loss 63.717194 reg_l1 69.15882 reg_l2 33.12054
loss 63.786354
STEP 35 ================================
prereg loss 49.79106 reg_l1 68.96804 reg_l2 32.9683
loss 49.86003
STEP 36 ================================
prereg loss 36.31053 reg_l1 68.7472 reg_l2 32.792416
loss 36.37928
STEP 37 ================================
prereg loss 25.264984 reg_l1 68.51139 reg_l2 32.605503
loss 25.333496
STEP 38 ================================
prereg loss 17.037882 reg_l1 68.270805 reg_l2 32.415546
loss 17.106153
STEP 39 ================================
prereg loss 11.640239 reg_l1 68.03439 reg_l2 32.22954
loss 11.708273
STEP 40 ================================
prereg loss 8.553898 reg_l1 67.809425 reg_l2 32.053303
loss 8.621707
STEP 41 ================================
prereg loss 7.307313 reg_l1 67.60062 reg_l2 31.890488
loss 7.3749137
cutoff 0.09925479 network size 200
STEP 42 ================================
prereg loss 7.6627507 reg_l1 67.31235 reg_l2 31.733881
loss 7.730063
STEP 43 ================================
prereg loss 8.534479 reg_l1 67.147 reg_l2 31.605145
loss 8.601626
STEP 44 ================================
prereg loss 9.781273 reg_l1 67.00388 reg_l2 31.494349
loss 9.848277
STEP 45 ================================
prereg loss 11.154817 reg_l1 66.882416 reg_l2 31.400911
loss 11.221699
STEP 46 ================================
prereg loss 12.492051 reg_l1 66.78167 reg_l2 31.324078
loss 12.558833
STEP 47 ================================
prereg loss 13.800513 reg_l1 66.700264 reg_l2 31.26284
loss 13.867213
STEP 48 ================================
prereg loss 14.916665 reg_l1 66.63915 reg_l2 31.217827
loss 14.983304
STEP 49 ================================
prereg loss 15.694889 reg_l1 66.59724 reg_l2 31.188276
loss 15.761486
STEP 50 ================================
prereg loss 16.103806 reg_l1 66.57277 reg_l2 31.172832
loss 16.170378
STEP 51 ================================
prereg loss 16.144815 reg_l1 66.56402 reg_l2 31.170166
loss 16.21138
cutoff 0.106963366 network size 199
STEP 52 ================================
prereg loss 16.112679 reg_l1 66.46235 reg_l2 31.167482
loss 16.17914
STEP 53 ================================
prereg loss 15.486773 reg_l1 66.48038 reg_l2 31.186918
loss 15.553253
STEP 54 ================================
prereg loss 14.598081 reg_l1 66.50954 reg_l2 31.215744
loss 14.66459
STEP 55 ================================
prereg loss 13.515445 reg_l1 66.54836 reg_l2 31.252722
loss 13.581993
STEP 56 ================================
prereg loss 12.316202 reg_l1 66.59548 reg_l2 31.296673
loss 12.382797
STEP 57 ================================
prereg loss 11.075281 reg_l1 66.649155 reg_l2 31.346043
loss 11.141931
STEP 58 ================================
prereg loss 9.84317 reg_l1 66.70815 reg_l2 31.399775
loss 9.909879
STEP 59 ================================
prereg loss 8.670749 reg_l1 66.77158 reg_l2 31.456976
loss 8.73752
STEP 60 ================================
prereg loss 7.602645 reg_l1 66.83811 reg_l2 31.51657
loss 7.669483
STEP 61 ================================
prereg loss 6.722285 reg_l1 66.90533 reg_l2 31.57655
loss 6.7891903
cutoff 0.11978733 network size 198
STEP 62 ================================
prereg loss 5.9582353 reg_l1 66.85251 reg_l2 31.621677
loss 6.025088
STEP 63 ================================
prereg loss 5.3103666 reg_l1 66.91847 reg_l2 31.680004
loss 5.377285
STEP 64 ================================
prereg loss 4.7723117 reg_l1 66.98289 reg_l2 31.736752
loss 4.8392944
STEP 65 ================================
prereg loss 4.342861 reg_l1 67.04495 reg_l2 31.791243
loss 4.409906
STEP 66 ================================
prereg loss 4.015496 reg_l1 67.10401 reg_l2 31.842985
loss 4.0825996
STEP 67 ================================
prereg loss 3.7797666 reg_l1 67.15937 reg_l2 31.891476
loss 3.846926
STEP 68 ================================
prereg loss 3.6235137 reg_l1 67.21044 reg_l2 31.93632
loss 3.6907241
STEP 69 ================================
prereg loss 3.5330105 reg_l1 67.25679 reg_l2 31.977146
loss 3.6002672
STEP 70 ================================
prereg loss 3.4932098 reg_l1 67.297966 reg_l2 32.01373
loss 3.5605078
STEP 71 ================================
prereg loss 3.4877198 reg_l1 67.3337 reg_l2 32.045826
loss 3.5550535
cutoff 0.11796977 network size 197
STEP 72 ================================
prereg loss 3.5037363 reg_l1 67.24588 reg_l2 32.059437
loss 3.5709822
STEP 73 ================================
prereg loss 3.5270298 reg_l1 67.2705 reg_l2 32.082413
loss 3.5943003
STEP 74 ================================
prereg loss 3.5469446 reg_l1 67.28945 reg_l2 32.1008
loss 3.614234
STEP 75 ================================
prereg loss 3.555851 reg_l1 67.30286 reg_l2 32.114746
loss 3.623154
STEP 76 ================================
prereg loss 3.5490096 reg_l1 67.31125 reg_l2 32.12457
loss 3.6163208
STEP 77 ================================
prereg loss 3.5245006 reg_l1 67.3148 reg_l2 32.130573
loss 3.5918155
STEP 78 ================================
prereg loss 3.482017 reg_l1 67.313934 reg_l2 32.132927
loss 3.549331
STEP 79 ================================
prereg loss 3.4237802 reg_l1 67.30904 reg_l2 32.13196
loss 3.4910893
STEP 80 ================================
prereg loss 3.352571 reg_l1 67.30064 reg_l2 32.128113
loss 3.4198716
STEP 81 ================================
prereg loss 3.2719896 reg_l1 67.2893 reg_l2 32.121784
loss 3.339279
cutoff 0.11852611 network size 196
STEP 82 ================================
prereg loss 3.1834795 reg_l1 67.15704 reg_l2 32.09934
loss 3.2506366
STEP 83 ================================
prereg loss 3.0946455 reg_l1 67.14174 reg_l2 32.08936
loss 3.1617873
STEP 84 ================================
prereg loss 3.0065157 reg_l1 67.125015 reg_l2 32.078175
loss 3.0736408
STEP 85 ================================
prereg loss 2.9216437 reg_l1 67.10746 reg_l2 32.066113
loss 2.9887512
STEP 86 ================================
prereg loss 2.8421526 reg_l1 67.08947 reg_l2 32.053566
loss 2.9092422
STEP 87 ================================
prereg loss 2.7699463 reg_l1 67.07146 reg_l2 32.040928
loss 2.8370178
STEP 88 ================================
prereg loss 2.7058945 reg_l1 67.05385 reg_l2 32.02845
loss 2.7729483
STEP 89 ================================
prereg loss 2.650022 reg_l1 67.03689 reg_l2 32.01645
loss 2.717059
STEP 90 ================================
prereg loss 2.602322 reg_l1 67.02084 reg_l2 32.005146
loss 2.669343
STEP 91 ================================
prereg loss 2.5625818 reg_l1 67.00589 reg_l2 31.994648
loss 2.6295877
cutoff 0.1182286 network size 195
STEP 92 ================================
prereg loss 2.487269 reg_l1 66.87395 reg_l2 31.97118
loss 2.554143
STEP 93 ================================
prereg loss 2.4660535 reg_l1 66.862206 reg_l2 31.96311
loss 2.5329156
STEP 94 ================================
prereg loss 2.4488287 reg_l1 66.85217 reg_l2 31.956469
loss 2.5156808
STEP 95 ================================
prereg loss 2.4339485 reg_l1 66.843735 reg_l2 31.95124
loss 2.5007923
STEP 96 ================================
prereg loss 2.4201033 reg_l1 66.836945 reg_l2 31.947416
loss 2.4869401
STEP 97 ================================
prereg loss 2.4065745 reg_l1 66.83179 reg_l2 31.944996
loss 2.4734063
STEP 98 ================================
prereg loss 2.3920512 reg_l1 66.828186 reg_l2 31.94396
loss 2.4588795
STEP 99 ================================
prereg loss 2.3756628 reg_l1 66.82603 reg_l2 31.944258
loss 2.442489
STEP 100 ================================
prereg loss 2.3571725 reg_l1 66.825294 reg_l2 31.945818
loss 2.4239979
2022-07-20T11:02:04.607

julia> serialize("cf-195-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-195-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T11:03:05.054
STEP 1 ================================
prereg loss 2.3365943 reg_l1 66.82585 reg_l2 31.948503
loss 2.4034202
cutoff 0.12025775 network size 194
STEP 2 ================================
prereg loss 2.3141582 reg_l1 66.70732 reg_l2 31.937792
loss 2.3808656
STEP 3 ================================
prereg loss 2.2902274 reg_l1 66.71011 reg_l2 31.942478
loss 2.3569374
STEP 4 ================================
prereg loss 2.2652667 reg_l1 66.71383 reg_l2 31.947983
loss 2.3319805
STEP 5 ================================
prereg loss 2.2397459 reg_l1 66.71838 reg_l2 31.954224
loss 2.3064642
STEP 6 ================================
prereg loss 2.2141008 reg_l1 66.723595 reg_l2 31.961067
loss 2.2808244
STEP 7 ================================
prereg loss 2.1887271 reg_l1 66.72944 reg_l2 31.968403
loss 2.2554567
STEP 8 ================================
prereg loss 2.1639328 reg_l1 66.735695 reg_l2 31.976116
loss 2.2306685
STEP 9 ================================
prereg loss 2.1399486 reg_l1 66.742294 reg_l2 31.984098
loss 2.2066908
STEP 10 ================================
prereg loss 2.1169224 reg_l1 66.7491 reg_l2 31.99225
loss 2.1836715
STEP 11 ================================
prereg loss 2.0949423 reg_l1 66.75605 reg_l2 32.000492
loss 2.1616983
cutoff 0.12029175 network size 193
STEP 12 ================================
prereg loss 2.0740519 reg_l1 66.642715 reg_l2 31.994251
loss 2.1406946
STEP 13 ================================
prereg loss 2.054253 reg_l1 66.64956 reg_l2 32.002403
loss 2.1209028
STEP 14 ================================
prereg loss 2.035523 reg_l1 66.65629 reg_l2 32.010384
loss 2.1021793
STEP 15 ================================
prereg loss 2.0178185 reg_l1 66.6627 reg_l2 32.018177
loss 2.0844812
STEP 16 ================================
prereg loss 2.0010638 reg_l1 66.66885 reg_l2 32.025707
loss 2.0677326
STEP 17 ================================
prereg loss 1.9851875 reg_l1 66.67464 reg_l2 32.03293
loss 2.0518622
STEP 18 ================================
prereg loss 1.9701029 reg_l1 66.67998 reg_l2 32.039825
loss 2.036783
STEP 19 ================================
prereg loss 1.955706 reg_l1 66.68495 reg_l2 32.04639
loss 2.0223908
STEP 20 ================================
prereg loss 1.9419031 reg_l1 66.68944 reg_l2 32.0526
loss 2.0085926
STEP 21 ================================
prereg loss 1.9285927 reg_l1 66.69348 reg_l2 32.058464
loss 1.9952862
cutoff 0.12229237 network size 192
STEP 22 ================================
prereg loss 1.9156873 reg_l1 66.57482 reg_l2 32.048992
loss 1.9822621
STEP 23 ================================
prereg loss 1.9031051 reg_l1 66.577934 reg_l2 32.05413
loss 1.969683
STEP 24 ================================
prereg loss 1.8907388 reg_l1 66.580635 reg_l2 32.0589
loss 1.9573195
STEP 25 ================================
prereg loss 1.8785813 reg_l1 66.58294 reg_l2 32.06334
loss 1.9451642
STEP 26 ================================
prereg loss 1.8666071 reg_l1 66.584785 reg_l2 32.067474
loss 1.9331919
STEP 27 ================================
prereg loss 1.8548516 reg_l1 66.58633 reg_l2 32.07132
loss 1.921438
STEP 28 ================================
prereg loss 1.8432467 reg_l1 66.587524 reg_l2 32.0749
loss 1.9098343
STEP 29 ================================
prereg loss 1.8317908 reg_l1 66.58838 reg_l2 32.078243
loss 1.8983792
STEP 30 ================================
prereg loss 1.8204805 reg_l1 66.58892 reg_l2 32.08134
loss 1.8870693
STEP 31 ================================
prereg loss 1.8093296 reg_l1 66.58924 reg_l2 32.084248
loss 1.8759189
cutoff 0.122825794 network size 191
STEP 32 ================================
prereg loss 1.7983488 reg_l1 66.46649 reg_l2 32.071922
loss 1.8648152
STEP 33 ================================
prereg loss 1.787518 reg_l1 66.46647 reg_l2 32.074543
loss 1.8539845
STEP 34 ================================
prereg loss 1.7768452 reg_l1 66.46626 reg_l2 32.077087
loss 1.8433114
STEP 35 ================================
prereg loss 1.7663313 reg_l1 66.46599 reg_l2 32.079556
loss 1.8327973
STEP 36 ================================
prereg loss 1.7559873 reg_l1 66.46566 reg_l2 32.081974
loss 1.8224529
STEP 37 ================================
prereg loss 1.7458113 reg_l1 66.465294 reg_l2 32.08438
loss 1.8122766
STEP 38 ================================
prereg loss 1.7358305 reg_l1 66.46488 reg_l2 32.086803
loss 1.8022954
STEP 39 ================================
prereg loss 1.7260909 reg_l1 66.46457 reg_l2 32.089275
loss 1.7925555
STEP 40 ================================
prereg loss 1.7165189 reg_l1 66.464264 reg_l2 32.091797
loss 1.7829832
STEP 41 ================================
prereg loss 1.7071133 reg_l1 66.4641 reg_l2 32.094418
loss 1.7735773
cutoff 0.12622043 network size 190
STEP 42 ================================
prereg loss 1.6978644 reg_l1 66.33776 reg_l2 32.08118
loss 1.7642021
STEP 43 ================================
prereg loss 1.6887646 reg_l1 66.33777 reg_l2 32.083996
loss 1.7551024
STEP 44 ================================
prereg loss 1.678516 reg_l1 66.33784 reg_l2 32.08686
loss 1.7448539
STEP 45 ================================
prereg loss 1.6655141 reg_l1 66.33687 reg_l2 32.08894
loss 1.731851
STEP 46 ================================
prereg loss 1.6508355 reg_l1 66.33501 reg_l2 32.09034
loss 1.7171705
STEP 47 ================================
prereg loss 1.6349962 reg_l1 66.332436 reg_l2 32.091206
loss 1.7013286
STEP 48 ================================
prereg loss 1.6185001 reg_l1 66.32933 reg_l2 32.091686
loss 1.6848295
STEP 49 ================================
prereg loss 1.6017585 reg_l1 66.32586 reg_l2 32.091908
loss 1.6680844
STEP 50 ================================
prereg loss 1.585085 reg_l1 66.32213 reg_l2 32.091984
loss 1.6514071
STEP 51 ================================
prereg loss 1.5686713 reg_l1 66.318214 reg_l2 32.092003
loss 1.6349895
cutoff 0.12635416 network size 189
STEP 52 ================================
prereg loss 1.5526154 reg_l1 66.18799 reg_l2 32.076153
loss 1.6188034
STEP 53 ================================
prereg loss 1.537017 reg_l1 66.184235 reg_l2 32.076397
loss 1.6032013
STEP 54 ================================
prereg loss 1.5217587 reg_l1 66.180695 reg_l2 32.076916
loss 1.5879394
STEP 55 ================================
prereg loss 1.5067545 reg_l1 66.17747 reg_l2 32.077736
loss 1.572932
STEP 56 ================================
prereg loss 1.4919219 reg_l1 66.1746 reg_l2 32.0789
loss 1.5580965
STEP 57 ================================
prereg loss 1.4772017 reg_l1 66.17215 reg_l2 32.08046
loss 1.5433738
STEP 58 ================================
prereg loss 1.4625216 reg_l1 66.17016 reg_l2 32.082428
loss 1.5286918
STEP 59 ================================
prereg loss 1.4478189 reg_l1 66.16867 reg_l2 32.084854
loss 1.5139875
STEP 60 ================================
prereg loss 1.4330701 reg_l1 66.1676 reg_l2 32.087704
loss 1.4992377
STEP 61 ================================
prereg loss 1.4182703 reg_l1 66.16713 reg_l2 32.090977
loss 1.4844375
cutoff 0.12893921 network size 188
STEP 62 ================================
prereg loss 1.4034297 reg_l1 66.0381 reg_l2 32.078026
loss 1.4694679
STEP 63 ================================
prereg loss 1.3885779 reg_l1 66.03852 reg_l2 32.082066
loss 1.4546164
STEP 64 ================================
prereg loss 1.3737484 reg_l1 66.03938 reg_l2 32.086468
loss 1.4397879
STEP 65 ================================
prereg loss 1.3589945 reg_l1 66.04063 reg_l2 32.09118
loss 1.4250351
STEP 66 ================================
prereg loss 1.344367 reg_l1 66.04221 reg_l2 32.096188
loss 1.4104092
STEP 67 ================================
prereg loss 1.3299125 reg_l1 66.04408 reg_l2 32.101402
loss 1.3959566
STEP 68 ================================
prereg loss 1.3156741 reg_l1 66.04613 reg_l2 32.106785
loss 1.3817202
STEP 69 ================================
prereg loss 1.3016769 reg_l1 66.048454 reg_l2 32.112354
loss 1.3677254
STEP 70 ================================
prereg loss 1.2879668 reg_l1 66.05084 reg_l2 32.118
loss 1.3540177
STEP 71 ================================
prereg loss 1.2745988 reg_l1 66.05341 reg_l2 32.123764
loss 1.3406522
cutoff 0.12947431 network size 187
STEP 72 ================================
prereg loss 1.261628 reg_l1 65.926605 reg_l2 32.112846
loss 1.3275547
STEP 73 ================================
prereg loss 1.2486528 reg_l1 65.92924 reg_l2 32.118656
loss 1.3145821
STEP 74 ================================
prereg loss 1.2358154 reg_l1 65.93167 reg_l2 32.124283
loss 1.3017471
STEP 75 ================================
prereg loss 1.2232969 reg_l1 65.93394 reg_l2 32.129704
loss 1.2892308
STEP 76 ================================
prereg loss 1.2110955 reg_l1 65.93599 reg_l2 32.134926
loss 1.2770314
STEP 77 ================================
prereg loss 1.1992073 reg_l1 65.937836 reg_l2 32.139935
loss 1.2651452
STEP 78 ================================
prereg loss 1.1876272 reg_l1 65.93945 reg_l2 32.14477
loss 1.2535666
STEP 79 ================================
prereg loss 1.1763358 reg_l1 65.94093 reg_l2 32.149403
loss 1.2422768
STEP 80 ================================
prereg loss 1.1653482 reg_l1 65.94216 reg_l2 32.15388
loss 1.2312903
STEP 81 ================================
prereg loss 1.1546608 reg_l1 65.94324 reg_l2 32.15818
loss 1.2206041
cutoff 0.1297801 network size 186
STEP 82 ================================
prereg loss 1.1442724 reg_l1 65.81439 reg_l2 32.145508
loss 1.2100868
STEP 83 ================================
prereg loss 1.1341786 reg_l1 65.81525 reg_l2 32.14959
loss 1.1999938
STEP 84 ================================
prereg loss 1.124381 reg_l1 65.81594 reg_l2 32.153545
loss 1.1901969
STEP 85 ================================
prereg loss 1.1148771 reg_l1 65.81655 reg_l2 32.1574
loss 1.1806936
STEP 86 ================================
prereg loss 1.1056658 reg_l1 65.81707 reg_l2 32.16117
loss 1.1714829
STEP 87 ================================
prereg loss 1.0967412 reg_l1 65.81759 reg_l2 32.164917
loss 1.1625588
STEP 88 ================================
prereg loss 1.0880995 reg_l1 65.81811 reg_l2 32.168644
loss 1.1539176
STEP 89 ================================
prereg loss 1.079732 reg_l1 65.81861 reg_l2 32.172333
loss 1.1455506
STEP 90 ================================
prereg loss 1.0716214 reg_l1 65.81905 reg_l2 32.17601
loss 1.1374404
STEP 91 ================================
prereg loss 1.0637605 reg_l1 65.819565 reg_l2 32.179672
loss 1.1295801
cutoff 0.13014664 network size 185
STEP 92 ================================
prereg loss 1.0561637 reg_l1 65.68997 reg_l2 32.16644
loss 1.1218536
STEP 93 ================================
prereg loss 1.0488238 reg_l1 65.690575 reg_l2 32.170166
loss 1.1145144
STEP 94 ================================
prereg loss 1.0417154 reg_l1 65.691246 reg_l2 32.17391
loss 1.1074066
STEP 95 ================================
prereg loss 1.034833 reg_l1 65.691925 reg_l2 32.177647
loss 1.1005249
STEP 96 ================================
prereg loss 1.0281636 reg_l1 65.69259 reg_l2 32.181408
loss 1.0938561
STEP 97 ================================
prereg loss 1.021706 reg_l1 65.69332 reg_l2 32.185173
loss 1.0873994
STEP 98 ================================
prereg loss 1.0154506 reg_l1 65.694115 reg_l2 32.188988
loss 1.0811447
STEP 99 ================================
prereg loss 1.0093923 reg_l1 65.69497 reg_l2 32.192837
loss 1.0750872
STEP 100 ================================
prereg loss 1.0032849 reg_l1 65.695854 reg_l2 32.196697
loss 1.0689808
2022-07-20T11:13:56.037

julia> serialize("cf-185-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-185-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T11:14:35.609
STEP 1 ================================
prereg loss 0.9972946 reg_l1 65.6966 reg_l2 32.20048
loss 1.0629913
cutoff 0.12980564 network size 184
STEP 2 ================================
prereg loss 0.99146223 reg_l1 65.56756 reg_l2 32.18732
loss 1.0570298
STEP 3 ================================
prereg loss 0.98578084 reg_l1 65.56828 reg_l2 32.19098
loss 1.0513492
STEP 4 ================================
prereg loss 0.98024756 reg_l1 65.56897 reg_l2 32.194565
loss 1.0458165
STEP 5 ================================
prereg loss 0.97486687 reg_l1 65.56951 reg_l2 32.198093
loss 1.0404364
STEP 6 ================================
prereg loss 0.9696364 reg_l1 65.57004 reg_l2 32.201546
loss 1.0352064
STEP 7 ================================
prereg loss 0.9645547 reg_l1 65.570595 reg_l2 32.205
loss 1.0301254
STEP 8 ================================
prereg loss 0.9596164 reg_l1 65.57112 reg_l2 32.208405
loss 1.0251875
STEP 9 ================================
prereg loss 0.9548204 reg_l1 65.57164 reg_l2 32.211807
loss 1.0203921
STEP 10 ================================
prereg loss 0.95016253 reg_l1 65.57207 reg_l2 32.21516
loss 1.0157346
STEP 11 ================================
prereg loss 0.945639 reg_l1 65.57247 reg_l2 32.21846
loss 1.0112115
cutoff 0.12982285 network size 183
STEP 12 ================================
prereg loss 0.9754695 reg_l1 65.44301 reg_l2 32.204865
loss 1.0409125
STEP 13 ================================
prereg loss 0.9673921 reg_l1 65.445755 reg_l2 32.21002
loss 1.0328379
STEP 14 ================================
prereg loss 0.9557011 reg_l1 65.45033 reg_l2 32.216805
loss 1.0211514
STEP 15 ================================
prereg loss 0.941529 reg_l1 65.45662 reg_l2 32.224957
loss 1.0069855
STEP 16 ================================
prereg loss 0.9260693 reg_l1 65.46423 reg_l2 32.23424
loss 0.9915336
STEP 17 ================================
prereg loss 0.91045576 reg_l1 65.472916 reg_l2 32.244404
loss 0.97592866
STEP 18 ================================
prereg loss 0.8957103 reg_l1 65.482315 reg_l2 32.255203
loss 0.9611926
STEP 19 ================================
prereg loss 0.8826355 reg_l1 65.49222 reg_l2 32.26635
loss 0.9481277
STEP 20 ================================
prereg loss 0.8716816 reg_l1 65.50231 reg_l2 32.277657
loss 0.93718386
STEP 21 ================================
prereg loss 0.8630963 reg_l1 65.51236 reg_l2 32.288868
loss 0.92860866
cutoff 0.13195522 network size 182
STEP 22 ================================
prereg loss 0.86240566 reg_l1 65.39014 reg_l2 32.282394
loss 0.92779577
STEP 23 ================================
prereg loss 0.8564379 reg_l1 65.39963 reg_l2 32.29311
loss 0.92183757
STEP 24 ================================
prereg loss 0.85253537 reg_l1 65.40876 reg_l2 32.30349
loss 0.91794413
STEP 25 ================================
prereg loss 0.85040045 reg_l1 65.41742 reg_l2 32.313354
loss 0.91581786
STEP 26 ================================
prereg loss 0.84966666 reg_l1 65.425354 reg_l2 32.32263
loss 0.915092
STEP 27 ================================
prereg loss 0.8499149 reg_l1 65.432434 reg_l2 32.33117
loss 0.91534734
STEP 28 ================================
prereg loss 0.8506906 reg_l1 65.43867 reg_l2 32.33893
loss 0.9161293
STEP 29 ================================
prereg loss 0.8516106 reg_l1 65.44392 reg_l2 32.345844
loss 0.91705453
STEP 30 ================================
prereg loss 0.85237455 reg_l1 65.448235 reg_l2 32.35193
loss 0.9178228
STEP 31 ================================
prereg loss 0.8527469 reg_l1 65.45159 reg_l2 32.35721
loss 0.91819847
cutoff 0.13392043 network size 181
STEP 32 ================================
prereg loss 0.8432328 reg_l1 65.32015 reg_l2 32.343758
loss 0.90855294
STEP 33 ================================
prereg loss 0.8414446 reg_l1 65.32281 reg_l2 32.348347
loss 0.9067674
STEP 34 ================================
prereg loss 0.8396459 reg_l1 65.32551 reg_l2 32.352962
loss 0.9049714
STEP 35 ================================
prereg loss 0.83786505 reg_l1 65.32817 reg_l2 32.357563
loss 0.90319324
STEP 36 ================================
prereg loss 0.8361198 reg_l1 65.3309 reg_l2 32.362164
loss 0.9014507
STEP 37 ================================
prereg loss 0.8344194 reg_l1 65.3336 reg_l2 32.366722
loss 0.89975303
STEP 38 ================================
prereg loss 0.83277106 reg_l1 65.336205 reg_l2 32.371193
loss 0.8981073
STEP 39 ================================
prereg loss 0.83116555 reg_l1 65.3387 reg_l2 32.375534
loss 0.8965043
STEP 40 ================================
prereg loss 0.8295966 reg_l1 65.34106 reg_l2 32.379715
loss 0.89493763
STEP 41 ================================
prereg loss 0.8280507 reg_l1 65.34319 reg_l2 32.383743
loss 0.8933939
cutoff 0.13422278 network size 180
STEP 42 ================================
prereg loss 0.92005527 reg_l1 65.21092 reg_l2 32.3696
loss 0.9852662
STEP 43 ================================
prereg loss 0.90313554 reg_l1 65.211815 reg_l2 32.372993
loss 0.9683474
STEP 44 ================================
prereg loss 0.8772528 reg_l1 65.21187 reg_l2 32.376038
loss 0.9424647
STEP 45 ================================
prereg loss 0.8496771 reg_l1 65.21112 reg_l2 32.37866
loss 0.9148882
STEP 46 ================================
prereg loss 0.82702476 reg_l1 65.20971 reg_l2 32.380913
loss 0.89223444
STEP 47 ================================
prereg loss 0.8135337 reg_l1 65.207985 reg_l2 32.382927
loss 0.87874174
STEP 48 ================================
prereg loss 0.81022096 reg_l1 65.206 reg_l2 32.384674
loss 0.87542695
STEP 49 ================================
prereg loss 0.8151976 reg_l1 65.20403 reg_l2 32.386253
loss 0.8804016
STEP 50 ================================
prereg loss 0.8246197 reg_l1 65.202194 reg_l2 32.387684
loss 0.8898219
STEP 51 ================================
prereg loss 0.83405447 reg_l1 65.2006 reg_l2 32.389034
loss 0.8992551
cutoff 0.13429075 network size 179
STEP 52 ================================
prereg loss 0.8398112 reg_l1 65.06512 reg_l2 32.372326
loss 0.90487635
STEP 53 ================================
prereg loss 0.8398341 reg_l1 65.06432 reg_l2 32.373676
loss 0.9048984
STEP 54 ================================
prereg loss 0.834007 reg_l1 65.064 reg_l2 32.375095
loss 0.89907104
STEP 55 ================================
prereg loss 0.8238374 reg_l1 65.06411 reg_l2 32.376625
loss 0.88890153
STEP 56 ================================
prereg loss 0.8117795 reg_l1 65.06468 reg_l2 32.378334
loss 0.87684417
STEP 57 ================================
prereg loss 0.80037993 reg_l1 65.06561 reg_l2 32.38024
loss 0.86544555
STEP 58 ================================
prereg loss 0.79158556 reg_l1 65.06693 reg_l2 32.38239
loss 0.8566525
STEP 59 ================================
prereg loss 0.7863457 reg_l1 65.06856 reg_l2 32.384804
loss 0.85141426
STEP 60 ================================
prereg loss 0.7845507 reg_l1 65.07047 reg_l2 32.387512
loss 0.8496212
STEP 61 ================================
prereg loss 0.785195 reg_l1 65.07258 reg_l2 32.390522
loss 0.8502676
cutoff 0.13826211 network size 178
STEP 62 ================================
prereg loss 0.7869105 reg_l1 64.936584 reg_l2 32.374714
loss 0.85184705
STEP 63 ================================
prereg loss 0.7883633 reg_l1 64.93898 reg_l2 32.378258
loss 0.85330224
STEP 64 ================================
prereg loss 0.788597 reg_l1 64.9414 reg_l2 32.382053
loss 0.8535384
STEP 65 ================================
prereg loss 0.78723735 reg_l1 64.94387 reg_l2 32.38604
loss 0.8521812
STEP 66 ================================
prereg loss 0.78447074 reg_l1 64.94637 reg_l2 32.390224
loss 0.8494171
STEP 67 ================================
prereg loss 0.78090864 reg_l1 64.94887 reg_l2 32.39453
loss 0.8458575
STEP 68 ================================
prereg loss 0.7773146 reg_l1 64.951324 reg_l2 32.398884
loss 0.84226596
STEP 69 ================================
prereg loss 0.77431947 reg_l1 64.953705 reg_l2 32.403297
loss 0.83927315
STEP 70 ================================
prereg loss 0.7723422 reg_l1 64.95604 reg_l2 32.407684
loss 0.8372983
STEP 71 ================================
prereg loss 0.7714702 reg_l1 64.958374 reg_l2 32.41206
loss 0.8364286
cutoff 0.13898265 network size 177
STEP 72 ================================
prereg loss 0.7714837 reg_l1 64.821724 reg_l2 32.39707
loss 0.83630544
STEP 73 ================================
prereg loss 0.771972 reg_l1 64.82405 reg_l2 32.401337
loss 0.83679605
STEP 74 ================================
prereg loss 0.77247065 reg_l1 64.82639 reg_l2 32.405518
loss 0.8372971
STEP 75 ================================
prereg loss 0.7725842 reg_l1 64.82869 reg_l2 32.409588
loss 0.8374129
STEP 76 ================================
prereg loss 0.7720916 reg_l1 64.83097 reg_l2 32.413532
loss 0.8369226
STEP 77 ================================
prereg loss 0.77096075 reg_l1 64.833244 reg_l2 32.417347
loss 0.835794
STEP 78 ================================
prereg loss 0.7693313 reg_l1 64.83538 reg_l2 32.421047
loss 0.83416665
STEP 79 ================================
prereg loss 0.76743394 reg_l1 64.837494 reg_l2 32.424583
loss 0.83227146
STEP 80 ================================
prereg loss 0.7655252 reg_l1 64.83945 reg_l2 32.427998
loss 0.8303647
STEP 81 ================================
prereg loss 0.76381135 reg_l1 64.84127 reg_l2 32.43128
loss 0.8286526
cutoff 0.14051232 network size 176
STEP 82 ================================
prereg loss 0.762401 reg_l1 64.70241 reg_l2 32.414684
loss 0.8271034
STEP 83 ================================
prereg loss 0.76130664 reg_l1 64.70404 reg_l2 32.417747
loss 0.8260107
STEP 84 ================================
prereg loss 0.7604662 reg_l1 64.705505 reg_l2 32.420685
loss 0.8251717
STEP 85 ================================
prereg loss 0.75977516 reg_l1 64.7068 reg_l2 32.42353
loss 0.82448196
STEP 86 ================================
prereg loss 0.7591309 reg_l1 64.70788 reg_l2 32.42627
loss 0.8238388
STEP 87 ================================
prereg loss 0.7584613 reg_l1 64.70884 reg_l2 32.42892
loss 0.8231701
STEP 88 ================================
prereg loss 0.75774103 reg_l1 64.709694 reg_l2 32.43153
loss 0.82245076
STEP 89 ================================
prereg loss 0.7569808 reg_l1 64.710434 reg_l2 32.434113
loss 0.8216912
STEP 90 ================================
prereg loss 0.75621986 reg_l1 64.711174 reg_l2 32.43668
loss 0.820931
STEP 91 ================================
prereg loss 0.75551015 reg_l1 64.71183 reg_l2 32.43922
loss 0.82022196
cutoff 0.14049894 network size 175
STEP 92 ================================
prereg loss 0.7620311 reg_l1 64.57196 reg_l2 32.42199
loss 0.82660306
STEP 93 ================================
prereg loss 0.76173013 reg_l1 64.572464 reg_l2 32.424347
loss 0.8263026
STEP 94 ================================
prereg loss 0.7611773 reg_l1 64.57283 reg_l2 32.426556
loss 0.8257501
STEP 95 ================================
prereg loss 0.76037574 reg_l1 64.5731 reg_l2 32.428654
loss 0.82494885
STEP 96 ================================
prereg loss 0.75937814 reg_l1 64.57331 reg_l2 32.430645
loss 0.8239514
STEP 97 ================================
prereg loss 0.7582459 reg_l1 64.57347 reg_l2 32.432556
loss 0.82281935
STEP 98 ================================
prereg loss 0.75709003 reg_l1 64.573586 reg_l2 32.434433
loss 0.8216636
STEP 99 ================================
prereg loss 0.7559962 reg_l1 64.57367 reg_l2 32.436306
loss 0.8205699
STEP 100 ================================
prereg loss 0.75500846 reg_l1 64.57384 reg_l2 32.438187
loss 0.8195823
2022-07-20T11:25:09.866

julia> serialize("cf-175-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-175-parameters-opt.ser", opt)
```

This is the best model so far.

```
julia> interleaving_steps!(100, 10)
2022-07-20T11:29:21.268
STEP 1 ================================
prereg loss 0.7541427 reg_l1 64.57403 reg_l2 32.44014
loss 0.81871676
cutoff 0.14176272 network size 174
STEP 2 ================================
prereg loss 0.7533821 reg_l1 64.432495 reg_l2 32.422028
loss 0.8178146
STEP 3 ================================
prereg loss 0.7526913 reg_l1 64.43279 reg_l2 32.424088
loss 0.8171241
STEP 4 ================================
prereg loss 0.7520414 reg_l1 64.433075 reg_l2 32.426174
loss 0.8164745
STEP 5 ================================
prereg loss 0.7513964 reg_l1 64.43337 reg_l2 32.4283
loss 0.8158298
STEP 6 ================================
prereg loss 0.7507409 reg_l1 64.43366 reg_l2 32.43047
loss 0.8151746
STEP 7 ================================
prereg loss 0.75007206 reg_l1 64.43398 reg_l2 32.432674
loss 0.81450605
STEP 8 ================================
prereg loss 0.7494018 reg_l1 64.434326 reg_l2 32.434937
loss 0.81383616
STEP 9 ================================
prereg loss 0.7487403 reg_l1 64.43474 reg_l2 32.43725
loss 0.8131751
STEP 10 ================================
prereg loss 0.7480993 reg_l1 64.435196 reg_l2 32.439667
loss 0.8125345
STEP 11 ================================
prereg loss 0.7474886 reg_l1 64.435715 reg_l2 32.442066
loss 0.81192434
cutoff 0.1422441 network size 173
STEP 12 ================================
prereg loss 0.7469024 reg_l1 64.29397 reg_l2 32.42428
loss 0.8111964
STEP 13 ================================
prereg loss 0.7463324 reg_l1 64.2947 reg_l2 32.42681
loss 0.8106271
STEP 14 ================================
prereg loss 0.74576575 reg_l1 64.29546 reg_l2 32.429333
loss 0.8100612
STEP 15 ================================
prereg loss 0.7451877 reg_l1 64.296196 reg_l2 32.43186
loss 0.8094839
STEP 16 ================================
prereg loss 0.7445876 reg_l1 64.297005 reg_l2 32.434387
loss 0.8088846
STEP 17 ================================
prereg loss 0.74396163 reg_l1 64.2978 reg_l2 32.436905
loss 0.8082594
STEP 18 ================================
prereg loss 0.74330765 reg_l1 64.29861 reg_l2 32.43941
loss 0.8076063
STEP 19 ================================
prereg loss 0.7426343 reg_l1 64.29941 reg_l2 32.44191
loss 0.8069337
STEP 20 ================================
prereg loss 0.7419503 reg_l1 64.30016 reg_l2 32.444366
loss 0.80625045
STEP 21 ================================
prereg loss 0.7412647 reg_l1 64.3009 reg_l2 32.446827
loss 0.8055656
cutoff 0.14414737 network size 172
STEP 22 ================================
prereg loss 0.74058765 reg_l1 64.157486 reg_l2 32.428444
loss 0.80474514
STEP 23 ================================
prereg loss 0.7399229 reg_l1 64.15822 reg_l2 32.430843
loss 0.8040811
STEP 24 ================================
prereg loss 0.739275 reg_l1 64.158905 reg_l2 32.433212
loss 0.8034339
STEP 25 ================================
prereg loss 0.7386414 reg_l1 64.15952 reg_l2 32.43551
loss 0.8028009
STEP 26 ================================
prereg loss 0.7380264 reg_l1 64.16004 reg_l2 32.437763
loss 0.8021864
STEP 27 ================================
prereg loss 0.73742163 reg_l1 64.16061 reg_l2 32.440018
loss 0.8015822
STEP 28 ================================
prereg loss 0.7368291 reg_l1 64.16114 reg_l2 32.442276
loss 0.8009902
STEP 29 ================================
prereg loss 0.7362476 reg_l1 64.161575 reg_l2 32.44447
loss 0.8004092
STEP 30 ================================
prereg loss 0.7356758 reg_l1 64.161995 reg_l2 32.446617
loss 0.7998378
STEP 31 ================================
prereg loss 0.7351136 reg_l1 64.16232 reg_l2 32.448734
loss 0.79927593
cutoff 0.1464328 network size 171
STEP 32 ================================
prereg loss 0.7345612 reg_l1 64.01625 reg_l2 32.429394
loss 0.7985774
STEP 33 ================================
prereg loss 0.73401755 reg_l1 64.01656 reg_l2 32.43146
loss 0.79803413
STEP 34 ================================
prereg loss 0.7334787 reg_l1 64.01683 reg_l2 32.433483
loss 0.79749554
STEP 35 ================================
prereg loss 0.73294204 reg_l1 64.01715 reg_l2 32.43553
loss 0.7969592
STEP 36 ================================
prereg loss 0.7324099 reg_l1 64.01745 reg_l2 32.43757
loss 0.79642737
STEP 37 ================================
prereg loss 0.7318743 reg_l1 64.01773 reg_l2 32.439587
loss 0.795892
STEP 38 ================================
prereg loss 0.7313352 reg_l1 64.018005 reg_l2 32.441563
loss 0.79535323
STEP 39 ================================
prereg loss 0.730789 reg_l1 64.01823 reg_l2 32.44355
loss 0.79480726
STEP 40 ================================
prereg loss 0.73023975 reg_l1 64.01848 reg_l2 32.445488
loss 0.79425824
STEP 41 ================================
prereg loss 0.72968984 reg_l1 64.01869 reg_l2 32.44742
loss 0.79370856
cutoff 0.14726362 network size 170
STEP 42 ================================
prereg loss 0.7291385 reg_l1 63.871677 reg_l2 32.427685
loss 0.7930102
STEP 43 ================================
prereg loss 0.72858757 reg_l1 63.871994 reg_l2 32.429657
loss 0.79245955
STEP 44 ================================
prereg loss 0.7280389 reg_l1 63.87229 reg_l2 32.431633
loss 0.7919112
STEP 45 ================================
prereg loss 0.7274979 reg_l1 63.872604 reg_l2 32.43359
loss 0.7913705
STEP 46 ================================
prereg loss 0.72696394 reg_l1 63.872875 reg_l2 32.43554
loss 0.7908368
STEP 47 ================================
prereg loss 0.7264349 reg_l1 63.873116 reg_l2 32.437458
loss 0.790308
STEP 48 ================================
prereg loss 0.7259118 reg_l1 63.87338 reg_l2 32.439377
loss 0.78978515
STEP 49 ================================
prereg loss 0.72539836 reg_l1 63.873585 reg_l2 32.44126
loss 0.78927195
STEP 50 ================================
prereg loss 0.7248901 reg_l1 63.873817 reg_l2 32.44316
loss 0.78876394
STEP 51 ================================
prereg loss 0.72438854 reg_l1 63.874096 reg_l2 32.445107
loss 0.7882626
cutoff 0.14733227 network size 169
STEP 52 ================================
prereg loss 0.72389334 reg_l1 63.727016 reg_l2 32.425297
loss 0.78762037
STEP 53 ================================
prereg loss 0.72340465 reg_l1 63.727245 reg_l2 32.42719
loss 0.7871319
STEP 54 ================================
prereg loss 0.72291946 reg_l1 63.72743 reg_l2 32.42903
loss 0.7866469
STEP 55 ================================
prereg loss 0.7224392 reg_l1 63.72767 reg_l2 32.430927
loss 0.7861669
STEP 56 ================================
prereg loss 0.72196245 reg_l1 63.727848 reg_l2 32.432785
loss 0.7856903
STEP 57 ================================
prereg loss 0.7214878 reg_l1 63.728085 reg_l2 32.43466
loss 0.7852159
STEP 58 ================================
prereg loss 0.7210158 reg_l1 63.72828 reg_l2 32.4365
loss 0.7847441
STEP 59 ================================
prereg loss 0.7205442 reg_l1 63.728462 reg_l2 32.438297
loss 0.7842727
STEP 60 ================================
prereg loss 0.72007346 reg_l1 63.728592 reg_l2 32.440105
loss 0.78380203
STEP 61 ================================
prereg loss 0.71960473 reg_l1 63.72878 reg_l2 32.441906
loss 0.78333354
cutoff 0.14808203 network size 168
STEP 62 ================================
prereg loss 0.7539655 reg_l1 63.58084 reg_l2 32.421764
loss 0.81754637
STEP 63 ================================
prereg loss 0.7499849 reg_l1 63.58142 reg_l2 32.423676
loss 0.8135663
STEP 64 ================================
prereg loss 0.7439205 reg_l1 63.582275 reg_l2 32.425674
loss 0.8075028
STEP 65 ================================
prereg loss 0.7374507 reg_l1 63.5834 reg_l2 32.42777
loss 0.8010341
STEP 66 ================================
prereg loss 0.7320618 reg_l1 63.58469 reg_l2 32.42994
loss 0.7956465
STEP 67 ================================
prereg loss 0.72863865 reg_l1 63.58605 reg_l2 32.43216
loss 0.7922247
STEP 68 ================================
prereg loss 0.7273254 reg_l1 63.5875 reg_l2 32.43447
loss 0.79091287
STEP 69 ================================
prereg loss 0.72762954 reg_l1 63.58896 reg_l2 32.436836
loss 0.7912185
STEP 70 ================================
prereg loss 0.7286914 reg_l1 63.590374 reg_l2 32.439243
loss 0.79228175
STEP 71 ================================
prereg loss 0.72961074 reg_l1 63.591686 reg_l2 32.441696
loss 0.7932024
cutoff 0.15587088 network size 167
STEP 72 ================================
prereg loss 0.73414636 reg_l1 63.437023 reg_l2 32.419857
loss 0.7975834
STEP 73 ================================
prereg loss 0.7331149 reg_l1 63.43804 reg_l2 32.42226
loss 0.79655296
STEP 74 ================================
prereg loss 0.7309835 reg_l1 63.438854 reg_l2 32.424606
loss 0.7944223
STEP 75 ================================
prereg loss 0.72824705 reg_l1 63.439476 reg_l2 32.426865
loss 0.79168653
STEP 76 ================================
prereg loss 0.7255033 reg_l1 63.439907 reg_l2 32.42904
loss 0.78894323
STEP 77 ================================
prereg loss 0.7232622 reg_l1 63.44016 reg_l2 32.431095
loss 0.78670233
STEP 78 ================================
prereg loss 0.72181195 reg_l1 63.44024 reg_l2 32.433025
loss 0.7852522
STEP 79 ================================
prereg loss 0.72116244 reg_l1 63.44025 reg_l2 32.434856
loss 0.7846027
STEP 80 ================================
prereg loss 0.72109044 reg_l1 63.44017 reg_l2 32.436565
loss 0.78453064
STEP 81 ================================
prereg loss 0.7212422 reg_l1 63.44001 reg_l2 32.438168
loss 0.7846822
cutoff 0.15584955 network size 166
STEP 82 ================================
prereg loss 0.7890021 reg_l1 63.283978 reg_l2 32.41536
loss 0.8522861
STEP 83 ================================
prereg loss 0.7852996 reg_l1 63.285595 reg_l2 32.41833
loss 0.8485852
STEP 84 ================================
prereg loss 0.7793147 reg_l1 63.28879 reg_l2 32.422573
loss 0.8426035
STEP 85 ================================
prereg loss 0.7716585 reg_l1 63.293358 reg_l2 32.427906
loss 0.8349518
STEP 86 ================================
prereg loss 0.7630567 reg_l1 63.29902 reg_l2 32.43413
loss 0.8263557
STEP 87 ================================
prereg loss 0.7542033 reg_l1 63.30553 reg_l2 32.441025
loss 0.8175089
STEP 88 ================================
prereg loss 0.74571073 reg_l1 63.31259 reg_l2 32.448402
loss 0.8090233
STEP 89 ================================
prereg loss 0.73804253 reg_l1 63.320007 reg_l2 32.456047
loss 0.8013625
STEP 90 ================================
prereg loss 0.7314843 reg_l1 63.32757 reg_l2 32.46381
loss 0.79481184
STEP 91 ================================
prereg loss 0.7261571 reg_l1 63.335052 reg_l2 32.47149
loss 0.7894922
cutoff 0.15597124 network size 165
STEP 92 ================================
prereg loss 0.7220664 reg_l1 63.186268 reg_l2 32.454636
loss 0.7852527
STEP 93 ================================
prereg loss 0.7190587 reg_l1 63.193027 reg_l2 32.461742
loss 0.7822517
STEP 94 ================================
prereg loss 0.7169558 reg_l1 63.199238 reg_l2 32.468384
loss 0.780155
STEP 95 ================================
prereg loss 0.71555644 reg_l1 63.20478 reg_l2 32.474483
loss 0.7787612
STEP 96 ================================
prereg loss 0.71466106 reg_l1 63.209625 reg_l2 32.479958
loss 0.7778707
STEP 97 ================================
prereg loss 0.71411973 reg_l1 63.21365 reg_l2 32.48477
loss 0.7773334
STEP 98 ================================
prereg loss 0.71372837 reg_l1 63.216877 reg_l2 32.488907
loss 0.77694523
STEP 99 ================================
prereg loss 0.7133243 reg_l1 63.21933 reg_l2 32.492336
loss 0.7765436
STEP 100 ================================
prereg loss 0.71277565 reg_l1 63.220997 reg_l2 32.495083
loss 0.7759966
2022-07-20T11:40:42.185

julia> serialize("cf-165-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-165-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T11:51:04.306
STEP 1 ================================
prereg loss 0.71197873 reg_l1 63.221943 reg_l2 32.49718
loss 0.77520066
cutoff 0.15671852 network size 164
STEP 2 ================================
prereg loss 0.71086067 reg_l1 63.065525 reg_l2 32.474144
loss 0.7739262
STEP 3 ================================
prereg loss 0.7093971 reg_l1 63.065254 reg_l2 32.475105
loss 0.77246237
STEP 4 ================================
prereg loss 0.70761156 reg_l1 63.064445 reg_l2 32.475636
loss 0.770676
STEP 5 ================================
prereg loss 0.70555884 reg_l1 63.06323 reg_l2 32.475746
loss 0.76862204
STEP 6 ================================
prereg loss 0.7033371 reg_l1 63.061672 reg_l2 32.47554
loss 0.7663987
STEP 7 ================================
prereg loss 0.7010656 reg_l1 63.059883 reg_l2 32.47509
loss 0.76412547
STEP 8 ================================
prereg loss 0.6988463 reg_l1 63.057915 reg_l2 32.474483
loss 0.7619042
STEP 9 ================================
prereg loss 0.6967707 reg_l1 63.05583 reg_l2 32.47377
loss 0.75982654
STEP 10 ================================
prereg loss 0.694906 reg_l1 63.053738 reg_l2 32.473038
loss 0.7579597
STEP 11 ================================
prereg loss 0.6932841 reg_l1 63.051693 reg_l2 32.472355
loss 0.7563358
cutoff 0.15757374 network size 163
STEP 12 ================================
prereg loss 0.6935392 reg_l1 62.892143 reg_l2 32.446922
loss 0.75643134
STEP 13 ================================
prereg loss 0.6932379 reg_l1 62.890953 reg_l2 32.44694
loss 0.75612885
STEP 14 ================================
prereg loss 0.69271606 reg_l1 62.89034 reg_l2 32.44752
loss 0.7556064
STEP 15 ================================
prereg loss 0.6919192 reg_l1 62.89033 reg_l2 32.44864
loss 0.75480956
STEP 16 ================================
prereg loss 0.6908391 reg_l1 62.890846 reg_l2 32.450233
loss 0.75372994
STEP 17 ================================
prereg loss 0.68950796 reg_l1 62.89184 reg_l2 32.452267
loss 0.7523998
STEP 18 ================================
prereg loss 0.68798655 reg_l1 62.893246 reg_l2 32.454666
loss 0.7508798
STEP 19 ================================
prereg loss 0.6863546 reg_l1 62.894997 reg_l2 32.457386
loss 0.7492496
STEP 20 ================================
prereg loss 0.6846884 reg_l1 62.897026 reg_l2 32.460354
loss 0.7475854
STEP 21 ================================
prereg loss 0.6830624 reg_l1 62.899273 reg_l2 32.463528
loss 0.74596167
cutoff 0.15915468 network size 162
STEP 22 ================================
prereg loss 0.6815289 reg_l1 62.7425 reg_l2 32.441463
loss 0.7442714
STEP 23 ================================
prereg loss 0.6801257 reg_l1 62.744972 reg_l2 32.44481
loss 0.7428707
STEP 24 ================================
prereg loss 0.6788695 reg_l1 62.74745 reg_l2 32.44814
loss 0.74161696
STEP 25 ================================
prereg loss 0.6777573 reg_l1 62.749905 reg_l2 32.451424
loss 0.74050725
STEP 26 ================================
prereg loss 0.6767916 reg_l1 62.75224 reg_l2 32.454613
loss 0.73954386
STEP 27 ================================
prereg loss 0.675967 reg_l1 62.75444 reg_l2 32.457653
loss 0.73872143
STEP 28 ================================
prereg loss 0.6752271 reg_l1 62.756485 reg_l2 32.46052
loss 0.7379836
STEP 29 ================================
prereg loss 0.6745432 reg_l1 62.758305 reg_l2 32.463196
loss 0.7373015
STEP 30 ================================
prereg loss 0.6738907 reg_l1 62.759872 reg_l2 32.46565
loss 0.7366506
STEP 31 ================================
prereg loss 0.6732464 reg_l1 62.7612 reg_l2 32.467854
loss 0.7360076
cutoff 0.16138926 network size 161
STEP 32 ================================
prereg loss 0.6725963 reg_l1 62.60092 reg_l2 32.443806
loss 0.7351972
STEP 33 ================================
prereg loss 0.67192715 reg_l1 62.60185 reg_l2 32.445602
loss 0.734529
STEP 34 ================================
prereg loss 0.67125106 reg_l1 62.602543 reg_l2 32.447166
loss 0.7338536
STEP 35 ================================
prereg loss 0.67055154 reg_l1 62.603004 reg_l2 32.448513
loss 0.73315454
STEP 36 ================================
prereg loss 0.6698282 reg_l1 62.603237 reg_l2 32.449677
loss 0.7324314
STEP 37 ================================
prereg loss 0.6690857 reg_l1 62.603256 reg_l2 32.45064
loss 0.7316889
STEP 38 ================================
prereg loss 0.66832954 reg_l1 62.603146 reg_l2 32.45148
loss 0.7309327
STEP 39 ================================
prereg loss 0.6675722 reg_l1 62.602856 reg_l2 32.452152
loss 0.7301751
STEP 40 ================================
prereg loss 0.66681933 reg_l1 62.602463 reg_l2 32.45273
loss 0.7294218
STEP 41 ================================
prereg loss 0.6660802 reg_l1 62.60199 reg_l2 32.453228
loss 0.72868216
cutoff 0.16234833 network size 160
STEP 42 ================================
prereg loss 0.79038155 reg_l1 62.439083 reg_l2 32.427307
loss 0.85282063
STEP 43 ================================
prereg loss 0.7732075 reg_l1 62.43934 reg_l2 32.428093
loss 0.8356468
STEP 44 ================================
prereg loss 0.74581635 reg_l1 62.440353 reg_l2 32.429222
loss 0.8082567
STEP 45 ================================
prereg loss 0.7169274 reg_l1 62.441944 reg_l2 32.430695
loss 0.77936935
STEP 46 ================================
prereg loss 0.69415253 reg_l1 62.44403 reg_l2 32.43246
loss 0.75659657
STEP 47 ================================
prereg loss 0.6817999 reg_l1 62.446434 reg_l2 32.43446
loss 0.7442463
STEP 48 ================================
prereg loss 0.68021494 reg_l1 62.449043 reg_l2 32.43667
loss 0.742664
STEP 49 ================================
prereg loss 0.6866218 reg_l1 62.45172 reg_l2 32.439087
loss 0.7490735
STEP 50 ================================
prereg loss 0.6964367 reg_l1 62.45437 reg_l2 32.441624
loss 0.7588911
STEP 51 ================================
prereg loss 0.70498735 reg_l1 62.45693 reg_l2 32.444283
loss 0.76744425
cutoff 0.16670623 network size 159
STEP 52 ================================
prereg loss 0.70894444 reg_l1 62.292603 reg_l2 32.41917
loss 0.771237
STEP 53 ================================
prereg loss 0.7070492 reg_l1 62.294754 reg_l2 32.421852
loss 0.769344
STEP 54 ================================
prereg loss 0.700134 reg_l1 62.29665 reg_l2 32.424458
loss 0.7624306
STEP 55 ================================
prereg loss 0.69045466 reg_l1 62.298214 reg_l2 32.426952
loss 0.7527529
STEP 56 ================================
prereg loss 0.68077713 reg_l1 62.29952 reg_l2 32.429264
loss 0.7430767
STEP 57 ================================
prereg loss 0.6734725 reg_l1 62.300503 reg_l2 32.431377
loss 0.735773
STEP 58 ================================
prereg loss 0.6698567 reg_l1 62.301228 reg_l2 32.433243
loss 0.73215795
STEP 59 ================================
prereg loss 0.6699096 reg_l1 62.301678 reg_l2 32.43486
loss 0.7322113
STEP 60 ================================
prereg loss 0.67264056 reg_l1 62.30189 reg_l2 32.436188
loss 0.73494244
STEP 61 ================================
prereg loss 0.6764022 reg_l1 62.301914 reg_l2 32.437214
loss 0.73870414
cutoff 0.16807584 network size 158
STEP 62 ================================
prereg loss 0.67925817 reg_l1 62.13366 reg_l2 32.409718
loss 0.74139184
STEP 63 ================================
prereg loss 0.6804365 reg_l1 62.13351 reg_l2 32.410408
loss 0.74257
STEP 64 ================================
prereg loss 0.6795158 reg_l1 62.13348 reg_l2 32.411045
loss 0.74164927
STEP 65 ================================
prereg loss 0.67675114 reg_l1 62.133514 reg_l2 32.411625
loss 0.7388846
STEP 66 ================================
prereg loss 0.6729067 reg_l1 62.133568 reg_l2 32.412174
loss 0.73504025
STEP 67 ================================
prereg loss 0.6689322 reg_l1 62.133686 reg_l2 32.41268
loss 0.73106587
STEP 68 ================================
prereg loss 0.6656472 reg_l1 62.133785 reg_l2 32.413147
loss 0.727781
STEP 69 ================================
prereg loss 0.6635035 reg_l1 62.133797 reg_l2 32.41359
loss 0.7256373
STEP 70 ================================
prereg loss 0.662545 reg_l1 62.13383 reg_l2 32.414
loss 0.7246789
STEP 71 ================================
prereg loss 0.6624711 reg_l1 62.13378 reg_l2 32.414406
loss 0.7246049
cutoff 0.16778627 network size 157
STEP 72 ================================
prereg loss 7.2129183 reg_l1 61.965893 reg_l2 32.38667
loss 7.274884
STEP 73 ================================
prereg loss 7.0025163 reg_l1 61.98026 reg_l2 32.400635
loss 7.0644965
STEP 74 ================================
prereg loss 6.598829 reg_l1 62.007263 reg_l2 32.426483
loss 6.660836
STEP 75 ================================
prereg loss 6.0428047 reg_l1 62.045197 reg_l2 32.462605
loss 6.10485
STEP 76 ================================
prereg loss 5.378168 reg_l1 62.09225 reg_l2 32.507454
loss 5.4402604
STEP 77 ================================
prereg loss 4.649933 reg_l1 62.146797 reg_l2 32.559486
loss 4.7120795
STEP 78 ================================
prereg loss 3.9023201 reg_l1 62.207115 reg_l2 32.61718
loss 3.9645274
STEP 79 ================================
prereg loss 3.1770527 reg_l1 62.271637 reg_l2 32.679092
loss 3.2393243
STEP 80 ================================
prereg loss 2.5115514 reg_l1 62.338787 reg_l2 32.743713
loss 2.5738902
STEP 81 ================================
prereg loss 1.9366589 reg_l1 62.407005 reg_l2 32.80963
loss 1.9990659
cutoff 0.17283721 network size 156
STEP 82 ================================
prereg loss 1.4751188 reg_l1 62.301933 reg_l2 32.84552
loss 1.5374207
STEP 83 ================================
prereg loss 1.1398642 reg_l1 62.36783 reg_l2 32.909782
loss 1.202232
STEP 84 ================================
prereg loss 0.93312836 reg_l1 62.430508 reg_l2 32.97119
loss 0.99555886
STEP 85 ================================
prereg loss 0.8463732 reg_l1 62.488667 reg_l2 33.028492
loss 0.9088619
STEP 86 ================================
prereg loss 0.8591673 reg_l1 62.541157 reg_l2 33.080555
loss 0.92170846
STEP 87 ================================
prereg loss 0.94618595 reg_l1 62.587135 reg_l2 33.126564
loss 1.0087731
STEP 88 ================================
prereg loss 1.0770053 reg_l1 62.62591 reg_l2 33.1658
loss 1.1396312
STEP 89 ================================
prereg loss 1.2238933 reg_l1 62.657085 reg_l2 33.19781
loss 1.2865504
STEP 90 ================================
prereg loss 1.3609948 reg_l1 62.680374 reg_l2 33.222313
loss 1.4236752
STEP 91 ================================
prereg loss 1.4681349 reg_l1 62.695793 reg_l2 33.239326
loss 1.5308306
cutoff 0.17695485 network size 155
STEP 92 ================================
prereg loss 1.5324843 reg_l1 62.526638 reg_l2 33.21772
loss 1.595011
STEP 93 ================================
prereg loss 1.5490676 reg_l1 62.527306 reg_l2 33.220566
loss 1.6115949
STEP 94 ================================
prereg loss 1.5200036 reg_l1 62.521435 reg_l2 33.21715
loss 1.582525
STEP 95 ================================
prereg loss 1.452921 reg_l1 62.5098 reg_l2 33.208214
loss 1.5154308
STEP 96 ================================
prereg loss 1.3588947 reg_l1 62.49327 reg_l2 33.194622
loss 1.421388
STEP 97 ================================
prereg loss 1.2503986 reg_l1 62.472797 reg_l2 33.177258
loss 1.3128715
STEP 98 ================================
prereg loss 1.1391168 reg_l1 62.44932 reg_l2 33.157055
loss 1.2015661
STEP 99 ================================
prereg loss 1.0347749 reg_l1 62.42384 reg_l2 33.13496
loss 1.0971987
STEP 100 ================================
prereg loss 0.94462 reg_l1 62.397194 reg_l2 33.111786
loss 1.0070173
2022-07-20T12:00:53.520

julia>
julia> serialize("cf-155-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-155-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T12:21:58.160
STEP 1 ================================
prereg loss 0.8727524 reg_l1 62.370243 reg_l2 33.088337
loss 0.93512267
cutoff 0.1772983 network size 154
STEP 2 ================================
prereg loss 0.8197436 reg_l1 62.166294 reg_l2 33.033817
loss 0.88190985
STEP 3 ================================
prereg loss 0.7861592 reg_l1 62.140636 reg_l2 33.01169
loss 0.84829986
STEP 4 ================================
prereg loss 0.7695361 reg_l1 62.11654 reg_l2 32.991108
loss 0.83165264
STEP 5 ================================
prereg loss 0.76609945 reg_l1 62.094498 reg_l2 32.97246
loss 0.82819396
STEP 6 ================================
prereg loss 0.7721108 reg_l1 62.07471 reg_l2 32.955994
loss 0.83418554
STEP 7 ================================
prereg loss 0.78402245 reg_l1 62.057556 reg_l2 32.941986
loss 0.84608
STEP 8 ================================
prereg loss 0.79813457 reg_l1 62.04313 reg_l2 32.930573
loss 0.8601777
STEP 9 ================================
prereg loss 0.8114997 reg_l1 62.031536 reg_l2 32.921818
loss 0.8735312
STEP 10 ================================
prereg loss 0.82194346 reg_l1 62.022717 reg_l2 32.91566
loss 0.8839662
STEP 11 ================================
prereg loss 0.82809645 reg_l1 62.016598 reg_l2 32.912083
loss 0.89011306
cutoff 0.18128853 network size 153
STEP 12 ================================
prereg loss 0.8319427 reg_l1 61.831764 reg_l2 32.87804
loss 0.89377445
STEP 13 ================================
prereg loss 0.82835174 reg_l1 61.83065 reg_l2 32.87919
loss 0.8901824
STEP 14 ================================
prereg loss 0.82048875 reg_l1 61.8318 reg_l2 32.882435
loss 0.8823205
STEP 15 ================================
prereg loss 0.80928296 reg_l1 61.834877 reg_l2 32.887524
loss 0.87111783
STEP 16 ================================
prereg loss 0.7958587 reg_l1 61.839584 reg_l2 32.894154
loss 0.85769826
STEP 17 ================================
prereg loss 0.78140014 reg_l1 61.84565 reg_l2 32.902065
loss 0.8432458
STEP 18 ================================
prereg loss 0.7670127 reg_l1 61.8527 reg_l2 32.910927
loss 0.8288654
STEP 19 ================================
prereg loss 0.753632 reg_l1 61.86052 reg_l2 32.920475
loss 0.8154925
STEP 20 ================================
prereg loss 0.7419701 reg_l1 61.86875 reg_l2 32.930428
loss 0.80383885
STEP 21 ================================
prereg loss 0.73246396 reg_l1 61.877144 reg_l2 32.940525
loss 0.7943411
cutoff 0.18289489 network size 152
STEP 22 ================================
prereg loss 0.7252949 reg_l1 61.702538 reg_l2 32.91709
loss 0.78699744
STEP 23 ================================
prereg loss 0.7203985 reg_l1 61.710484 reg_l2 32.926765
loss 0.78210896
STEP 24 ================================
prereg loss 0.71751124 reg_l1 61.717934 reg_l2 32.935963
loss 0.77922916
STEP 25 ================================
prereg loss 0.7162356 reg_l1 61.724716 reg_l2 32.94452
loss 0.7779603
STEP 26 ================================
prereg loss 0.71598136 reg_l1 61.730667 reg_l2 32.952316
loss 0.77771205
STEP 27 ================================
prereg loss 0.71636415 reg_l1 61.735794 reg_l2 32.959328
loss 0.77809995
STEP 28 ================================
prereg loss 0.71704936 reg_l1 61.74004 reg_l2 32.96551
loss 0.7787894
STEP 29 ================================
prereg loss 0.71766806 reg_l1 61.743378 reg_l2 32.97085
loss 0.77941144
STEP 30 ================================
prereg loss 0.717928 reg_l1 61.74578 reg_l2 32.975323
loss 0.77967376
STEP 31 ================================
prereg loss 0.7176355 reg_l1 61.747307 reg_l2 32.978954
loss 0.7793828
cutoff 0.18294476 network size 151
STEP 32 ================================
prereg loss 0.716693 reg_l1 61.565025 reg_l2 32.94832
loss 0.778258
STEP 33 ================================
prereg loss 0.71509576 reg_l1 61.564953 reg_l2 32.950428
loss 0.77666074
STEP 34 ================================
prereg loss 0.7129092 reg_l1 61.5642 reg_l2 32.951893
loss 0.7744734
STEP 35 ================================
prereg loss 0.71026397 reg_l1 61.56286 reg_l2 32.952793
loss 0.7718268
STEP 36 ================================
prereg loss 0.70731014 reg_l1 61.560986 reg_l2 32.953236
loss 0.7688711
STEP 37 ================================
prereg loss 0.704216 reg_l1 61.558792 reg_l2 32.9533
loss 0.7657748
STEP 38 ================================
prereg loss 0.7011396 reg_l1 61.556282 reg_l2 32.95311
loss 0.7626959
STEP 39 ================================
prereg loss 0.69824296 reg_l1 61.55362 reg_l2 32.95276
loss 0.75979656
STEP 40 ================================
prereg loss 0.6954639 reg_l1 61.550903 reg_l2 32.952347
loss 0.7570148
STEP 41 ================================
prereg loss 0.6929398 reg_l1 61.54815 reg_l2 32.95188
loss 0.754488
cutoff 0.18360439 network size 150
STEP 42 ================================
prereg loss 0.6739769 reg_l1 61.361797 reg_l2 32.917747
loss 0.7353387
STEP 43 ================================
prereg loss 0.67416435 reg_l1 61.360306 reg_l2 32.918312
loss 0.73552465
STEP 44 ================================
prereg loss 0.67399263 reg_l1 61.359753 reg_l2 32.91971
loss 0.7353524
STEP 45 ================================
prereg loss 0.6733899 reg_l1 61.360012 reg_l2 32.921917
loss 0.7347499
STEP 46 ================================
prereg loss 0.6723515 reg_l1 61.361042 reg_l2 32.924847
loss 0.73371255
STEP 47 ================================
prereg loss 0.67093676 reg_l1 61.362766 reg_l2 32.928406
loss 0.7322995
STEP 48 ================================
prereg loss 0.6692439 reg_l1 61.3651 reg_l2 32.93254
loss 0.730609
STEP 49 ================================
prereg loss 0.6673918 reg_l1 61.367874 reg_l2 32.937138
loss 0.72875965
STEP 50 ================================
prereg loss 0.6655099 reg_l1 61.371044 reg_l2 32.9421
loss 0.7268809
STEP 51 ================================
prereg loss 0.663711 reg_l1 61.37453 reg_l2 32.947342
loss 0.72508556
cutoff 0.18361099 network size 149
STEP 52 ================================
prereg loss 0.66209185 reg_l1 61.194553 reg_l2 32.91902
loss 0.7232864
STEP 53 ================================
prereg loss 0.66071254 reg_l1 61.19827 reg_l2 32.924484
loss 0.72191083
STEP 54 ================================
prereg loss 0.65957546 reg_l1 61.201996 reg_l2 32.929943
loss 0.72077745
STEP 55 ================================
prereg loss 0.6586673 reg_l1 61.205658 reg_l2 32.935368
loss 0.719873
STEP 56 ================================
prereg loss 0.6580184 reg_l1 61.209213 reg_l2 32.94069
loss 0.7192276
STEP 57 ================================
prereg loss 0.65758955 reg_l1 61.212627 reg_l2 32.945843
loss 0.7188022
STEP 58 ================================
prereg loss 0.6573264 reg_l1 61.215786 reg_l2 32.95079
loss 0.7185422
STEP 59 ================================
prereg loss 0.6571697 reg_l1 61.218727 reg_l2 32.95549
loss 0.71838844
STEP 60 ================================
prereg loss 0.65705466 reg_l1 61.22136 reg_l2 32.959904
loss 0.718276
STEP 61 ================================
prereg loss 0.6569284 reg_l1 61.22369 reg_l2 32.96403
loss 0.7181521
cutoff 0.18445274 network size 148
STEP 62 ================================
prereg loss 0.6678328 reg_l1 61.041245 reg_l2 32.933815
loss 0.728874
STEP 63 ================================
prereg loss 0.6657777 reg_l1 61.04399 reg_l2 32.938293
loss 0.72682166
STEP 64 ================================
prereg loss 0.66318923 reg_l1 61.04734 reg_l2 32.9433
loss 0.72423655
STEP 65 ================================
prereg loss 0.6604903 reg_l1 61.051098 reg_l2 32.948727
loss 0.7215414
STEP 66 ================================
prereg loss 0.65805066 reg_l1 61.05523 reg_l2 32.95447
loss 0.7191059
STEP 67 ================================
prereg loss 0.65609616 reg_l1 61.05954 reg_l2 32.960396
loss 0.7171557
STEP 68 ================================
prereg loss 0.65468526 reg_l1 61.06394 reg_l2 32.96644
loss 0.7157492
STEP 69 ================================
prereg loss 0.65374 reg_l1 61.068275 reg_l2 32.972458
loss 0.7148083
STEP 70 ================================
prereg loss 0.65311414 reg_l1 61.072445 reg_l2 32.97836
loss 0.7141866
STEP 71 ================================
prereg loss 0.65261704 reg_l1 61.076405 reg_l2 32.984123
loss 0.71369344
cutoff 0.18655899 network size 147
STEP 72 ================================
prereg loss 0.67045015 reg_l1 60.893585 reg_l2 32.95489
loss 0.73134375
STEP 73 ================================
prereg loss 0.664148 reg_l1 60.898434 reg_l2 32.961548
loss 0.7250464
STEP 74 ================================
prereg loss 0.6570575 reg_l1 60.90413 reg_l2 32.96905
loss 0.71796167
STEP 75 ================================
prereg loss 0.65002 reg_l1 60.91042 reg_l2 32.977165
loss 0.7109304
STEP 76 ================================
prereg loss 0.64378285 reg_l1 60.917133 reg_l2 32.98566
loss 0.7047
STEP 77 ================================
prereg loss 0.6388174 reg_l1 60.92404 reg_l2 32.994373
loss 0.6997415
STEP 78 ================================
prereg loss 0.63537765 reg_l1 60.930943 reg_l2 33.00307
loss 0.6963086
STEP 79 ================================
prereg loss 0.63346815 reg_l1 60.937702 reg_l2 33.01162
loss 0.69440585
STEP 80 ================================
prereg loss 0.6328965 reg_l1 60.94414 reg_l2 33.01983
loss 0.6938406
STEP 81 ================================
prereg loss 0.6333408 reg_l1 60.950157 reg_l2 33.027584
loss 0.69429094
cutoff 0.18995412 network size 146
STEP 82 ================================
prereg loss 0.6682458 reg_l1 60.765648 reg_l2 32.9987
loss 0.7290114
STEP 83 ================================
prereg loss 0.6690238 reg_l1 60.77061 reg_l2 33.005318
loss 0.72979444
STEP 84 ================================
prereg loss 0.6697245 reg_l1 60.774967 reg_l2 33.011257
loss 0.7304995
STEP 85 ================================
prereg loss 0.6701302 reg_l1 60.778587 reg_l2 33.016525
loss 0.7309088
STEP 86 ================================
prereg loss 0.670095 reg_l1 60.781548 reg_l2 33.021057
loss 0.73087656
STEP 87 ================================
prereg loss 0.66955054 reg_l1 60.783806 reg_l2 33.024925
loss 0.73033434
STEP 88 ================================
prereg loss 0.6684923 reg_l1 60.785408 reg_l2 33.02812
loss 0.72927773
STEP 89 ================================
prereg loss 0.666959 reg_l1 60.786377 reg_l2 33.030685
loss 0.72774535
STEP 90 ================================
prereg loss 0.6650279 reg_l1 60.786777 reg_l2 33.032673
loss 0.7258147
STEP 91 ================================
prereg loss 0.6627906 reg_l1 60.786663 reg_l2 33.034187
loss 0.72357726
cutoff 0.1936693 network size 145
STEP 92 ================================
prereg loss 0.9053257 reg_l1 60.592484 reg_l2 32.997776
loss 0.9659182
STEP 93 ================================
prereg loss 0.86690134 reg_l1 60.592606 reg_l2 32.99913
loss 0.92749393
STEP 94 ================================
prereg loss 0.8088908 reg_l1 60.593422 reg_l2 33.000805
loss 0.86948425
STEP 95 ================================
prereg loss 0.7503474 reg_l1 60.59483 reg_l2 33.002842
loss 0.81094223
STEP 96 ================================
prereg loss 0.7069339 reg_l1 60.596775 reg_l2 33.005222
loss 0.7675307
STEP 97 ================================
prereg loss 0.6865222 reg_l1 60.599163 reg_l2 33.007988
loss 0.74712133
STEP 98 ================================
prereg loss 0.6883171 reg_l1 60.601856 reg_l2 33.01107
loss 0.748919
STEP 99 ================================
prereg loss 0.70499474 reg_l1 60.604687 reg_l2 33.014404
loss 0.7655994
STEP 100 ================================
prereg loss 0.72537565 reg_l1 60.60757 reg_l2 33.017895
loss 0.7859832
2022-07-20T12:31:20.474

julia> serialize("cf-145-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-145-parameters-opt.ser", opt)
```

cf-165 and cf-145 models are both pretty good.

```
julia> interleaving_steps!(100, 10)
2022-07-20T12:38:06.935
STEP 1 ================================
prereg loss 0.7397718 reg_l1 60.610405 reg_l2 33.021496
loss 0.8003822
cutoff 0.19392657 network size 144
STEP 2 ================================
prereg loss 0.74259967 reg_l1 60.41918 reg_l2 32.98755
loss 0.80301887
STEP 3 ================================
prereg loss 0.7332174 reg_l1 60.42175 reg_l2 32.991207
loss 0.7936392
STEP 4 ================================
prereg loss 0.7151902 reg_l1 60.424145 reg_l2 32.994812
loss 0.7756143
STEP 5 ================================
prereg loss 0.6944109 reg_l1 60.42626 reg_l2 32.998295
loss 0.75483716
STEP 6 ================================
prereg loss 0.67693657 reg_l1 60.428185 reg_l2 33.00161
loss 0.73736477
STEP 7 ================================
prereg loss 0.66711706 reg_l1 60.429886 reg_l2 33.004734
loss 0.72754693
STEP 8 ================================
prereg loss 0.665714 reg_l1 60.431458 reg_l2 33.007713
loss 0.7261455
STEP 9 ================================
prereg loss 0.67091984 reg_l1 60.43292 reg_l2 33.01047
loss 0.73135275
STEP 10 ================================
prereg loss 0.6791705 reg_l1 60.43416 reg_l2 33.013012
loss 0.73960465
STEP 11 ================================
prereg loss 0.68650794 reg_l1 60.43533 reg_l2 33.015312
loss 0.7469433
cutoff 0.19465335 network size 143
STEP 12 ================================
prereg loss 0.68991446 reg_l1 60.24169 reg_l2 32.979473
loss 0.75015616
STEP 13 ================================
prereg loss 0.6881076 reg_l1 60.242577 reg_l2 32.98128
loss 0.7483502
STEP 14 ================================
prereg loss 0.6816379 reg_l1 60.243332 reg_l2 32.982887
loss 0.7418812
STEP 15 ================================
prereg loss 0.6723893 reg_l1 60.243996 reg_l2 32.984295
loss 0.73263335
STEP 16 ================================
prereg loss 0.66273946 reg_l1 60.244564 reg_l2 32.985588
loss 0.722984
STEP 17 ================================
prereg loss 0.654721 reg_l1 60.245056 reg_l2 32.986794
loss 0.71496606
STEP 18 ================================
prereg loss 0.64946526 reg_l1 60.245476 reg_l2 32.98794
loss 0.7097107
STEP 19 ================================
prereg loss 0.6470472 reg_l1 60.24586 reg_l2 32.989086
loss 0.7072931
STEP 20 ================================
prereg loss 0.64669436 reg_l1 60.24621 reg_l2 32.99024
loss 0.7069406
STEP 21 ================================
prereg loss 0.6472195 reg_l1 60.24656 reg_l2 32.99146
loss 0.70746607
cutoff 0.19677795 network size 142
STEP 22 ================================
prereg loss 0.66272366 reg_l1 60.050156 reg_l2 32.954025
loss 0.7227738
STEP 23 ================================
prereg loss 0.6623275 reg_l1 60.05164 reg_l2 32.956432
loss 0.72237915
STEP 24 ================================
prereg loss 0.6601329 reg_l1 60.054016 reg_l2 32.95982
loss 0.7201869
STEP 25 ================================
prereg loss 0.65642285 reg_l1 60.057198 reg_l2 32.96403
loss 0.7164801
STEP 26 ================================
prereg loss 0.6517661 reg_l1 60.061028 reg_l2 32.968937
loss 0.71182716
STEP 27 ================================
prereg loss 0.646862 reg_l1 60.065327 reg_l2 32.974365
loss 0.7069273
STEP 28 ================================
prereg loss 0.64236236 reg_l1 60.070007 reg_l2 32.980175
loss 0.7024324
STEP 29 ================================
prereg loss 0.6387422 reg_l1 60.07487 reg_l2 32.986206
loss 0.6988171
STEP 30 ================================
prereg loss 0.6362311 reg_l1 60.079796 reg_l2 32.992332
loss 0.69631094
STEP 31 ================================
prereg loss 0.6348087 reg_l1 60.08466 reg_l2 32.9984
loss 0.69489336
cutoff 0.19771184 network size 141
STEP 32 ================================
prereg loss 3.8189316 reg_l1 59.891647 reg_l2 32.965195
loss 3.8788233
STEP 33 ================================
prereg loss 3.4171658 reg_l1 59.897053 reg_l2 32.971436
loss 3.4770627
STEP 34 ================================
prereg loss 2.7795203 reg_l1 59.903034 reg_l2 32.978035
loss 2.8394234
STEP 35 ================================
prereg loss 2.1469464 reg_l1 59.909378 reg_l2 32.984997
loss 2.2068558
STEP 36 ================================
prereg loss 1.7037642 reg_l1 59.916058 reg_l2 32.99242
loss 1.7636802
STEP 37 ================================
prereg loss 1.5315177 reg_l1 59.92294 reg_l2 33.000313
loss 1.5914407
STEP 38 ================================
prereg loss 1.6073312 reg_l1 59.929874 reg_l2 33.00858
loss 1.667261
STEP 39 ================================
prereg loss 1.8325355 reg_l1 59.93659 reg_l2 33.01699
loss 1.8924721
STEP 40 ================================
prereg loss 2.0831802 reg_l1 59.942913 reg_l2 33.025352
loss 2.1431231
STEP 41 ================================
prereg loss 2.255836 reg_l1 59.948673 reg_l2 33.03349
loss 2.3157847
cutoff 0.19162688 network size 140
STEP 42 ================================
prereg loss 19.44829 reg_l1 59.76211 reg_l2 33.004475
loss 19.508053
STEP 43 ================================
prereg loss 17.581408 reg_l1 59.756973 reg_l2 33.00364
loss 17.641165
STEP 44 ================================
prereg loss 14.385131 reg_l1 59.7441 reg_l2 32.996483
loss 14.444875
STEP 45 ================================
prereg loss 10.673547 reg_l1 59.724915 reg_l2 32.984245
loss 10.733272
STEP 46 ================================
prereg loss 7.177667 reg_l1 59.70031 reg_l2 32.967762
loss 7.2373676
STEP 47 ================================
prereg loss 4.470287 reg_l1 59.67109 reg_l2 32.94766
loss 4.529958
STEP 48 ================================
prereg loss 2.916829 reg_l1 59.637657 reg_l2 32.924145
loss 2.9764667
STEP 49 ================================
prereg loss 2.6130958 reg_l1 59.600403 reg_l2 32.89723
loss 2.672696
STEP 50 ================================
prereg loss 3.3555026 reg_l1 59.559723 reg_l2 32.86688
loss 3.4150624
STEP 51 ================================
prereg loss 4.6894445 reg_l1 59.516285 reg_l2 32.833183
loss 4.748961
cutoff 0.1807492 network size 139
STEP 52 ================================
prereg loss 8.87089 reg_l1 59.29015 reg_l2 32.763725
loss 8.93018
STEP 53 ================================
prereg loss 9.804788 reg_l1 59.24625 reg_l2 32.725494
loss 9.864034
STEP 54 ================================
prereg loss 9.543468 reg_l1 59.203003 reg_l2 32.686047
loss 9.602672
STEP 55 ================================
prereg loss 8.31577 reg_l1 59.161438 reg_l2 32.646667
loss 8.374931
STEP 56 ================================
prereg loss 6.5689135 reg_l1 59.122673 reg_l2 32.608807
loss 6.628036
STEP 57 ================================
prereg loss 4.777052 reg_l1 59.087738 reg_l2 32.573906
loss 4.8361397
STEP 58 ================================
prereg loss 3.3126664 reg_l1 59.057518 reg_l2 32.5432
loss 3.371724
STEP 59 ================================
prereg loss 2.3853278 reg_l1 59.032646 reg_l2 32.517647
loss 2.4443605
STEP 60 ================================
prereg loss 2.0381956 reg_l1 59.013664 reg_l2 32.498047
loss 2.0972092
STEP 61 ================================
prereg loss 2.1790857 reg_l1 59.000782 reg_l2 32.484802
loss 2.2380865
cutoff 0.19700466 network size 138
STEP 62 ================================
prereg loss 3.3659718 reg_l1 58.796906 reg_l2 32.43918
loss 3.4247687
STEP 63 ================================
prereg loss 3.9749033 reg_l1 58.797695 reg_l2 32.44056
loss 4.033701
STEP 64 ================================
prereg loss 4.485804 reg_l1 58.805153 reg_l2 32.449314
loss 4.544609
STEP 65 ================================
prereg loss 4.7787585 reg_l1 58.81851 reg_l2 32.464615
loss 4.837577
STEP 66 ================================
prereg loss 4.7984123 reg_l1 58.8368 reg_l2 32.48548
loss 4.8572493
STEP 67 ================================
prereg loss 4.5504794 reg_l1 58.85912 reg_l2 32.510967
loss 4.6093388
STEP 68 ================================
prereg loss 4.08927 reg_l1 58.88454 reg_l2 32.540062
loss 4.1481547
STEP 69 ================================
prereg loss 3.5001876 reg_l1 58.91216 reg_l2 32.571754
loss 3.5591
STEP 70 ================================
prereg loss 2.8801951 reg_l1 58.940994 reg_l2 32.605022
loss 2.939136
STEP 71 ================================
prereg loss 2.3210704 reg_l1 58.970177 reg_l2 32.638866
loss 2.3800406
cutoff 0.19950789 network size 137
STEP 72 ================================
prereg loss 1.9531723 reg_l1 58.799175 reg_l2 32.63237
loss 2.0119715
STEP 73 ================================
prereg loss 1.6595823 reg_l1 58.8268 reg_l2 32.664875
loss 1.7184091
STEP 74 ================================
prereg loss 1.5393101 reg_l1 58.85295 reg_l2 32.695778
loss 1.598163
STEP 75 ================================
prereg loss 1.5698019 reg_l1 58.876945 reg_l2 32.724247
loss 1.6286789
STEP 76 ================================
prereg loss 1.7046598 reg_l1 58.898296 reg_l2 32.749664
loss 1.7635581
STEP 77 ================================
prereg loss 1.8869269 reg_l1 58.916534 reg_l2 32.771492
loss 1.9458435
STEP 78 ================================
prereg loss 2.0614321 reg_l1 58.93143 reg_l2 32.78939
loss 2.1203635
STEP 79 ================================
prereg loss 2.1850657 reg_l1 58.942833 reg_l2 32.8032
loss 2.2440085
STEP 80 ================================
prereg loss 2.2338588 reg_l1 58.950798 reg_l2 32.812943
loss 2.2928097
STEP 81 ================================
prereg loss 2.2043717 reg_l1 58.955433 reg_l2 32.818794
loss 2.2633271
cutoff 0.20397727 network size 136
STEP 82 ================================
prereg loss 1.9102817 reg_l1 58.753075 reg_l2 32.779526
loss 1.9690348
STEP 83 ================================
prereg loss 1.8009305 reg_l1 58.753845 reg_l2 32.78058
loss 1.8596843
STEP 84 ================================
prereg loss 1.6877558 reg_l1 58.753845 reg_l2 32.78063
loss 1.7465097
STEP 85 ================================
prereg loss 1.5902798 reg_l1 58.753387 reg_l2 32.78002
loss 1.6490332
STEP 86 ================================
prereg loss 1.5214103 reg_l1 58.7526 reg_l2 32.77901
loss 1.580163
STEP 87 ================================
prereg loss 1.4857459 reg_l1 58.751793 reg_l2 32.77794
loss 1.5444977
STEP 88 ================================
prereg loss 1.4802983 reg_l1 58.75111 reg_l2 32.777046
loss 1.5390494
STEP 89 ================================
prereg loss 1.497366 reg_l1 58.75071 reg_l2 32.77653
loss 1.5561167
STEP 90 ================================
prereg loss 1.5260419 reg_l1 58.750675 reg_l2 32.77653
loss 1.5847925
STEP 91 ================================
prereg loss 1.5558697 reg_l1 58.751106 reg_l2 32.77718
loss 1.6146208
cutoff 0.2017569 network size 135
STEP 92 ================================
prereg loss 82.40871 reg_l1 58.550316 reg_l2 32.737835
loss 82.467255
STEP 93 ================================
prereg loss 79.644615 reg_l1 58.548046 reg_l2 32.73507
loss 79.70316
STEP 94 ================================
prereg loss 74.61068 reg_l1 58.542877 reg_l2 32.729004
loss 74.66922
STEP 95 ================================
prereg loss 67.9871 reg_l1 58.53519 reg_l2 32.720276
loss 68.04563
STEP 96 ================================
prereg loss 60.350235 reg_l1 58.52514 reg_l2 32.70941
loss 60.40876
STEP 97 ================================
prereg loss 52.191444 reg_l1 58.512844 reg_l2 32.696606
loss 52.249958
STEP 98 ================================
prereg loss 43.92161 reg_l1 58.498524 reg_l2 32.68226
loss 43.98011
STEP 99 ================================
prereg loss 35.909924 reg_l1 58.482357 reg_l2 32.666534
loss 35.968407
STEP 100 ================================
prereg loss 28.464924 reg_l1 58.46433 reg_l2 32.649494
loss 28.523388
2022-07-20T12:47:19.034

julia> interleaving_steps!(100, 10)
2022-07-20T12:51:30.036
STEP 1 ================================
prereg loss 21.838644 reg_l1 58.444557 reg_l2 32.6311
loss 21.897089
cutoff 0.2046135 network size 134
STEP 2 ================================
prereg loss 16.224676 reg_l1 58.218422 reg_l2 32.569424
loss 16.282894
STEP 3 ================================
prereg loss 11.753709 reg_l1 58.195198 reg_l2 32.54815
loss 11.811904
STEP 4 ================================
prereg loss 8.486243 reg_l1 58.170433 reg_l2 32.525307
loss 8.544414
STEP 5 ================================
prereg loss 6.409921 reg_l1 58.144196 reg_l2 32.500835
loss 6.4680653
STEP 6 ================================
prereg loss 5.435598 reg_l1 58.116653 reg_l2 32.47471
loss 5.4937143
STEP 7 ================================
prereg loss 5.400454 reg_l1 58.088013 reg_l2 32.446884
loss 5.458542
STEP 8 ================================
prereg loss 6.0815153 reg_l1 58.05847 reg_l2 32.417465
loss 6.1395736
STEP 9 ================================
prereg loss 7.2150693 reg_l1 58.0283 reg_l2 32.38655
loss 7.2730975
STEP 10 ================================
prereg loss 8.530533 reg_l1 57.997868 reg_l2 32.35443
loss 8.588531
STEP 11 ================================
prereg loss 9.777528 reg_l1 57.967556 reg_l2 32.321472
loss 9.835495
cutoff 0.20664372 network size 133
STEP 12 ================================
prereg loss 12.172334 reg_l1 57.731205 reg_l2 32.245373
loss 12.230065
STEP 13 ================================
prereg loss 12.679965 reg_l1 57.70118 reg_l2 32.21051
loss 12.737666
STEP 14 ================================
prereg loss 12.713755 reg_l1 57.67116 reg_l2 32.174908
loss 12.771426
STEP 15 ================================
prereg loss 12.322494 reg_l1 57.642086 reg_l2 32.13964
loss 12.380136
STEP 16 ================================
prereg loss 11.602768 reg_l1 57.614906 reg_l2 32.10575
loss 11.660383
STEP 17 ================================
prereg loss 10.673435 reg_l1 57.590412 reg_l2 32.07419
loss 10.731026
STEP 18 ================================
prereg loss 9.654742 reg_l1 57.569298 reg_l2 32.045795
loss 9.712312
STEP 19 ================================
prereg loss 8.65126 reg_l1 57.552227 reg_l2 32.021294
loss 8.708813
STEP 20 ================================
prereg loss 7.742506 reg_l1 57.539513 reg_l2 32.001186
loss 7.8000455
STEP 21 ================================
prereg loss 6.980367 reg_l1 57.531498 reg_l2 31.985842
loss 7.0378985
cutoff 0.20967726 network size 132
STEP 22 ================================
prereg loss 6.389603 reg_l1 57.318615 reg_l2 31.931498
loss 6.446922
STEP 23 ================================
prereg loss 5.972217 reg_l1 57.320164 reg_l2 31.926123
loss 6.029537
STEP 24 ================================
prereg loss 5.7121525 reg_l1 57.32636 reg_l2 31.925667
loss 5.769479
STEP 25 ================================
prereg loss 5.5793877 reg_l1 57.33708 reg_l2 31.930027
loss 5.636725
STEP 26 ================================
prereg loss 5.544436 reg_l1 57.351974 reg_l2 31.938902
loss 5.601788
STEP 27 ================================
prereg loss 5.574075 reg_l1 57.370613 reg_l2 31.951885
loss 5.631446
STEP 28 ================================
prereg loss 5.6373525 reg_l1 57.392536 reg_l2 31.968458
loss 5.694745
STEP 29 ================================
prereg loss 5.7077193 reg_l1 57.417236 reg_l2 31.988104
loss 5.7651367
STEP 30 ================================
prereg loss 5.7644186 reg_l1 57.444168 reg_l2 32.01029
loss 5.8218627
STEP 31 ================================
prereg loss 5.795172 reg_l1 57.47276 reg_l2 32.0344
loss 5.852645
cutoff 0.20953082 network size 131
STEP 32 ================================
prereg loss 5.793163 reg_l1 57.2929 reg_l2 32.015938
loss 5.8504558
STEP 33 ================================
prereg loss 5.7567234 reg_l1 57.323242 reg_l2 32.04215
loss 5.814047
STEP 34 ================================
prereg loss 5.6884775 reg_l1 57.353542 reg_l2 32.068466
loss 5.745831
STEP 35 ================================
prereg loss 5.593672 reg_l1 57.3833 reg_l2 32.094303
loss 5.6510553
STEP 36 ================================
prereg loss 5.4792094 reg_l1 57.411964 reg_l2 32.119175
loss 5.5366216
STEP 37 ================================
prereg loss 5.352683 reg_l1 57.439137 reg_l2 32.14257
loss 5.4101224
STEP 38 ================================
prereg loss 5.220637 reg_l1 57.46442 reg_l2 32.16405
loss 5.2781014
STEP 39 ================================
prereg loss 5.088919 reg_l1 57.487495 reg_l2 32.183285
loss 5.1464067
STEP 40 ================================
prereg loss 4.962246 reg_l1 57.50814 reg_l2 32.199978
loss 5.019754
STEP 41 ================================
prereg loss 4.843615 reg_l1 57.526154 reg_l2 32.21395
loss 4.901141
cutoff 0.21148126 network size 130
STEP 42 ================================
prereg loss 4.7349186 reg_l1 57.330036 reg_l2 32.180393
loss 4.7922487
STEP 43 ================================
prereg loss 4.637013 reg_l1 57.34268 reg_l2 32.188732
loss 4.6943555
STEP 44 ================================
prereg loss 4.5499635 reg_l1 57.35275 reg_l2 32.194298
loss 4.607316
STEP 45 ================================
prereg loss 4.473272 reg_l1 57.36033 reg_l2 32.197243
loss 4.530632
STEP 46 ================================
prereg loss 4.406097 reg_l1 57.36566 reg_l2 32.197792
loss 4.463463
STEP 47 ================================
prereg loss 4.3473916 reg_l1 57.369 reg_l2 32.196194
loss 4.404761
STEP 48 ================================
prereg loss 4.2960815 reg_l1 57.370636 reg_l2 32.19278
loss 4.353452
STEP 49 ================================
prereg loss 4.251044 reg_l1 57.370903 reg_l2 32.18789
loss 4.308415
STEP 50 ================================
prereg loss 4.211176 reg_l1 57.370148 reg_l2 32.18188
loss 4.268546
STEP 51 ================================
prereg loss 4.1754484 reg_l1 57.36872 reg_l2 32.175125
loss 4.232817
cutoff 0.21347646 network size 129
STEP 52 ================================
prereg loss 4.142936 reg_l1 57.153477 reg_l2 32.122417
loss 4.20009
STEP 53 ================================
prereg loss 4.1129603 reg_l1 57.1517 reg_l2 32.115253
loss 4.170112
STEP 54 ================================
prereg loss 4.0846176 reg_l1 57.150246 reg_l2 32.108418
loss 4.141768
STEP 55 ================================
prereg loss 4.057275 reg_l1 57.149338 reg_l2 32.10214
loss 4.114424
STEP 56 ================================
prereg loss 4.030492 reg_l1 57.149273 reg_l2 32.096725
loss 4.0876412
STEP 57 ================================
prereg loss 4.003931 reg_l1 57.150196 reg_l2 32.09237
loss 4.0610814
STEP 58 ================================
prereg loss 3.9775343 reg_l1 57.15227 reg_l2 32.089268
loss 4.0346866
STEP 59 ================================
prereg loss 3.9510934 reg_l1 57.15567 reg_l2 32.0875
loss 4.0082493
STEP 60 ================================
prereg loss 3.9246504 reg_l1 57.160404 reg_l2 32.087227
loss 3.9818108
STEP 61 ================================
prereg loss 3.8982215 reg_l1 57.166553 reg_l2 32.088432
loss 3.955388
cutoff 0.21360491 network size 128
STEP 62 ================================
prereg loss 3.8719127 reg_l1 56.96047 reg_l2 32.0455
loss 3.9288733
STEP 63 ================================
prereg loss 3.8457706 reg_l1 56.969376 reg_l2 32.04967
loss 3.90274
STEP 64 ================================
prereg loss 3.8198948 reg_l1 56.97952 reg_l2 32.055214
loss 3.8768742
STEP 65 ================================
prereg loss 3.7942948 reg_l1 56.990875 reg_l2 32.062057
loss 3.8512857
STEP 66 ================================
prereg loss 3.7690244 reg_l1 57.003284 reg_l2 32.07008
loss 3.8260276
STEP 67 ================================
prereg loss 3.7440758 reg_l1 57.01663 reg_l2 32.079124
loss 3.8010924
STEP 68 ================================
prereg loss 3.719471 reg_l1 57.030746 reg_l2 32.08905
loss 3.7765017
STEP 69 ================================
prereg loss 3.6952097 reg_l1 57.045498 reg_l2 32.09967
loss 3.7522552
STEP 70 ================================
prereg loss 3.671306 reg_l1 57.06068 reg_l2 32.11083
loss 3.7283666
STEP 71 ================================
prereg loss 3.6477334 reg_l1 57.076176 reg_l2 32.12236
loss 3.7048097
cutoff 0.21866491 network size 127
STEP 72 ================================
prereg loss 3.6245089 reg_l1 56.87311 reg_l2 32.08624
loss 3.681382
STEP 73 ================================
prereg loss 3.601635 reg_l1 56.888752 reg_l2 32.098003
loss 3.6585238
STEP 74 ================================
prereg loss 3.5790563 reg_l1 56.904213 reg_l2 32.109627
loss 3.6359606
STEP 75 ================================
prereg loss 3.5567896 reg_l1 56.919388 reg_l2 32.120975
loss 3.613709
STEP 76 ================================
prereg loss 3.5347857 reg_l1 56.934128 reg_l2 32.131943
loss 3.5917199
STEP 77 ================================
prereg loss 3.5129995 reg_l1 56.948387 reg_l2 32.142376
loss 3.569948
STEP 78 ================================
prereg loss 3.4913747 reg_l1 56.96199 reg_l2 32.152237
loss 3.5483367
STEP 79 ================================
prereg loss 3.4698703 reg_l1 56.975002 reg_l2 32.16144
loss 3.5268452
STEP 80 ================================
prereg loss 3.4484801 reg_l1 56.987293 reg_l2 32.169952
loss 3.5054674
STEP 81 ================================
prereg loss 3.4271944 reg_l1 56.9989 reg_l2 32.177753
loss 3.4841933
cutoff 0.2192402 network size 126
STEP 82 ================================
prereg loss 30.765188 reg_l1 56.790543 reg_l2 32.136757
loss 30.82198
STEP 83 ================================
prereg loss 29.962097 reg_l1 56.82354 reg_l2 32.168762
loss 30.01892
STEP 84 ================================
prereg loss 28.597197 reg_l1 56.875202 reg_l2 32.221916
loss 28.654072
STEP 85 ================================
prereg loss 26.799253 reg_l1 56.9426 reg_l2 32.293
loss 26.856195
STEP 86 ================================
prereg loss 24.494139 reg_l1 57.02262 reg_l2 32.378685
loss 24.55116
STEP 87 ================================
prereg loss 21.892237 reg_l1 57.11552 reg_l2 32.47901
loss 21.949352
STEP 88 ================================
prereg loss 19.139194 reg_l1 57.218956 reg_l2 32.59147
loss 19.196413
STEP 89 ================================
prereg loss 16.362606 reg_l1 57.330654 reg_l2 32.713615
loss 16.419937
STEP 90 ================================
prereg loss 13.682619 reg_l1 57.448513 reg_l2 32.843204
loss 13.7400675
STEP 91 ================================
prereg loss 11.265259 reg_l1 57.570496 reg_l2 32.97796
loss 11.322829
cutoff 0.2214371 network size 125
STEP 92 ================================
prereg loss 9.202319 reg_l1 57.47194 reg_l2 33.065502
loss 9.259791
STEP 93 ================================
prereg loss 7.430223 reg_l1 57.59408 reg_l2 33.20194
loss 7.4878173
STEP 94 ================================
prereg loss 5.998945 reg_l1 57.714455 reg_l2 33.337036
loss 6.0566597
STEP 95 ================================
prereg loss 4.93719 reg_l1 57.83153 reg_l2 33.46892
loss 4.995022
STEP 96 ================================
prereg loss 4.245742 reg_l1 57.94373 reg_l2 33.595665
loss 4.3036857
STEP 97 ================================
prereg loss 3.9020884 reg_l1 58.04951 reg_l2 33.715324
loss 3.9601378
STEP 98 ================================
prereg loss 3.8656223 reg_l1 58.14742 reg_l2 33.82603
loss 3.9237697
STEP 99 ================================
prereg loss 4.0704722 reg_l1 58.235985 reg_l2 33.92592
loss 4.1287084
STEP 100 ================================
prereg loss 4.4270663 reg_l1 58.313778 reg_l2 34.013184
loss 4.48538
2022-07-20T13:00:03.773

julia> serialize("cf-125-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-125-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T13:04:14.582
STEP 1 ================================
prereg loss 4.8377986 reg_l1 58.379936 reg_l2 34.08664
loss 4.8961787
cutoff 0.2241193 network size 124
STEP 2 ================================
prereg loss 5.145268 reg_l1 58.20976 reg_l2 34.09538
loss 5.203478
STEP 3 ================================
prereg loss 5.3435173 reg_l1 58.253517 reg_l2 34.142284
loss 5.4017706
STEP 4 ================================
prereg loss 5.4660606 reg_l1 58.28732 reg_l2 34.177494
loss 5.524348
STEP 5 ================================
prereg loss 5.5031643 reg_l1 58.31184 reg_l2 34.201614
loss 5.561476
STEP 6 ================================
prereg loss 5.455319 reg_l1 58.327667 reg_l2 34.21549
loss 5.5136466
STEP 7 ================================
prereg loss 5.3313246 reg_l1 58.33573 reg_l2 34.22011
loss 5.3896604
STEP 8 ================================
prereg loss 5.1457324 reg_l1 58.33692 reg_l2 34.216583
loss 5.204069
STEP 9 ================================
prereg loss 4.9162045 reg_l1 58.33227 reg_l2 34.20613
loss 4.974537
STEP 10 ================================
prereg loss 4.661106 reg_l1 58.322758 reg_l2 34.18993
loss 4.719429
STEP 11 ================================
prereg loss 4.397637 reg_l1 58.309452 reg_l2 34.169235
loss 4.4559464
cutoff 0.23023386 network size 123
STEP 12 ================================
prereg loss 4.1404595 reg_l1 58.063072 reg_l2 34.09217
loss 4.1985226
STEP 13 ================================
prereg loss 3.9009633 reg_l1 58.04498 reg_l2 34.065865
loss 3.9590082
STEP 14 ================================
prereg loss 3.662982 reg_l1 58.02582 reg_l2 34.03832
loss 3.7210078
STEP 15 ================================
prereg loss 3.4330184 reg_l1 58.005245 reg_l2 34.009026
loss 3.4910238
STEP 16 ================================
prereg loss 3.24518 reg_l1 57.984303 reg_l2 33.979195
loss 3.3031642
STEP 17 ================================
prereg loss 3.1031315 reg_l1 57.96381 reg_l2 33.949936
loss 3.1610954
STEP 18 ================================
prereg loss 3.0055506 reg_l1 57.944595 reg_l2 33.92213
loss 3.0634952
STEP 19 ================================
prereg loss 2.9473157 reg_l1 57.927273 reg_l2 33.896564
loss 3.005243
STEP 20 ================================
prereg loss 2.921451 reg_l1 57.912445 reg_l2 33.87392
loss 2.9793634
STEP 21 ================================
prereg loss 2.918545 reg_l1 57.90045 reg_l2 33.85462
loss 2.9764454
cutoff 0.22415166 network size 122
STEP 22 ================================
prereg loss 3.172671 reg_l1 57.66739 reg_l2 33.788734
loss 3.2303386
STEP 23 ================================
prereg loss 3.1960266 reg_l1 57.66308 reg_l2 33.778103
loss 3.2536895
STEP 24 ================================
prereg loss 3.214873 reg_l1 57.66266 reg_l2 33.772133
loss 3.2725358
STEP 25 ================================
prereg loss 3.2250192 reg_l1 57.66598 reg_l2 33.77064
loss 3.2826853
STEP 26 ================================
prereg loss 3.2241504 reg_l1 57.672825 reg_l2 33.773357
loss 3.2818232
STEP 27 ================================
prereg loss 3.2116091 reg_l1 57.682873 reg_l2 33.779953
loss 3.269292
STEP 28 ================================
prereg loss 3.1880252 reg_l1 57.695835 reg_l2 33.79004
loss 3.245721
STEP 29 ================================
prereg loss 3.1550055 reg_l1 57.711308 reg_l2 33.803185
loss 3.2127168
STEP 30 ================================
prereg loss 3.114764 reg_l1 57.72895 reg_l2 33.818935
loss 3.172493
STEP 31 ================================
prereg loss 3.0698166 reg_l1 57.74832 reg_l2 33.83681
loss 3.127565
cutoff 0.23236978 network size 121
STEP 32 ================================
prereg loss 3.0227194 reg_l1 57.536667 reg_l2 33.802338
loss 3.080256
STEP 33 ================================
prereg loss 2.975806 reg_l1 57.558353 reg_l2 33.823067
loss 3.0333643
STEP 34 ================================
prereg loss 2.9310884 reg_l1 57.58058 reg_l2 33.84449
loss 2.988669
STEP 35 ================================
prereg loss 2.8901005 reg_l1 57.60298 reg_l2 33.866234
loss 2.9477034
STEP 36 ================================
prereg loss 2.85385 reg_l1 57.625256 reg_l2 33.887833
loss 2.9114752
STEP 37 ================================
prereg loss 2.8228512 reg_l1 57.647064 reg_l2 33.90896
loss 2.8804982
STEP 38 ================================
prereg loss 2.7970412 reg_l1 57.66815 reg_l2 33.929253
loss 2.8547094
STEP 39 ================================
prereg loss 2.7760541 reg_l1 57.688263 reg_l2 33.94848
loss 2.8337424
STEP 40 ================================
prereg loss 2.759274 reg_l1 57.707226 reg_l2 33.966396
loss 2.8169813
STEP 41 ================================
prereg loss 2.745926 reg_l1 57.724903 reg_l2 33.98286
loss 2.8036509
cutoff 0.24167418 network size 120
STEP 42 ================================
prereg loss 2.7351482 reg_l1 57.49951 reg_l2 33.93927
loss 2.7926476
STEP 43 ================================
prereg loss 2.726104 reg_l1 57.514347 reg_l2 33.952404
loss 2.7836185
STEP 44 ================================
prereg loss 2.7180533 reg_l1 57.527676 reg_l2 33.96383
loss 2.7755811
STEP 45 ================================
prereg loss 2.7103803 reg_l1 57.539555 reg_l2 33.97352
loss 2.7679198
STEP 46 ================================
prereg loss 2.7026398 reg_l1 57.54995 reg_l2 33.981552
loss 2.7601898
STEP 47 ================================
prereg loss 2.6945531 reg_l1 57.559006 reg_l2 33.98797
loss 2.7521122
STEP 48 ================================
prereg loss 2.6859982 reg_l1 57.566776 reg_l2 33.992912
loss 2.743565
STEP 49 ================================
prereg loss 2.6769793 reg_l1 57.573383 reg_l2 33.996502
loss 2.7345526
STEP 50 ================================
prereg loss 2.667604 reg_l1 57.57893 reg_l2 33.998875
loss 2.725183
STEP 51 ================================
prereg loss 2.658033 reg_l1 57.583584 reg_l2 34.00019
loss 2.7156165
cutoff 0.24409312 network size 119
STEP 52 ================================
prereg loss 2.648455 reg_l1 57.343353 reg_l2 33.941032
loss 2.7057981
STEP 53 ================================
prereg loss 2.63907 reg_l1 57.34659 reg_l2 33.94075
loss 2.6964166
STEP 54 ================================
prereg loss 2.6300402 reg_l1 57.3493 reg_l2 33.939873
loss 2.6873894
STEP 55 ================================
prereg loss 2.621497 reg_l1 57.35165 reg_l2 33.9386
loss 2.6788485
STEP 56 ================================
prereg loss 2.6135266 reg_l1 57.353737 reg_l2 33.93705
loss 2.6708803
STEP 57 ================================
prereg loss 2.606167 reg_l1 57.355682 reg_l2 33.935352
loss 2.6635227
STEP 58 ================================
prereg loss 2.5994093 reg_l1 57.35755 reg_l2 33.933617
loss 2.656767
STEP 59 ================================
prereg loss 2.5932178 reg_l1 57.359463 reg_l2 33.931946
loss 2.6505773
STEP 60 ================================
prereg loss 2.587513 reg_l1 57.361492 reg_l2 33.93043
loss 2.6448746
STEP 61 ================================
prereg loss 2.5822055 reg_l1 57.363632 reg_l2 33.929108
loss 2.6395693
cutoff 0.24504544 network size 118
STEP 62 ================================
prereg loss 2.5771956 reg_l1 57.120914 reg_l2 33.86801
loss 2.6343164
STEP 63 ================================
prereg loss 2.5723855 reg_l1 57.123486 reg_l2 33.867252
loss 2.629509
STEP 64 ================================
prereg loss 2.5676713 reg_l1 57.126297 reg_l2 33.866806
loss 2.6247976
STEP 65 ================================
prereg loss 2.562974 reg_l1 57.12936 reg_l2 33.866688
loss 2.6201034
STEP 66 ================================
prereg loss 2.5582228 reg_l1 57.132618 reg_l2 33.866848
loss 2.6153555
STEP 67 ================================
prereg loss 2.5533679 reg_l1 57.136116 reg_l2 33.867336
loss 2.610504
STEP 68 ================================
prereg loss 2.5483785 reg_l1 57.139816 reg_l2 33.868076
loss 2.6055183
STEP 69 ================================
prereg loss 2.5432334 reg_l1 57.143673 reg_l2 33.86907
loss 2.600377
STEP 70 ================================
prereg loss 2.537941 reg_l1 57.147675 reg_l2 33.87025
loss 2.5950887
STEP 71 ================================
prereg loss 2.5325067 reg_l1 57.151802 reg_l2 33.871613
loss 2.5896585
cutoff 0.24990079 network size 117
STEP 72 ================================
prereg loss 2.526962 reg_l1 56.906094 reg_l2 33.81066
loss 2.583868
STEP 73 ================================
prereg loss 2.5213246 reg_l1 56.910355 reg_l2 33.81224
loss 2.578235
STEP 74 ================================
prereg loss 2.515639 reg_l1 56.914627 reg_l2 33.813873
loss 2.5725536
STEP 75 ================================
prereg loss 2.5099359 reg_l1 56.918873 reg_l2 33.815502
loss 2.5668547
STEP 76 ================================
prereg loss 2.5042512 reg_l1 56.923073 reg_l2 33.817123
loss 2.5611744
STEP 77 ================================
prereg loss 2.4986043 reg_l1 56.927193 reg_l2 33.81866
loss 2.5555315
STEP 78 ================================
prereg loss 2.4930258 reg_l1 56.93121 reg_l2 33.820126
loss 2.549957
STEP 79 ================================
prereg loss 2.4873738 reg_l1 56.935127 reg_l2 33.82148
loss 2.544309
STEP 80 ================================
prereg loss 2.4816663 reg_l1 56.939114 reg_l2 33.822956
loss 2.5386055
STEP 81 ================================
prereg loss 2.4760294 reg_l1 56.943108 reg_l2 33.82449
loss 2.5329726
cutoff 0.25058737 network size 116
STEP 82 ================================
prereg loss 2.470471 reg_l1 56.696552 reg_l2 33.7633
loss 2.5271676
STEP 83 ================================
prereg loss 2.4650075 reg_l1 56.700577 reg_l2 33.76492
loss 2.521708
STEP 84 ================================
prereg loss 2.459631 reg_l1 56.704586 reg_l2 33.766556
loss 2.5163355
STEP 85 ================================
prereg loss 2.4543498 reg_l1 56.708508 reg_l2 33.768147
loss 2.5110583
STEP 86 ================================
prereg loss 2.4491549 reg_l1 56.712437 reg_l2 33.769707
loss 2.5058672
STEP 87 ================================
prereg loss 2.4440434 reg_l1 56.716248 reg_l2 33.77123
loss 2.5007596
STEP 88 ================================
prereg loss 2.4390006 reg_l1 56.72 reg_l2 33.772686
loss 2.4957206
STEP 89 ================================
prereg loss 2.4340363 reg_l1 56.723682 reg_l2 33.77409
loss 2.4907598
STEP 90 ================================
prereg loss 2.429138 reg_l1 56.727306 reg_l2 33.775425
loss 2.4858654
STEP 91 ================================
prereg loss 2.424302 reg_l1 56.730835 reg_l2 33.776722
loss 2.4810328
cutoff 0.2513612 network size 115
STEP 92 ================================
prereg loss 2.4195256 reg_l1 56.482967 reg_l2 33.71477
loss 2.4760087
STEP 93 ================================
prereg loss 2.4148085 reg_l1 56.48639 reg_l2 33.71598
loss 2.4712949
STEP 94 ================================
prereg loss 2.4101493 reg_l1 56.48979 reg_l2 33.71714
loss 2.466639
STEP 95 ================================
prereg loss 2.4055436 reg_l1 56.49313 reg_l2 33.718277
loss 2.4620366
STEP 96 ================================
prereg loss 2.400995 reg_l1 56.4964 reg_l2 33.71937
loss 2.4574914
STEP 97 ================================
prereg loss 2.3964953 reg_l1 56.499664 reg_l2 33.72046
loss 2.452995
STEP 98 ================================
prereg loss 2.392043 reg_l1 56.502903 reg_l2 33.721535
loss 2.448546
STEP 99 ================================
prereg loss 2.3876429 reg_l1 56.506084 reg_l2 33.722614
loss 2.444149
STEP 100 ================================
prereg loss 2.3832824 reg_l1 56.50926 reg_l2 33.723705
loss 2.4397917
2022-07-20T13:12:21.983

julia> serialize("cf-115-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-115-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 10)
2022-07-20T13:58:35.897
STEP 1 ================================
prereg loss 2.3789663 reg_l1 56.512436 reg_l2 33.724792
loss 2.4354787
cutoff 0.25358328 network size 114
STEP 2 ================================
prereg loss 2.3746827 reg_l1 56.262024 reg_l2 33.661594
loss 2.4309447
STEP 3 ================================
prereg loss 2.3704388 reg_l1 56.265224 reg_l2 33.662754
loss 2.426704
STEP 4 ================================
prereg loss 2.366216 reg_l1 56.268414 reg_l2 33.663914
loss 2.4224844
STEP 5 ================================
prereg loss 2.3620255 reg_l1 56.271587 reg_l2 33.665104
loss 2.418297
STEP 6 ================================
prereg loss 2.3578484 reg_l1 56.274776 reg_l2 33.66634
loss 2.4141233
STEP 7 ================================
prereg loss 2.3536973 reg_l1 56.277985 reg_l2 33.667606
loss 2.4099753
STEP 8 ================================
prereg loss 2.3495631 reg_l1 56.281204 reg_l2 33.668896
loss 2.4058442
STEP 9 ================================
prereg loss 2.3454409 reg_l1 56.284405 reg_l2 33.67022
loss 2.4017253
STEP 10 ================================
prereg loss 2.3413382 reg_l1 56.28763 reg_l2 33.671574
loss 2.3976257
STEP 11 ================================
prereg loss 2.3372457 reg_l1 56.290863 reg_l2 33.67298
loss 2.3935366
cutoff 0.25848943 network size 113
STEP 12 ================================
prereg loss 3.0997357 reg_l1 56.0356 reg_l2 33.60756
loss 3.1557713
STEP 13 ================================
prereg loss 3.0839243 reg_l1 56.036217 reg_l2 33.605843
loss 3.1399605
STEP 14 ================================
prereg loss 3.0579886 reg_l1 56.034397 reg_l2 33.601303
loss 3.114023
STEP 15 ================================
prereg loss 3.0248442 reg_l1 56.030495 reg_l2 33.5944
loss 3.0808747
STEP 16 ================================
prereg loss 2.987509 reg_l1 56.024967 reg_l2 33.585617
loss 3.043534
STEP 17 ================================
prereg loss 2.9487987 reg_l1 56.018166 reg_l2 33.575394
loss 3.0048168
STEP 18 ================================
prereg loss 2.911062 reg_l1 56.010506 reg_l2 33.564205
loss 2.9670725
STEP 19 ================================
prereg loss 2.876033 reg_l1 56.00238 reg_l2 33.55248
loss 2.9320354
STEP 20 ================================
prereg loss 2.8448095 reg_l1 55.9941 reg_l2 33.5406
loss 2.9008036
STEP 21 ================================
prereg loss 2.8178654 reg_l1 55.985954 reg_l2 33.528934
loss 2.8738513
cutoff 0.2718681 network size 112
STEP 22 ================================
prereg loss 4.2570243 reg_l1 55.70639 reg_l2 33.443893
loss 4.312731
STEP 23 ================================
prereg loss 4.0366144 reg_l1 55.69121 reg_l2 33.424206
loss 4.0923057
STEP 24 ================================
prereg loss 3.7712216 reg_l1 55.66982 reg_l2 33.397472
loss 3.8268914
STEP 25 ================================
prereg loss 3.4896352 reg_l1 55.64353 reg_l2 33.36524
loss 3.5452788
STEP 26 ================================
prereg loss 3.216276 reg_l1 55.61371 reg_l2 33.329075
loss 3.2718897
STEP 27 ================================
prereg loss 2.9673626 reg_l1 55.581604 reg_l2 33.290386
loss 3.0229442
STEP 28 ================================
prereg loss 2.7581193 reg_l1 55.548073 reg_l2 33.250137
loss 2.8136675
STEP 29 ================================
prereg loss 2.5952532 reg_l1 55.514225 reg_l2 33.209595
loss 2.6507676
STEP 30 ================================
prereg loss 2.479655 reg_l1 55.481052 reg_l2 33.169846
loss 2.535136
STEP 31 ================================
prereg loss 2.4079332 reg_l1 55.449356 reg_l2 33.131832
loss 2.4633825
cutoff 0.2733952 network size 111
STEP 32 ================================
prereg loss 2.3559976 reg_l1 55.146416 reg_l2 33.02155
loss 2.411144
STEP 33 ================================
prereg loss 2.364956 reg_l1 55.12041 reg_l2 32.98975
loss 2.4200764
STEP 34 ================================
prereg loss 2.3923275 reg_l1 55.09775 reg_l2 32.961773
loss 2.4474254
STEP 35 ================================
prereg loss 2.4297352 reg_l1 55.07869 reg_l2 32.93783
loss 2.484814
STEP 36 ================================
prereg loss 2.470007 reg_l1 55.063236 reg_l2 32.918026
loss 2.5250702
STEP 37 ================================
prereg loss 2.5070686 reg_l1 55.051422 reg_l2 32.902344
loss 2.56212
STEP 38 ================================
prereg loss 2.537403 reg_l1 55.04314 reg_l2 32.89063
loss 2.5924463
STEP 39 ================================
prereg loss 2.5583658 reg_l1 55.03824 reg_l2 32.882725
loss 2.613404
STEP 40 ================================
prereg loss 2.5686407 reg_l1 55.03652 reg_l2 32.878387
loss 2.6236773
STEP 41 ================================
prereg loss 2.568073 reg_l1 55.03772 reg_l2 32.877396
loss 2.6231108
cutoff 0.27601984 network size 110
STEP 42 ================================
prereg loss 2.8432438 reg_l1 54.765606 reg_l2 32.803215
loss 2.8980095
STEP 43 ================================
prereg loss 2.8103144 reg_l1 54.77287 reg_l2 32.809162
loss 2.8650873
STEP 44 ================================
prereg loss 2.7618382 reg_l1 54.78315 reg_l2 32.818565
loss 2.8166213
STEP 45 ================================
prereg loss 2.7024314 reg_l1 54.795918 reg_l2 32.830868
loss 2.7572274
STEP 46 ================================
prereg loss 2.6366093 reg_l1 54.81075 reg_l2 32.84556
loss 2.69142
STEP 47 ================================
prereg loss 2.568905 reg_l1 54.827156 reg_l2 32.862125
loss 2.6237323
STEP 48 ================================
prereg loss 2.5033782 reg_l1 54.84472 reg_l2 32.88003
loss 2.5582228
STEP 49 ================================
prereg loss 2.4433177 reg_l1 54.862972 reg_l2 32.89876
loss 2.4981806
STEP 50 ================================
prereg loss 2.3911386 reg_l1 54.881535 reg_l2 32.91784
loss 2.4460201
STEP 51 ================================
prereg loss 2.3483148 reg_l1 54.89995 reg_l2 32.936794
loss 2.4032147
cutoff 0.28014857 network size 109
STEP 52 ================================
prereg loss 2.3154016 reg_l1 54.637726 reg_l2 32.87675
loss 2.3700392
STEP 53 ================================
prereg loss 2.2921348 reg_l1 54.654842 reg_l2 32.894264
loss 2.3467896
STEP 54 ================================
prereg loss 2.2775657 reg_l1 54.670902 reg_l2 32.910545
loss 2.3322365
STEP 55 ================================
prereg loss 2.2702363 reg_l1 54.685635 reg_l2 32.925323
loss 2.3249218
STEP 56 ================================
prereg loss 2.2683125 reg_l1 54.698875 reg_l2 32.938393
loss 2.3230114
STEP 57 ================================
prereg loss 2.2698052 reg_l1 54.710526 reg_l2 32.949615
loss 2.3245158
STEP 58 ================================
prereg loss 2.2731981 reg_l1 54.72047 reg_l2 32.958897
loss 2.3279185
STEP 59 ================================
prereg loss 2.2769227 reg_l1 54.72872 reg_l2 32.966167
loss 2.3316514
STEP 60 ================================
prereg loss 2.279739 reg_l1 54.735218 reg_l2 32.971466
loss 2.334474
STEP 61 ================================
prereg loss 2.2807636 reg_l1 54.740067 reg_l2 32.974804
loss 2.3355036
cutoff 0.28035703 network size 108
STEP 62 ================================
prereg loss 2.2795181 reg_l1 54.46296 reg_l2 32.89773
loss 2.333981
STEP 63 ================================
prereg loss 2.2758613 reg_l1 54.464752 reg_l2 32.897526
loss 2.330326
STEP 64 ================================
prereg loss 2.2699766 reg_l1 54.465195 reg_l2 32.89579
loss 2.324442
STEP 65 ================================
prereg loss 2.2622519 reg_l1 54.46443 reg_l2 32.892666
loss 2.3167162
STEP 66 ================================
prereg loss 2.2532177 reg_l1 54.462685 reg_l2 32.888386
loss 2.3076804
STEP 67 ================================
prereg loss 2.2434623 reg_l1 54.460083 reg_l2 32.883133
loss 2.2979224
STEP 68 ================================
prereg loss 2.2335517 reg_l1 54.45683 reg_l2 32.877144
loss 2.2880087
STEP 69 ================================
prereg loss 2.2239878 reg_l1 54.453106 reg_l2 32.87063
loss 2.278441
STEP 70 ================================
prereg loss 2.2151542 reg_l1 54.449066 reg_l2 32.863766
loss 2.2696033
STEP 71 ================================
prereg loss 2.2073221 reg_l1 54.444912 reg_l2 32.85675
loss 2.2617671
cutoff 0.28687334 network size 107
STEP 72 ================================
prereg loss 2.2006137 reg_l1 54.153866 reg_l2 32.767452
loss 2.2547677
STEP 73 ================================
prereg loss 2.1950397 reg_l1 54.149834 reg_l2 32.760647
loss 2.2491896
STEP 74 ================================
prereg loss 2.1905181 reg_l1 54.146072 reg_l2 32.754143
loss 2.2446642
STEP 75 ================================
prereg loss 2.1868834 reg_l1 54.142666 reg_l2 32.74807
loss 2.2410262
STEP 76 ================================
prereg loss 2.183929 reg_l1 54.139645 reg_l2 32.742504
loss 2.2380686
STEP 77 ================================
prereg loss 2.1813993 reg_l1 54.13715 reg_l2 32.73751
loss 2.2355366
STEP 78 ================================
prereg loss 2.179085 reg_l1 54.135143 reg_l2 32.73311
loss 2.23322
STEP 79 ================================
prereg loss 2.1768343 reg_l1 54.133633 reg_l2 32.729317
loss 2.230968
STEP 80 ================================
prereg loss 2.174477 reg_l1 54.132683 reg_l2 32.72617
loss 2.2286098
STEP 81 ================================
prereg loss 2.1718934 reg_l1 54.13225 reg_l2 32.723648
loss 2.2260256
cutoff 0.2914589 network size 106
STEP 82 ================================
prereg loss 31.18395 reg_l1 53.840866 reg_l2 32.63678
loss 31.237791
STEP 83 ================================
prereg loss 30.109858 reg_l1 53.837616 reg_l2 32.633293
loss 30.163694
STEP 84 ================================
prereg loss 28.18814 reg_l1 53.831867 reg_l2 32.628944
loss 28.241972
STEP 85 ================================
prereg loss 25.654957 reg_l1 53.824455 reg_l2 32.6244
loss 25.708782
STEP 86 ================================
prereg loss 22.742434 reg_l1 53.816166 reg_l2 32.620285
loss 22.79625
STEP 87 ================================
prereg loss 19.66481 reg_l1 53.807632 reg_l2 32.61714
loss 19.718618
STEP 88 ================================
prereg loss 16.609629 reg_l1 53.799427 reg_l2 32.615395
loss 16.663427
STEP 89 ================================
prereg loss 13.732717 reg_l1 53.792065 reg_l2 32.615345
loss 13.786509
STEP 90 ================================
prereg loss 11.1547985 reg_l1 53.78589 reg_l2 32.61724
loss 11.208585
STEP 91 ================================
prereg loss 8.960571 reg_l1 53.78129 reg_l2 32.62119
loss 9.014353
cutoff 0.28800133 network size 105
STEP 92 ================================
prereg loss 6.370114 reg_l1 53.49047 reg_l2 32.544285
loss 6.4236045
STEP 93 ================================
prereg loss 5.4738975 reg_l1 53.49322 reg_l2 32.554367
loss 5.5273905
STEP 94 ================================
prereg loss 4.9835367 reg_l1 53.498238 reg_l2 32.566357
loss 5.037035
STEP 95 ================================
prereg loss 4.825301 reg_l1 53.505188 reg_l2 32.579727
loss 4.8788066
STEP 96 ================================
prereg loss 4.919296 reg_l1 53.514072 reg_l2 32.594265
loss 4.97281
STEP 97 ================================
prereg loss 5.1827564 reg_l1 53.524765 reg_l2 32.60972
loss 5.2362814
STEP 98 ================================
prereg loss 5.535289 reg_l1 53.537155 reg_l2 32.625824
loss 5.588826
STEP 99 ================================
prereg loss 5.904996 reg_l1 53.55103 reg_l2 32.64226
loss 5.958547
STEP 100 ================================
prereg loss 6.233285 reg_l1 53.56618 reg_l2 32.65879
loss 6.286851
2022-07-20T14:06:58.600
```

Let's switch to 5 sparsifications per 100 steps:

```
julia> interleaving_steps!(100, 20)
2022-07-20T14:37:57.046
STEP 1 ================================
prereg loss 6.478077 reg_l1 53.582386 reg_l2 32.675148
loss 6.531659
cutoff 0.2781881 network size 104
STEP 2 ================================
prereg loss 6.7741404 reg_l1 53.321186 reg_l2 32.613735
loss 6.8274617
STEP 3 ================================
prereg loss 6.7938576 reg_l1 53.33832 reg_l2 32.628815
loss 6.847196
STEP 4 ================================
prereg loss 6.698625 reg_l1 53.355446 reg_l2 32.64287
loss 6.7519803
STEP 5 ================================
prereg loss 6.506171 reg_l1 53.37235 reg_l2 32.65579
loss 6.5595436
STEP 6 ================================
prereg loss 6.2412896 reg_l1 53.388847 reg_l2 32.667496
loss 6.2946787
STEP 7 ================================
prereg loss 5.931747 reg_l1 53.40475 reg_l2 32.677906
loss 5.985152
STEP 8 ================================
prereg loss 5.604977 reg_l1 53.41991 reg_l2 32.68697
loss 5.658397
STEP 9 ================================
prereg loss 5.2855244 reg_l1 53.434204 reg_l2 32.69471
loss 5.3389587
STEP 10 ================================
prereg loss 4.9931736 reg_l1 53.447506 reg_l2 32.70111
loss 5.0466213
STEP 11 ================================
prereg loss 4.742061 reg_l1 53.459805 reg_l2 32.70619
loss 4.795521
STEP 12 ================================
prereg loss 4.540517 reg_l1 53.47101 reg_l2 32.71001
loss 4.593988
STEP 13 ================================
prereg loss 4.391276 reg_l1 53.481094 reg_l2 32.712616
loss 4.444757
STEP 14 ================================
prereg loss 4.292468 reg_l1 53.490078 reg_l2 32.71409
loss 4.345958
STEP 15 ================================
prereg loss 4.238496 reg_l1 53.49798 reg_l2 32.714508
loss 4.2919936
STEP 16 ================================
prereg loss 4.221335 reg_l1 53.50482 reg_l2 32.714
loss 4.27484
STEP 17 ================================
prereg loss 4.2315106 reg_l1 53.5107 reg_l2 32.71265
loss 4.2850213
STEP 18 ================================
prereg loss 4.2592916 reg_l1 53.51568 reg_l2 32.7106
loss 4.3128076
STEP 19 ================================
prereg loss 4.2954435 reg_l1 53.519875 reg_l2 32.707993
loss 4.3489633
STEP 20 ================================
prereg loss 4.331817 reg_l1 53.52338 reg_l2 32.70495
loss 4.3853407
STEP 21 ================================
prereg loss 4.3618984 reg_l1 53.526318 reg_l2 32.701603
loss 4.415425
cutoff 0.29380295 network size 103
STEP 22 ================================
prereg loss 4.380904 reg_l1 53.235043 reg_l2 32.611797
loss 4.4341393
STEP 23 ================================
prereg loss 4.3858986 reg_l1 53.237236 reg_l2 32.60829
loss 4.439136
STEP 24 ================================
prereg loss 4.3756313 reg_l1 53.239273 reg_l2 32.604916
loss 4.4288707
STEP 25 ================================
prereg loss 4.350282 reg_l1 53.241257 reg_l2 32.60179
loss 4.4035234
STEP 26 ================================
prereg loss 4.3112183 reg_l1 53.243317 reg_l2 32.599037
loss 4.3644614
STEP 27 ================================
prereg loss 4.2606525 reg_l1 53.24554 reg_l2 32.59675
loss 4.313898
STEP 28 ================================
prereg loss 4.2012744 reg_l1 53.24806 reg_l2 32.595028
loss 4.2545223
STEP 29 ================================
prereg loss 4.13601 reg_l1 53.250942 reg_l2 32.593952
loss 4.189261
STEP 30 ================================
prereg loss 4.0676775 reg_l1 53.254284 reg_l2 32.593563
loss 4.1209316
STEP 31 ================================
prereg loss 3.9988577 reg_l1 53.258137 reg_l2 32.593906
loss 4.052116
STEP 32 ================================
prereg loss 3.9316702 reg_l1 53.262516 reg_l2 32.59499
loss 3.9849327
STEP 33 ================================
prereg loss 3.8677824 reg_l1 53.26751 reg_l2 32.596844
loss 3.9210498
STEP 34 ================================
prereg loss 3.8082976 reg_l1 53.273083 reg_l2 32.599438
loss 3.8615708
STEP 35 ================================
prereg loss 3.7538133 reg_l1 53.279232 reg_l2 32.602745
loss 3.8070924
STEP 36 ================================
prereg loss 3.7044494 reg_l1 53.285957 reg_l2 32.606743
loss 3.7577355
STEP 37 ================================
prereg loss 3.6600008 reg_l1 53.29324 reg_l2 32.61135
loss 3.713294
STEP 38 ================================
prereg loss 3.6199584 reg_l1 53.301 reg_l2 32.61651
loss 3.6732595
STEP 39 ================================
prereg loss 3.5836706 reg_l1 53.309227 reg_l2 32.622147
loss 3.6369798
STEP 40 ================================
prereg loss 3.5503497 reg_l1 53.31784 reg_l2 32.62818
loss 3.6036675
STEP 41 ================================
prereg loss 3.5193136 reg_l1 53.326775 reg_l2 32.634537
loss 3.5726404
cutoff 0.29600754 network size 102
STEP 42 ================================
prereg loss 3.4898658 reg_l1 53.03994 reg_l2 32.553486
loss 3.5429058
STEP 43 ================================
prereg loss 3.4614644 reg_l1 53.049305 reg_l2 32.560204
loss 3.5145137
STEP 44 ================================
prereg loss 3.4337173 reg_l1 53.05876 reg_l2 32.56696
loss 3.486776
STEP 45 ================================
prereg loss 3.40637 reg_l1 53.068268 reg_l2 32.57368
loss 3.459438
STEP 46 ================================
prereg loss 3.3792574 reg_l1 53.077713 reg_l2 32.58028
loss 3.4323351
STEP 47 ================================
prereg loss 3.3523474 reg_l1 53.087032 reg_l2 32.586693
loss 3.4054344
STEP 48 ================================
prereg loss 3.3257473 reg_l1 53.096176 reg_l2 32.592857
loss 3.3788435
STEP 49 ================================
prereg loss 3.2995389 reg_l1 53.105106 reg_l2 32.5987
loss 3.352644
STEP 50 ================================
prereg loss 3.2738466 reg_l1 53.113724 reg_l2 32.604187
loss 3.3269603
STEP 51 ================================
prereg loss 3.2487636 reg_l1 53.122036 reg_l2 32.60929
loss 3.3018856
STEP 52 ================================
prereg loss 3.224444 reg_l1 53.12999 reg_l2 32.613976
loss 3.2775738
STEP 53 ================================
prereg loss 3.2009196 reg_l1 53.137554 reg_l2 32.618214
loss 3.2540572
STEP 54 ================================
prereg loss 3.178187 reg_l1 53.14474 reg_l2 32.622005
loss 3.2313316
STEP 55 ================================
prereg loss 3.156186 reg_l1 53.15156 reg_l2 32.625362
loss 3.2093377
STEP 56 ================================
prereg loss 3.1348612 reg_l1 53.157944 reg_l2 32.628292
loss 3.1880193
STEP 57 ================================
prereg loss 3.114104 reg_l1 53.163982 reg_l2 32.630836
loss 3.167268
STEP 58 ================================
prereg loss 3.0937731 reg_l1 53.169674 reg_l2 32.632984
loss 3.1469429
STEP 59 ================================
prereg loss 3.0737348 reg_l1 53.174984 reg_l2 32.634804
loss 3.1269097
STEP 60 ================================
prereg loss 3.0538175 reg_l1 53.18002 reg_l2 32.63633
loss 3.1069975
STEP 61 ================================
prereg loss 3.0339236 reg_l1 53.184784 reg_l2 32.637592
loss 3.0871084
cutoff 0.30534485 network size 101
STEP 62 ================================
prereg loss 3.0139027 reg_l1 52.883965 reg_l2 32.545403
loss 3.0667865
STEP 63 ================================
prereg loss 2.9937034 reg_l1 52.88834 reg_l2 32.54632
loss 3.0465918
STEP 64 ================================
prereg loss 2.9732444 reg_l1 52.89253 reg_l2 32.54711
loss 3.0261369
STEP 65 ================================
prereg loss 2.952482 reg_l1 52.896645 reg_l2 32.547844
loss 3.0053787
STEP 66 ================================
prereg loss 2.931415 reg_l1 52.900692 reg_l2 32.548553
loss 2.9843159
STEP 67 ================================
prereg loss 2.9100578 reg_l1 52.904686 reg_l2 32.549274
loss 2.9629624
STEP 68 ================================
prereg loss 2.8884144 reg_l1 52.90869 reg_l2 32.55007
loss 2.941323
STEP 69 ================================
prereg loss 2.866557 reg_l1 52.912727 reg_l2 32.550964
loss 2.9194696
STEP 70 ================================
prereg loss 2.844512 reg_l1 52.91684 reg_l2 32.551983
loss 2.8974288
STEP 71 ================================
prereg loss 2.822359 reg_l1 52.921036 reg_l2 32.55314
loss 2.8752801
STEP 72 ================================
prereg loss 2.8001308 reg_l1 52.925316 reg_l2 32.554474
loss 2.8530562
STEP 73 ================================
prereg loss 2.777868 reg_l1 52.929756 reg_l2 32.555996
loss 2.8307977
STEP 74 ================================
prereg loss 2.755638 reg_l1 52.93432 reg_l2 32.557724
loss 2.8085723
STEP 75 ================================
prereg loss 2.7334886 reg_l1 52.939014 reg_l2 32.559628
loss 2.7864275
STEP 76 ================================
prereg loss 2.7114673 reg_l1 52.94385 reg_l2 32.561737
loss 2.7644112
STEP 77 ================================
prereg loss 2.6895607 reg_l1 52.94883 reg_l2 32.564045
loss 2.7425094
STEP 78 ================================
prereg loss 2.6678388 reg_l1 52.953964 reg_l2 32.56652
loss 2.7207928
STEP 79 ================================
prereg loss 2.6463137 reg_l1 52.959213 reg_l2 32.56918
loss 2.6992729
STEP 80 ================================
prereg loss 2.624995 reg_l1 52.964573 reg_l2 32.571995
loss 2.6779597
STEP 81 ================================
prereg loss 2.6038916 reg_l1 52.97003 reg_l2 32.57494
loss 2.6568615
cutoff 0.30692953 network size 100
STEP 82 ================================
prereg loss 2.5830288 reg_l1 52.66866 reg_l2 32.4838
loss 2.6356974
STEP 83 ================================
prereg loss 2.5624146 reg_l1 52.674397 reg_l2 32.487022
loss 2.615089
STEP 84 ================================
prereg loss 2.542031 reg_l1 52.68019 reg_l2 32.49033
loss 2.5947113
STEP 85 ================================
prereg loss 2.5219169 reg_l1 52.686012 reg_l2 32.49369
loss 2.5746028
STEP 86 ================================
prereg loss 2.502257 reg_l1 52.691837 reg_l2 32.497074
loss 2.554949
STEP 87 ================================
prereg loss 2.4853485 reg_l1 52.697536 reg_l2 32.500343
loss 2.538046
STEP 88 ================================
prereg loss 2.4686697 reg_l1 52.703094 reg_l2 32.503498
loss 2.5213728
STEP 89 ================================
prereg loss 2.4522223 reg_l1 52.708492 reg_l2 32.50651
loss 2.5049307
STEP 90 ================================
prereg loss 2.4359748 reg_l1 52.71374 reg_l2 32.509384
loss 2.4886885
STEP 91 ================================
prereg loss 2.4199271 reg_l1 52.718838 reg_l2 32.512108
loss 2.472646
STEP 92 ================================
prereg loss 2.4040468 reg_l1 52.723785 reg_l2 32.51468
loss 2.4567707
STEP 93 ================================
prereg loss 2.3883224 reg_l1 52.728565 reg_l2 32.517117
loss 2.441051
STEP 94 ================================
prereg loss 2.3727543 reg_l1 52.733196 reg_l2 32.51941
loss 2.4254875
STEP 95 ================================
prereg loss 2.3573062 reg_l1 52.73769 reg_l2 32.521564
loss 2.410044
STEP 96 ================================
prereg loss 2.342 reg_l1 52.74206 reg_l2 32.523598
loss 2.394742
STEP 97 ================================
prereg loss 2.326803 reg_l1 52.746292 reg_l2 32.525536
loss 2.3795493
STEP 98 ================================
prereg loss 2.311706 reg_l1 52.750427 reg_l2 32.527363
loss 2.3644564
STEP 99 ================================
prereg loss 2.2967157 reg_l1 52.754482 reg_l2 32.529114
loss 2.3494701
STEP 100 ================================
prereg loss 2.2818089 reg_l1 52.758442 reg_l2 32.53078
loss 2.3345673
2022-07-20T14:45:27.660

julia> serialize("cf-100-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-100-parameters-opt.ser", opt)

julia> # I am tired of cutting large weights, let's actually use a larger L1
```

`loss` function and all that depends on it needs to be re-entered:

```
julia> function loss(dmm_lite::DMM_Lite_)
           l = 0.0f0
           for i in 1:140 # 1:12 # 1:35 ok speed (use 0.001 reg mult),
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
loss (generic function with 1 method)

julia> function training_step!()
           reset_dicts!()
           adam_step!(opt, trainable["network_matrix"],
                      convert(Dict{String, Dict{String, Dict{String, Dict{String, Float32}}}},
                              gradient(loss, trainable)[1]["network_matrix"]))
       end
training_step! (generic function with 1 method)

julia> function steps!(n_steps)
           printlog_v(io, now())
           for i in 1:n_steps
               printlog_v(io, "STEP ", i, " ================================")
               training_step!()
           end
           printlog_v(io, now())
       end
steps! (generic function with 1 method)

julia>

julia> function sparsifying_steps!(n_steps)
           printlog_v(io, now())
           for i in 1:n_steps
               printlog_v(io, "STEP ", i, " ================================")
               training_step!()
               lim = min_abs_dict(trainable["network_matrix"])
               trim_network(trainable, opt, lim)
               printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
           end
           printlog_v(io, now())
       end
sparsifying_steps! (generic function with 1 method)

julia>

julia> # one sparsifying step out of N

julia> function interleaving_steps!(n_steps, N=2)
           printlog_v(io, now())
           for i in 1:n_steps
               printlog_v(io, "STEP ", i, " ================================")
               training_step!()
               if i%N == 1
                   lim = min_abs_dict(trainable["network_matrix"])
                   trim_network(trainable, opt, lim)
                   printlog_v(io, "cutoff ", lim, " network size ", count(trainable["network_matrix"]))
               end
           end
           printlog_v(io, now())
       end
interleaving_steps! (generic function with 2 methods)

julia> steps!(100)
2022-07-20T14:51:29.212
STEP 1 ================================
prereg loss 2.2669868 reg_l1 52.762333 reg_l2 32.53241
loss 7.54322
STEP 2 ================================
prereg loss 2.252442 reg_l1 52.759144 reg_l2 32.52532
loss 7.5283566
STEP 3 ================================
prereg loss 2.2381465 reg_l1 52.750195 reg_l2 32.51162
loss 7.5131664
STEP 4 ================================
prereg loss 2.2240806 reg_l1 52.736393 reg_l2 32.492638
loss 7.49772
STEP 5 ================================
prereg loss 2.210227 reg_l1 52.718475 reg_l2 32.469315
loss 7.4820747
STEP 6 ================================
prereg loss 2.1965709 reg_l1 52.696995 reg_l2 32.442413
loss 7.4662704
STEP 7 ================================
prereg loss 2.183098 reg_l1 52.672493 reg_l2 32.41255
loss 7.450348
STEP 8 ================================
prereg loss 2.1697752 reg_l1 52.645382 reg_l2 32.380222
loss 7.434314
STEP 9 ================================
prereg loss 2.1565845 reg_l1 52.61603 reg_l2 32.34585
loss 7.418188
STEP 10 ================================
prereg loss 2.1435182 reg_l1 52.584736 reg_l2 32.309822
loss 7.401992
STEP 11 ================================
prereg loss 2.130541 reg_l1 52.551743 reg_l2 32.272415
loss 7.3857155
STEP 12 ================================
prereg loss 2.117649 reg_l1 52.51733 reg_l2 32.233902
loss 7.3693824
STEP 13 ================================
prereg loss 2.104839 reg_l1 52.48168 reg_l2 32.194527
loss 7.3530073
STEP 14 ================================
prereg loss 2.0920897 reg_l1 52.444992 reg_l2 32.15446
loss 7.336589
STEP 15 ================================
prereg loss 2.0793972 reg_l1 52.407394 reg_l2 32.113873
loss 7.3201365
STEP 16 ================================
prereg loss 2.0667577 reg_l1 52.369038 reg_l2 32.072906
loss 7.3036613
STEP 17 ================================
prereg loss 2.0541787 reg_l1 52.33004 reg_l2 32.031696
loss 7.287183
STEP 18 ================================
prereg loss 2.0416367 reg_l1 52.29049 reg_l2 31.990324
loss 7.270686
STEP 19 ================================
prereg loss 2.0291562 reg_l1 52.250526 reg_l2 31.948895
loss 7.254209
STEP 20 ================================
prereg loss 2.0167289 reg_l1 52.21017 reg_l2 31.907475
loss 7.2377462
STEP 21 ================================
prereg loss 2.0043628 reg_l1 52.169506 reg_l2 31.86613
loss 7.2213135
STEP 22 ================================
prereg loss 1.9920664 reg_l1 52.1286 reg_l2 31.824911
loss 7.2049265
STEP 23 ================================
prereg loss 1.9798412 reg_l1 52.087475 reg_l2 31.783855
loss 7.1885886
STEP 24 ================================
prereg loss 1.9676908 reg_l1 52.04622 reg_l2 31.743006
loss 7.1723127
STEP 25 ================================
prereg loss 1.9556205 reg_l1 52.00485 reg_l2 31.702387
loss 7.156105
STEP 26 ================================
prereg loss 1.9436575 reg_l1 51.963352 reg_l2 31.662024
loss 7.1399927
STEP 27 ================================
prereg loss 1.9317759 reg_l1 51.921844 reg_l2 31.62194
loss 7.1239605
STEP 28 ================================
prereg loss 1.9199898 reg_l1 51.880272 reg_l2 31.582138
loss 7.108017
STEP 29 ================================
prereg loss 1.908302 reg_l1 51.838688 reg_l2 31.542646
loss 7.0921707
STEP 30 ================================
prereg loss 1.8967305 reg_l1 51.797127 reg_l2 31.503462
loss 7.076443
STEP 31 ================================
prereg loss 1.8852717 reg_l1 51.75557 reg_l2 31.464617
loss 7.0608287
STEP 32 ================================
prereg loss 1.8739127 reg_l1 51.714027 reg_l2 31.426094
loss 7.0453157
STEP 33 ================================
prereg loss 1.8626667 reg_l1 51.672546 reg_l2 31.387892
loss 7.0299215
STEP 34 ================================
prereg loss 1.8515333 reg_l1 51.63113 reg_l2 31.350035
loss 7.0146465
STEP 35 ================================
prereg loss 1.840505 reg_l1 51.589764 reg_l2 31.31252
loss 6.9994817
STEP 36 ================================
prereg loss 1.8295819 reg_l1 51.548485 reg_l2 31.275358
loss 6.9844303
STEP 37 ================================
prereg loss 1.8187823 reg_l1 51.50727 reg_l2 31.238531
loss 6.9695096
STEP 38 ================================
prereg loss 1.8080759 reg_l1 51.46616 reg_l2 31.202053
loss 6.954692
STEP 39 ================================
prereg loss 1.7974744 reg_l1 51.425137 reg_l2 31.165941
loss 6.939988
STEP 40 ================================
prereg loss 1.7869776 reg_l1 51.38422 reg_l2 31.130165
loss 6.9254
STEP 41 ================================
prereg loss 1.7765825 reg_l1 51.343426 reg_l2 31.094738
loss 6.910925
STEP 42 ================================
prereg loss 1.7662921 reg_l1 51.302708 reg_l2 31.059671
loss 6.896563
STEP 43 ================================
prereg loss 1.7560914 reg_l1 51.26215 reg_l2 31.024954
loss 6.882306
STEP 44 ================================
prereg loss 1.7459862 reg_l1 51.221714 reg_l2 30.990599
loss 6.8681574
STEP 45 ================================
prereg loss 1.7359605 reg_l1 51.18138 reg_l2 30.956594
loss 6.854099
STEP 46 ================================
prereg loss 1.7260298 reg_l1 51.141182 reg_l2 30.922943
loss 6.840148
STEP 47 ================================
prereg loss 1.7161888 reg_l1 51.10112 reg_l2 30.88965
loss 6.826301
STEP 48 ================================
prereg loss 1.7064345 reg_l1 51.0612 reg_l2 30.856703
loss 6.8125544
STEP 49 ================================
prereg loss 1.6967486 reg_l1 51.021427 reg_l2 30.824106
loss 6.7988915
STEP 50 ================================
prereg loss 1.6871485 reg_l1 50.98177 reg_l2 30.79187
loss 6.7853255
STEP 51 ================================
prereg loss 1.6776239 reg_l1 50.942272 reg_l2 30.759962
loss 6.771851
STEP 52 ================================
prereg loss 1.6681813 reg_l1 50.9029 reg_l2 30.728422
loss 6.7584715
STEP 53 ================================
prereg loss 1.6588099 reg_l1 50.863697 reg_l2 30.697216
loss 6.74518
STEP 54 ================================
prereg loss 1.6495088 reg_l1 50.82461 reg_l2 30.666355
loss 6.7319703
STEP 55 ================================
prereg loss 1.6402837 reg_l1 50.785706 reg_l2 30.635817
loss 6.7188544
STEP 56 ================================
prereg loss 1.6311232 reg_l1 50.746902 reg_l2 30.605629
loss 6.7058134
STEP 57 ================================
prereg loss 1.6220366 reg_l1 50.708256 reg_l2 30.575771
loss 6.692862
STEP 58 ================================
prereg loss 1.6130117 reg_l1 50.66975 reg_l2 30.546236
loss 6.679987
STEP 59 ================================
prereg loss 1.6040719 reg_l1 50.631374 reg_l2 30.517015
loss 6.6672096
STEP 60 ================================
prereg loss 1.595197 reg_l1 50.593147 reg_l2 30.488106
loss 6.6545115
STEP 61 ================================
prereg loss 1.586388 reg_l1 50.55503 reg_l2 30.459517
loss 6.6418915
STEP 62 ================================
prereg loss 1.5776441 reg_l1 50.517056 reg_l2 30.431234
loss 6.6293497
STEP 63 ================================
prereg loss 1.5689766 reg_l1 50.479206 reg_l2 30.403263
loss 6.6168976
STEP 64 ================================
prereg loss 1.5603726 reg_l1 50.441517 reg_l2 30.375572
loss 6.6045246
STEP 65 ================================
prereg loss 1.5518495 reg_l1 50.403927 reg_l2 30.348188
loss 6.5922422
STEP 66 ================================
prereg loss 1.5433899 reg_l1 50.36645 reg_l2 30.321087
loss 6.580035
STEP 67 ================================
prereg loss 1.5350043 reg_l1 50.329105 reg_l2 30.294268
loss 6.567915
STEP 68 ================================
prereg loss 1.5266831 reg_l1 50.291862 reg_l2 30.267723
loss 6.555869
STEP 69 ================================
prereg loss 1.518436 reg_l1 50.254753 reg_l2 30.241463
loss 6.5439115
STEP 70 ================================
prereg loss 1.5102594 reg_l1 50.217754 reg_l2 30.21547
loss 6.532035
STEP 71 ================================
prereg loss 1.5021485 reg_l1 50.18086 reg_l2 30.189734
loss 6.5202346
STEP 72 ================================
prereg loss 1.4941155 reg_l1 50.14406 reg_l2 30.16426
loss 6.508521
STEP 73 ================================
prereg loss 1.4861484 reg_l1 50.107384 reg_l2 30.139048
loss 6.4968867
STEP 74 ================================
prereg loss 1.4782515 reg_l1 50.07079 reg_l2 30.114098
loss 6.4853306
STEP 75 ================================
prereg loss 1.4704223 reg_l1 50.034313 reg_l2 30.089388
loss 6.4738536
STEP 76 ================================
prereg loss 1.4626676 reg_l1 49.997948 reg_l2 30.064924
loss 6.4624624
STEP 77 ================================
prereg loss 1.4549882 reg_l1 49.961662 reg_l2 30.040707
loss 6.4511547
STEP 78 ================================
prereg loss 1.4473698 reg_l1 49.925472 reg_l2 30.016718
loss 6.4399176
STEP 79 ================================
prereg loss 1.4398223 reg_l1 49.889378 reg_l2 29.992985
loss 6.42876
STEP 80 ================================
prereg loss 1.4323422 reg_l1 49.853394 reg_l2 29.969463
loss 6.4176817
STEP 81 ================================
prereg loss 1.4249297 reg_l1 49.81747 reg_l2 29.946196
loss 6.406677
STEP 82 ================================
prereg loss 1.417584 reg_l1 49.78166 reg_l2 29.923143
loss 6.39575
STEP 83 ================================
prereg loss 1.410303 reg_l1 49.74593 reg_l2 29.900322
loss 6.3848963
STEP 84 ================================
prereg loss 1.4030933 reg_l1 49.710285 reg_l2 29.877722
loss 6.374122
STEP 85 ================================
prereg loss 1.3959446 reg_l1 49.674732 reg_l2 29.855345
loss 6.363418
STEP 86 ================================
prereg loss 1.388859 reg_l1 49.63927 reg_l2 29.833187
loss 6.352786
STEP 87 ================================
prereg loss 1.3818431 reg_l1 49.60387 reg_l2 29.811243
loss 6.3422303
STEP 88 ================================
prereg loss 1.3748833 reg_l1 49.568577 reg_l2 29.789522
loss 6.331741
STEP 89 ================================
prereg loss 1.3679906 reg_l1 49.533367 reg_l2 29.768005
loss 6.321327
STEP 90 ================================
prereg loss 1.3611573 reg_l1 49.498215 reg_l2 29.746695
loss 6.310979
STEP 91 ================================
prereg loss 1.3543856 reg_l1 49.46796 reg_l2 29.725605
loss 6.301182
STEP 92 ================================
prereg loss 1.347677 reg_l1 49.438744 reg_l2 29.704708
loss 6.2915516
STEP 93 ================================
prereg loss 1.3410262 reg_l1 49.409016 reg_l2 29.684
loss 6.281928
STEP 94 ================================
prereg loss 1.3344362 reg_l1 49.37882 reg_l2 29.663473
loss 6.272318
STEP 95 ================================
prereg loss 1.3279052 reg_l1 49.34823 reg_l2 29.643122
loss 6.262728
STEP 96 ================================
prereg loss 1.3214372 reg_l1 49.31727 reg_l2 29.622944
loss 6.2531643
STEP 97 ================================
prereg loss 1.3150208 reg_l1 49.28601 reg_l2 29.60293
loss 6.243622
STEP 98 ================================
prereg loss 1.3086663 reg_l1 49.254494 reg_l2 29.583096
loss 6.2341156
STEP 99 ================================
prereg loss 1.3023713 reg_l1 49.222694 reg_l2 29.563414
loss 6.224641
STEP 100 ================================
prereg loss 1.2961274 reg_l1 49.190727 reg_l2 29.54392
loss 6.2152
2022-07-20T14:58:48.607
```

Now let's return to 5 sparsifications per 100 steps mode:

```
julia> interleaving_steps!(100, 20)
2022-07-20T14:59:03.587
STEP 1 ================================
prereg loss 1.2899444 reg_l1 49.15855 reg_l2 29.524565
loss 6.205799
cutoff 0.0027290583 network size 99
STEP 2 ================================
prereg loss 1.2838135 reg_l1 49.12897 reg_l2 29.50539
loss 6.1967106
STEP 3 ================================
prereg loss 1.2777447 reg_l1 49.099247 reg_l2 29.486362
loss 6.1876698
STEP 4 ================================
prereg loss 1.2717249 reg_l1 49.069416 reg_l2 29.467497
loss 6.178667
STEP 5 ================================
prereg loss 1.2657627 reg_l1 49.039482 reg_l2 29.448765
loss 6.169711
STEP 6 ================================
prereg loss 1.2598546 reg_l1 49.00944 reg_l2 29.430187
loss 6.160799
STEP 7 ================================
prereg loss 1.2540066 reg_l1 48.98199 reg_l2 29.411774
loss 6.1522055
STEP 8 ================================
prereg loss 1.2482054 reg_l1 48.955223 reg_l2 29.393497
loss 6.1437283
STEP 9 ================================
prereg loss 1.2424628 reg_l1 48.928047 reg_l2 29.375364
loss 6.1352673
STEP 10 ================================
prereg loss 1.2367716 reg_l1 48.900517 reg_l2 29.35738
loss 6.1268234
STEP 11 ================================
prereg loss 1.2311283 reg_l1 48.87268 reg_l2 29.339525
loss 6.1183963
STEP 12 ================================
prereg loss 1.225547 reg_l1 48.844555 reg_l2 29.3218
loss 6.1100025
STEP 13 ================================
prereg loss 1.2200139 reg_l1 48.816177 reg_l2 29.304224
loss 6.101632
STEP 14 ================================
prereg loss 1.2145324 reg_l1 48.78761 reg_l2 29.286764
loss 6.093293
STEP 15 ================================
prereg loss 1.209103 reg_l1 48.75883 reg_l2 29.26945
loss 6.084986
STEP 16 ================================
prereg loss 1.203725 reg_l1 48.729904 reg_l2 29.252266
loss 6.0767155
STEP 17 ================================
prereg loss 1.1983969 reg_l1 48.700832 reg_l2 29.23521
loss 6.0684805
STEP 18 ================================
prereg loss 1.1931205 reg_l1 48.674255 reg_l2 29.2183
loss 6.060546
STEP 19 ================================
prereg loss 1.1878957 reg_l1 48.647465 reg_l2 29.201506
loss 6.0526423
STEP 20 ================================
prereg loss 1.1827182 reg_l1 48.620335 reg_l2 29.184853
loss 6.0447516
STEP 21 ================================
prereg loss 1.1775936 reg_l1 48.5929 reg_l2 29.168316
loss 6.036884
cutoff 0.0030430488 network size 98
STEP 22 ================================
prereg loss 1.1725107 reg_l1 48.56214 reg_l2 29.151901
loss 6.0287247
STEP 23 ================================
prereg loss 1.1674833 reg_l1 48.53452 reg_l2 29.135605
loss 6.0209355
STEP 24 ================================
prereg loss 1.1624992 reg_l1 48.50693 reg_l2 29.11945
loss 6.013192
STEP 25 ================================
prereg loss 1.157568 reg_l1 48.479397 reg_l2 29.103407
loss 6.005508
STEP 26 ================================
prereg loss 1.1526777 reg_l1 48.45186 reg_l2 29.087486
loss 5.9978633
STEP 27 ================================
prereg loss 1.1478394 reg_l1 48.42437 reg_l2 29.071693
loss 5.990277
STEP 28 ================================
prereg loss 1.1430435 reg_l1 48.3969 reg_l2 29.056017
loss 5.9827337
STEP 29 ================================
prereg loss 1.1382974 reg_l1 48.3695 reg_l2 29.040453
loss 5.9752474
STEP 30 ================================
prereg loss 1.1335946 reg_l1 48.342075 reg_l2 29.025015
loss 5.967802
STEP 31 ================================
prereg loss 1.1289375 reg_l1 48.31471 reg_l2 29.00969
loss 5.960408
STEP 32 ================================
prereg loss 1.1243262 reg_l1 48.28736 reg_l2 28.994478
loss 5.9530625
STEP 33 ================================
prereg loss 1.1197555 reg_l1 48.26004 reg_l2 28.979385
loss 5.94576
STEP 34 ================================
prereg loss 1.1152363 reg_l1 48.236233 reg_l2 28.964413
loss 5.9388595
STEP 35 ================================
prereg loss 1.1107529 reg_l1 48.21306 reg_l2 28.94954
loss 5.932059
STEP 36 ================================
prereg loss 1.1063191 reg_l1 48.189465 reg_l2 28.934774
loss 5.9252653
STEP 37 ================================
prereg loss 1.1019224 reg_l1 48.16674 reg_l2 28.920115
loss 5.9185967
STEP 38 ================================
prereg loss 1.0975736 reg_l1 48.146793 reg_l2 28.905535
loss 5.9122534
STEP 39 ================================
prereg loss 1.0932642 reg_l1 48.12616 reg_l2 28.891048
loss 5.90588
STEP 40 ================================
prereg loss 1.0889932 reg_l1 48.10487 reg_l2 28.876656
loss 5.8994803
STEP 41 ================================
prereg loss 1.0847697 reg_l1 48.083027 reg_l2 28.862347
loss 5.8930726
cutoff 0.006227193 network size 97
STEP 42 ================================
prereg loss 1.0805863 reg_l1 48.05442 reg_l2 28.848064
loss 5.886029
STEP 43 ================================
prereg loss 1.0764408 reg_l1 48.03147 reg_l2 28.83391
loss 5.879588
STEP 44 ================================
prereg loss 1.0723358 reg_l1 48.008354 reg_l2 28.81984
loss 5.8731713
STEP 45 ================================
prereg loss 1.0682762 reg_l1 47.985077 reg_l2 28.805853
loss 5.866784
STEP 46 ================================
prereg loss 1.06425 reg_l1 47.962475 reg_l2 28.791946
loss 5.8604975
STEP 47 ================================
prereg loss 1.0602643 reg_l1 47.943146 reg_l2 28.778124
loss 5.854579
STEP 48 ================================
prereg loss 1.0563186 reg_l1 47.923252 reg_l2 28.764374
loss 5.8486443
STEP 49 ================================
prereg loss 1.0524093 reg_l1 47.902912 reg_l2 28.750708
loss 5.8427005
STEP 50 ================================
prereg loss 1.0485445 reg_l1 47.884354 reg_l2 28.737112
loss 5.83698
STEP 51 ================================
prereg loss 1.0447106 reg_l1 47.86573 reg_l2 28.72358
loss 5.8312836
STEP 52 ================================
prereg loss 1.0409151 reg_l1 47.850285 reg_l2 28.710121
loss 5.8259435
STEP 53 ================================
prereg loss 1.0371604 reg_l1 47.83421 reg_l2 28.696718
loss 5.8205814
STEP 54 ================================
prereg loss 1.033441 reg_l1 47.817303 reg_l2 28.683384
loss 5.8151712
STEP 55 ================================
prereg loss 1.0297593 reg_l1 47.79963 reg_l2 28.670092
loss 5.8097224
STEP 56 ================================
prereg loss 1.026115 reg_l1 47.781277 reg_l2 28.656868
loss 5.8042426
STEP 57 ================================
prereg loss 1.0225022 reg_l1 47.762314 reg_l2 28.64369
loss 5.7987337
STEP 58 ================================
prereg loss 1.0189276 reg_l1 47.74283 reg_l2 28.630575
loss 5.7932105
STEP 59 ================================
prereg loss 1.015389 reg_l1 47.72283 reg_l2 28.617504
loss 5.787672
STEP 60 ================================
prereg loss 1.0118871 reg_l1 47.702415 reg_l2 28.604502
loss 5.782129
STEP 61 ================================
prereg loss 1.0084156 reg_l1 47.685555 reg_l2 28.591566
loss 5.7769713
cutoff 0.0016233998 network size 96
STEP 62 ================================
prereg loss 1.0049775 reg_l1 47.667007 reg_l2 28.578682
loss 5.771678
STEP 63 ================================
prereg loss 1.0015755 reg_l1 47.648907 reg_l2 28.565847
loss 5.766466
STEP 64 ================================
prereg loss 0.998206 reg_l1 47.63041 reg_l2 28.553082
loss 5.761247
STEP 65 ================================
prereg loss 0.9948692 reg_l1 47.611538 reg_l2 28.540358
loss 5.756023
STEP 66 ================================
prereg loss 0.99156976 reg_l1 47.59233 reg_l2 28.527693
loss 5.750803
STEP 67 ================================
prereg loss 0.9882969 reg_l1 47.572834 reg_l2 28.515089
loss 5.7455807
STEP 68 ================================
prereg loss 0.9850605 reg_l1 47.55445 reg_l2 28.502531
loss 5.7405057
STEP 69 ================================
prereg loss 0.9818521 reg_l1 47.536724 reg_l2 28.490047
loss 5.7355247
STEP 70 ================================
prereg loss 0.97868294 reg_l1 47.518547 reg_l2 28.477598
loss 5.730538
STEP 71 ================================
prereg loss 0.97554106 reg_l1 47.5008 reg_l2 28.465223
loss 5.725621
STEP 72 ================================
prereg loss 0.9724301 reg_l1 47.48361 reg_l2 28.452883
loss 5.7207913
STEP 73 ================================
prereg loss 0.9693518 reg_l1 47.465954 reg_l2 28.44062
loss 5.715947
STEP 74 ================================
prereg loss 0.9662972 reg_l1 47.447838 reg_l2 28.428387
loss 5.711081
STEP 75 ================================
prereg loss 0.9632825 reg_l1 47.429333 reg_l2 28.416204
loss 5.706216
STEP 76 ================================
prereg loss 0.9602908 reg_l1 47.410458 reg_l2 28.40408
loss 5.701337
STEP 77 ================================
prereg loss 0.9573322 reg_l1 47.39128 reg_l2 28.392012
loss 5.6964602
STEP 78 ================================
prereg loss 0.9544018 reg_l1 47.372185 reg_l2 28.379986
loss 5.6916203
STEP 79 ================================
prereg loss 0.951501 reg_l1 47.355164 reg_l2 28.368021
loss 5.6870174
STEP 80 ================================
prereg loss 0.9486292 reg_l1 47.33764 reg_l2 28.356106
loss 5.6823936
STEP 81 ================================
prereg loss 0.9457844 reg_l1 47.319664 reg_l2 28.344242
loss 5.677751
cutoff 0.00013774261 network size 95
STEP 82 ================================
prereg loss 0.94296557 reg_l1 47.30116 reg_l2 28.332418
loss 5.6730814
STEP 83 ================================
prereg loss 0.9401838 reg_l1 47.282883 reg_l2 28.32065
loss 5.668472
STEP 84 ================================
prereg loss 0.93741876 reg_l1 47.264465 reg_l2 28.308931
loss 5.6638656
STEP 85 ================================
prereg loss 0.934686 reg_l1 47.246624 reg_l2 28.29728
loss 5.6593485
STEP 86 ================================
prereg loss 0.9319814 reg_l1 47.229023 reg_l2 28.285656
loss 5.654884
STEP 87 ================================
prereg loss 0.9293027 reg_l1 47.2112 reg_l2 28.274082
loss 5.650423
STEP 88 ================================
prereg loss 0.9266486 reg_l1 47.193142 reg_l2 28.26256
loss 5.6459627
STEP 89 ================================
prereg loss 0.92402166 reg_l1 47.17493 reg_l2 28.251081
loss 5.641515
STEP 90 ================================
prereg loss 0.92142177 reg_l1 47.156532 reg_l2 28.239653
loss 5.6370754
STEP 91 ================================
prereg loss 0.9188521 reg_l1 47.138996 reg_l2 28.228277
loss 5.6327515
STEP 92 ================================
prereg loss 0.9162999 reg_l1 47.121387 reg_l2 28.21695
loss 5.6284385
STEP 93 ================================
prereg loss 0.9137752 reg_l1 47.10354 reg_l2 28.205666
loss 5.6241293
STEP 94 ================================
prereg loss 0.91127735 reg_l1 47.08548 reg_l2 28.194433
loss 5.6198254
STEP 95 ================================
prereg loss 0.90880436 reg_l1 47.06723 reg_l2 28.183252
loss 5.6155276
STEP 96 ================================
prereg loss 0.90635663 reg_l1 47.04883 reg_l2 28.17211
loss 5.61124
STEP 97 ================================
prereg loss 0.9039318 reg_l1 47.03122 reg_l2 28.16101
loss 5.6070538
STEP 98 ================================
prereg loss 0.90152776 reg_l1 47.013596 reg_l2 28.149958
loss 5.6028876
STEP 99 ================================
prereg loss 0.89915293 reg_l1 46.995754 reg_l2 28.138954
loss 5.598728
STEP 100 ================================
prereg loss 0.8967973 reg_l1 46.97769 reg_l2 28.128
loss 5.5945663
2022-07-20T15:06:12.912

julia> serialize("cf-95-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-95-parameters-opt.ser", opt)
```

That's much better!

```
julia> interleaving_steps!(100, 20)
2022-07-20T15:15:03.782
STEP 1 ================================
prereg loss 0.89446646 reg_l1 46.959476 reg_l2 28.117086
loss 5.590414
cutoff 0.00013942661 network size 94
STEP 2 ================================
prereg loss 0.8921592 reg_l1 46.94093 reg_l2 28.106228
loss 5.586252
STEP 3 ================================
prereg loss 0.88987815 reg_l1 46.922974 reg_l2 28.095404
loss 5.5821757
STEP 4 ================================
prereg loss 0.88761055 reg_l1 46.905 reg_l2 28.084627
loss 5.57811
STEP 5 ================================
prereg loss 0.88539916 reg_l1 46.88706 reg_l2 28.07389
loss 5.5741053
STEP 6 ================================
prereg loss 0.88315445 reg_l1 46.869118 reg_l2 28.063234
loss 5.5700665
STEP 7 ================================
prereg loss 0.88096213 reg_l1 46.85118 reg_l2 28.052631
loss 5.56608
STEP 8 ================================
prereg loss 0.87878764 reg_l1 46.83324 reg_l2 28.042051
loss 5.562112
STEP 9 ================================
prereg loss 0.8766344 reg_l1 46.815296 reg_l2 28.031519
loss 5.558164
STEP 10 ================================
prereg loss 0.8745079 reg_l1 46.797337 reg_l2 28.021015
loss 5.5542417
STEP 11 ================================
prereg loss 0.87239444 reg_l1 46.7803 reg_l2 28.010551
loss 5.5504246
STEP 12 ================================
prereg loss 0.870303 reg_l1 46.765266 reg_l2 28.000122
loss 5.5468297
STEP 13 ================================
prereg loss 0.8681787 reg_l1 46.74992 reg_l2 27.989702
loss 5.543171
STEP 14 ================================
prereg loss 0.86604846 reg_l1 46.734264 reg_l2 27.979326
loss 5.539475
STEP 15 ================================
prereg loss 0.8639085 reg_l1 46.71837 reg_l2 27.968952
loss 5.5357456
STEP 16 ================================
prereg loss 0.8617706 reg_l1 46.702225 reg_l2 27.95862
loss 5.5319934
STEP 17 ================================
prereg loss 0.85963243 reg_l1 46.68588 reg_l2 27.948317
loss 5.5282207
STEP 18 ================================
prereg loss 0.85750484 reg_l1 46.66936 reg_l2 27.938053
loss 5.5244412
STEP 19 ================================
prereg loss 0.8553854 reg_l1 46.65269 reg_l2 27.92782
loss 5.5206547
STEP 20 ================================
prereg loss 0.8533082 reg_l1 46.635857 reg_l2 27.917635
loss 5.516894
STEP 21 ================================
prereg loss 0.8511915 reg_l1 46.618958 reg_l2 27.907532
loss 5.5130873
cutoff 0.0026820297 network size 93
STEP 22 ================================
prereg loss 0.8491231 reg_l1 46.599247 reg_l2 27.897457
loss 5.509048
STEP 23 ================================
prereg loss 0.84707683 reg_l1 46.582832 reg_l2 27.887432
loss 5.50536
STEP 24 ================================
prereg loss 0.84505004 reg_l1 46.56641 reg_l2 27.877468
loss 5.501691
STEP 25 ================================
prereg loss 0.84304625 reg_l1 46.54998 reg_l2 27.86753
loss 5.4980445
STEP 26 ================================
prereg loss 0.8410655 reg_l1 46.533573 reg_l2 27.85763
loss 5.494423
STEP 27 ================================
prereg loss 0.8391076 reg_l1 46.517155 reg_l2 27.847778
loss 5.4908233
STEP 28 ================================
prereg loss 0.83717525 reg_l1 46.50074 reg_l2 27.837978
loss 5.4872494
STEP 29 ================================
prereg loss 0.83526 reg_l1 46.48431 reg_l2 27.82821
loss 5.483691
STEP 30 ================================
prereg loss 0.8333737 reg_l1 46.467922 reg_l2 27.818481
loss 5.480166
STEP 31 ================================
prereg loss 0.8315058 reg_l1 46.451504 reg_l2 27.808798
loss 5.4766564
STEP 32 ================================
prereg loss 0.82966244 reg_l1 46.43509 reg_l2 27.799152
loss 5.473171
STEP 33 ================================
prereg loss 0.82783484 reg_l1 46.418686 reg_l2 27.789549
loss 5.4697037
STEP 34 ================================
prereg loss 0.8260309 reg_l1 46.402264 reg_l2 27.779984
loss 5.466257
STEP 35 ================================
prereg loss 0.82424366 reg_l1 46.38586 reg_l2 27.770458
loss 5.4628296
STEP 36 ================================
prereg loss 0.8224739 reg_l1 46.36945 reg_l2 27.760967
loss 5.4594193
STEP 37 ================================
prereg loss 0.8207222 reg_l1 46.35305 reg_l2 27.751514
loss 5.456027
STEP 38 ================================
prereg loss 0.8189795 reg_l1 46.336662 reg_l2 27.742125
loss 5.4526463
STEP 39 ================================
prereg loss 0.81725365 reg_l1 46.320286 reg_l2 27.732777
loss 5.449282
STEP 40 ================================
prereg loss 0.8155462 reg_l1 46.3039 reg_l2 27.72346
loss 5.445936
STEP 41 ================================
prereg loss 0.8138454 reg_l1 46.287506 reg_l2 27.714165
loss 5.4425964
cutoff 0.0021815018 network size 92
STEP 42 ================================
prereg loss 0.81216496 reg_l1 46.2689 reg_l2 27.70489
loss 5.439055
STEP 43 ================================
prereg loss 0.8104923 reg_l1 46.25413 reg_l2 27.69565
loss 5.4359055
STEP 44 ================================
prereg loss 0.80883455 reg_l1 46.239334 reg_l2 27.686445
loss 5.432768
STEP 45 ================================
prereg loss 0.8071844 reg_l1 46.224545 reg_l2 27.677256
loss 5.429639
STEP 46 ================================
prereg loss 0.8055515 reg_l1 46.20974 reg_l2 27.66809
loss 5.4265256
STEP 47 ================================
prereg loss 0.80392915 reg_l1 46.194916 reg_l2 27.658945
loss 5.423421
STEP 48 ================================
prereg loss 0.8023223 reg_l1 46.18009 reg_l2 27.649836
loss 5.4203315
STEP 49 ================================
prereg loss 0.8007254 reg_l1 46.165264 reg_l2 27.640749
loss 5.417252
STEP 50 ================================
prereg loss 0.79914165 reg_l1 46.150433 reg_l2 27.631693
loss 5.4141846
STEP 51 ================================
prereg loss 0.79757637 reg_l1 46.135597 reg_l2 27.622667
loss 5.411136
STEP 52 ================================
prereg loss 0.7960204 reg_l1 46.120777 reg_l2 27.613678
loss 5.408098
STEP 53 ================================
prereg loss 0.7944786 reg_l1 46.10595 reg_l2 27.604717
loss 5.4050736
STEP 54 ================================
prereg loss 0.7929558 reg_l1 46.091118 reg_l2 27.595787
loss 5.4020677
STEP 55 ================================
prereg loss 0.79144424 reg_l1 46.076286 reg_l2 27.586891
loss 5.399073
STEP 56 ================================
prereg loss 0.7899538 reg_l1 46.061447 reg_l2 27.578045
loss 5.3960986
STEP 57 ================================
prereg loss 0.78847003 reg_l1 46.046677 reg_l2 27.569258
loss 5.393138
STEP 58 ================================
prereg loss 0.7870089 reg_l1 46.03189 reg_l2 27.560488
loss 5.3901978
STEP 59 ================================
prereg loss 0.7855529 reg_l1 46.01709 reg_l2 27.551764
loss 5.387262
STEP 60 ================================
prereg loss 0.78412247 reg_l1 46.002304 reg_l2 27.543058
loss 5.384353
STEP 61 ================================
prereg loss 0.78270155 reg_l1 45.987495 reg_l2 27.53438
loss 5.381451
cutoff 0.012608112 network size 91
STEP 62 ================================
prereg loss 0.78129756 reg_l1 45.96007 reg_l2 27.525566
loss 5.377305
STEP 63 ================================
prereg loss 0.77991015 reg_l1 45.946888 reg_l2 27.51698
loss 5.374599
STEP 64 ================================
prereg loss 0.7784858 reg_l1 45.933693 reg_l2 27.508398
loss 5.3718553
STEP 65 ================================
prereg loss 0.7770769 reg_l1 45.920433 reg_l2 27.499796
loss 5.36912
STEP 66 ================================
prereg loss 0.77568614 reg_l1 45.907112 reg_l2 27.491138
loss 5.3663974
STEP 67 ================================
prereg loss 0.7743073 reg_l1 45.893745 reg_l2 27.48245
loss 5.363682
STEP 68 ================================
prereg loss 0.77294123 reg_l1 45.88037 reg_l2 27.473743
loss 5.360978
STEP 69 ================================
prereg loss 0.7715907 reg_l1 45.866932 reg_l2 27.46502
loss 5.358284
STEP 70 ================================
prereg loss 0.77024555 reg_l1 45.853485 reg_l2 27.45628
loss 5.355594
STEP 71 ================================
prereg loss 0.7689152 reg_l1 45.840004 reg_l2 27.447542
loss 5.352916
STEP 72 ================================
prereg loss 0.76758814 reg_l1 45.826523 reg_l2 27.438816
loss 5.3502407
STEP 73 ================================
prereg loss 0.76627016 reg_l1 45.813015 reg_l2 27.430094
loss 5.347572
STEP 74 ================================
prereg loss 0.76495856 reg_l1 45.799503 reg_l2 27.421392
loss 5.3449087
STEP 75 ================================
prereg loss 0.7636528 reg_l1 45.785984 reg_l2 27.412716
loss 5.3422513
STEP 76 ================================
prereg loss 0.7623563 reg_l1 45.772472 reg_l2 27.404066
loss 5.3396034
STEP 77 ================================
prereg loss 0.7610568 reg_l1 45.758987 reg_l2 27.395458
loss 5.3369555
STEP 78 ================================
prereg loss 0.75976044 reg_l1 45.745506 reg_l2 27.38689
loss 5.334311
STEP 79 ================================
prereg loss 0.7584695 reg_l1 45.732025 reg_l2 27.37834
loss 5.331672
STEP 80 ================================
prereg loss 0.75718313 reg_l1 45.718525 reg_l2 27.369808
loss 5.3290358
STEP 81 ================================
prereg loss 0.7559005 reg_l1 45.705048 reg_l2 27.361315
loss 5.326405
cutoff 0.057989337 network size 90
STEP 82 ================================
prereg loss 0.7546176 reg_l1 45.633568 reg_l2 27.349478
loss 5.3179746
STEP 83 ================================
prereg loss 0.7533439 reg_l1 45.621 reg_l2 27.34113
loss 5.315444
STEP 84 ================================
prereg loss 0.75207144 reg_l1 45.608437 reg_l2 27.332819
loss 5.3129153
STEP 85 ================================
prereg loss 0.7508054 reg_l1 45.59587 reg_l2 27.32452
loss 5.3103924
STEP 86 ================================
prereg loss 0.74954486 reg_l1 45.58331 reg_l2 27.316236
loss 5.3078756
STEP 87 ================================
prereg loss 0.748291 reg_l1 45.57073 reg_l2 27.307983
loss 5.305364
STEP 88 ================================
prereg loss 0.74704427 reg_l1 45.558136 reg_l2 27.299755
loss 5.302858
STEP 89 ================================
prereg loss 0.7458078 reg_l1 45.545555 reg_l2 27.291529
loss 5.300363
STEP 90 ================================
prereg loss 0.7445755 reg_l1 45.532974 reg_l2 27.283335
loss 5.297873
STEP 91 ================================
prereg loss 0.743353 reg_l1 45.52038 reg_l2 27.275162
loss 5.2953906
STEP 92 ================================
prereg loss 0.7421406 reg_l1 45.507767 reg_l2 27.267006
loss 5.2929177
STEP 93 ================================
prereg loss 0.7409383 reg_l1 45.495163 reg_l2 27.25887
loss 5.2904544
STEP 94 ================================
prereg loss 0.7397435 reg_l1 45.482548 reg_l2 27.250746
loss 5.287998
STEP 95 ================================
prereg loss 0.7385574 reg_l1 45.46992 reg_l2 27.242647
loss 5.2855496
STEP 96 ================================
prereg loss 0.7373848 reg_l1 45.457283 reg_l2 27.234562
loss 5.283113
STEP 97 ================================
prereg loss 0.7362205 reg_l1 45.444637 reg_l2 27.226494
loss 5.280684
STEP 98 ================================
prereg loss 0.7350661 reg_l1 45.432003 reg_l2 27.218454
loss 5.2782664
STEP 99 ================================
prereg loss 0.73392373 reg_l1 45.41936 reg_l2 27.210432
loss 5.2758603
STEP 100 ================================
prereg loss 0.7327869 reg_l1 45.406693 reg_l2 27.202427
loss 5.2734566
2022-07-20T15:21:57.512

julia> serialize("cf-90-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-90-parameters-opt.ser", opt)

julia> interleaving_steps!(100, 20)
2022-07-20T15:22:30.327
STEP 1 ================================
prereg loss 0.73166275 reg_l1 45.39404 reg_l2 27.194433
loss 5.2710667
cutoff 0.039634436 network size 89
STEP 2 ================================
prereg loss 0.73054695 reg_l1 45.341743 reg_l2 27.184902
loss 5.2647214
STEP 3 ================================
prereg loss 0.7294419 reg_l1 45.33077 reg_l2 27.177084
loss 5.262519
STEP 4 ================================
prereg loss 0.72835964 reg_l1 45.31982 reg_l2 27.169285
loss 5.2603416
STEP 5 ================================
prereg loss 0.7272527 reg_l1 45.308884 reg_l2 27.16152
loss 5.2581415
STEP 6 ================================
prereg loss 0.7261715 reg_l1 45.297943 reg_l2 27.153776
loss 5.2559657
STEP 7 ================================
prereg loss 0.72509974 reg_l1 45.286983 reg_l2 27.14604
loss 5.253798
STEP 8 ================================
prereg loss 0.7240343 reg_l1 45.275993 reg_l2 27.138308
loss 5.2516336
STEP 9 ================================
prereg loss 0.7229756 reg_l1 45.265022 reg_l2 27.130589
loss 5.249478
STEP 10 ================================
prereg loss 0.72192365 reg_l1 45.254013 reg_l2 27.122875
loss 5.2473254
STEP 11 ================================
prereg loss 0.72087955 reg_l1 45.24302 reg_l2 27.115173
loss 5.2451816
STEP 12 ================================
prereg loss 0.71984136 reg_l1 45.23201 reg_l2 27.107475
loss 5.2430425
STEP 13 ================================
prereg loss 0.7188113 reg_l1 45.220978 reg_l2 27.0998
loss 5.2409096
STEP 14 ================================
prereg loss 0.717786 reg_l1 45.209934 reg_l2 27.092121
loss 5.2387795
STEP 15 ================================
prereg loss 0.71676666 reg_l1 45.19887 reg_l2 27.084457
loss 5.2366543
STEP 16 ================================
prereg loss 0.71575534 reg_l1 45.187828 reg_l2 27.076815
loss 5.2345386
STEP 17 ================================
prereg loss 0.7147494 reg_l1 45.17677 reg_l2 27.069193
loss 5.232426
STEP 18 ================================
prereg loss 0.7137515 reg_l1 45.165707 reg_l2 27.061577
loss 5.230322
STEP 19 ================================
prereg loss 0.71275914 reg_l1 45.154644 reg_l2 27.05398
loss 5.2282233
STEP 20 ================================
prereg loss 0.71177274 reg_l1 45.14356 reg_l2 27.0464
loss 5.226129
STEP 21 ================================
prereg loss 0.7107958 reg_l1 45.132496 reg_l2 27.038847
loss 5.2240458
cutoff 0.07812405 network size 88
STEP 22 ================================
prereg loss 0.71012187 reg_l1 45.043293 reg_l2 27.025206
loss 5.214451
STEP 23 ================================
prereg loss 0.709149 reg_l1 45.033237 reg_l2 27.017847
loss 5.212473
STEP 24 ================================
prereg loss 0.7081909 reg_l1 45.02318 reg_l2 27.010494
loss 5.210509
STEP 25 ================================
prereg loss 0.707236 reg_l1 45.013126 reg_l2 27.003162
loss 5.2085485
STEP 26 ================================
prereg loss 0.7062918 reg_l1 45.003044 reg_l2 26.995844
loss 5.2065964
STEP 27 ================================
prereg loss 0.70536566 reg_l1 44.992977 reg_l2 26.988548
loss 5.2046633
STEP 28 ================================
prereg loss 0.704425 reg_l1 44.982918 reg_l2 26.981293
loss 5.202717
STEP 29 ================================
prereg loss 0.7035037 reg_l1 44.972893 reg_l2 26.974047
loss 5.200793
STEP 30 ================================
prereg loss 0.70258635 reg_l1 44.962826 reg_l2 26.966818
loss 5.1988688
STEP 31 ================================
prereg loss 0.70167327 reg_l1 44.952747 reg_l2 26.959593
loss 5.196948
STEP 32 ================================
prereg loss 0.70077056 reg_l1 44.942677 reg_l2 26.95238
loss 5.1950383
STEP 33 ================================
prereg loss 0.69986814 reg_l1 44.932594 reg_l2 26.945189
loss 5.1931276
STEP 34 ================================
prereg loss 0.6989713 reg_l1 44.92252 reg_l2 26.937994
loss 5.191223
STEP 35 ================================
prereg loss 0.6980778 reg_l1 44.91241 reg_l2 26.930824
loss 5.1893187
STEP 36 ================================
prereg loss 0.69719046 reg_l1 44.902317 reg_l2 26.923653
loss 5.1874223
STEP 37 ================================
prereg loss 0.69630796 reg_l1 44.892204 reg_l2 26.916496
loss 5.1855288
STEP 38 ================================
prereg loss 0.69542795 reg_l1 44.882076 reg_l2 26.909353
loss 5.1836357
STEP 39 ================================
prereg loss 0.69455546 reg_l1 44.87196 reg_l2 26.90222
loss 5.1817513
STEP 40 ================================
prereg loss 0.6936836 reg_l1 44.86183 reg_l2 26.895103
loss 5.179867
STEP 41 ================================
prereg loss 0.69282097 reg_l1 44.851707 reg_l2 26.888008
loss 5.177992
cutoff 0.06959124 network size 87
STEP 42 ================================
prereg loss 0.69560045 reg_l1 44.771973 reg_l2 26.876066
loss 5.1727977
STEP 43 ================================
prereg loss 0.69460666 reg_l1 44.762527 reg_l2 26.8692
loss 5.17086
STEP 44 ================================
prereg loss 0.6935231 reg_l1 44.753117 reg_l2 26.862423
loss 5.1688347
STEP 45 ================================
prereg loss 0.6923662 reg_l1 44.74376 reg_l2 26.855766
loss 5.1667423
STEP 46 ================================
prereg loss 0.6911489 reg_l1 44.734394 reg_l2 26.849169
loss 5.1645885
STEP 47 ================================
prereg loss 0.6898892 reg_l1 44.725044 reg_l2 26.842617
loss 5.1623936
STEP 48 ================================
prereg loss 0.6885962 reg_l1 44.715702 reg_l2 26.836123
loss 5.1601667
STEP 49 ================================
prereg loss 0.68728554 reg_l1 44.706337 reg_l2 26.829641
loss 5.1579194
STEP 50 ================================
prereg loss 0.68596506 reg_l1 44.69693 reg_l2 26.823166
loss 5.1556582
STEP 51 ================================
prereg loss 0.68463844 reg_l1 44.687515 reg_l2 26.816683
loss 5.15339
STEP 52 ================================
prereg loss 0.6833178 reg_l1 44.678017 reg_l2 26.81017
loss 5.151119
STEP 53 ================================
prereg loss 0.68200487 reg_l1 44.668457 reg_l2 26.803625
loss 5.148851
STEP 54 ================================
prereg loss 0.6807053 reg_l1 44.658867 reg_l2 26.797033
loss 5.146592
STEP 55 ================================
prereg loss 0.6794164 reg_l1 44.649166 reg_l2 26.790403
loss 5.144333
STEP 56 ================================
prereg loss 0.67814374 reg_l1 44.639397 reg_l2 26.783714
loss 5.142083
STEP 57 ================================
prereg loss 0.6768866 reg_l1 44.62959 reg_l2 26.776985
loss 5.1398454
STEP 58 ================================
prereg loss 0.6756492 reg_l1 44.619698 reg_l2 26.770184
loss 5.137619
STEP 59 ================================
prereg loss 0.67442524 reg_l1 44.609745 reg_l2 26.763348
loss 5.1354
STEP 60 ================================
prereg loss 0.67321944 reg_l1 44.599724 reg_l2 26.756456
loss 5.133192
STEP 61 ================================
prereg loss 0.672031 reg_l1 44.58966 reg_l2 26.74953
loss 5.130997
cutoff 0.07311721 network size 86
STEP 62 ================================
prereg loss 0.67085856 reg_l1 44.50642 reg_l2 26.737226
loss 5.1215005
STEP 63 ================================
prereg loss 0.66970664 reg_l1 44.49806 reg_l2 26.730497
loss 5.1195126
STEP 64 ================================
prereg loss 0.6685669 reg_l1 44.48966 reg_l2 26.723743
loss 5.1175327
STEP 65 ================================
prereg loss 0.66769993 reg_l1 44.48124 reg_l2 26.716972
loss 5.1158237
STEP 66 ================================
prereg loss 0.6670691 reg_l1 44.47282 reg_l2 26.710176
loss 5.1143513
STEP 67 ================================
prereg loss 0.6664512 reg_l1 44.46439 reg_l2 26.703342
loss 5.1128902
STEP 68 ================================
prereg loss 0.665846 reg_l1 44.45597 reg_l2 26.696503
loss 5.111443
STEP 69 ================================
prereg loss 0.66525006 reg_l1 44.447556 reg_l2 26.689648
loss 5.1100054
STEP 70 ================================
prereg loss 0.6646665 reg_l1 44.439163 reg_l2 26.6828
loss 5.108583
STEP 71 ================================
prereg loss 0.6640933 reg_l1 44.43078 reg_l2 26.675957
loss 5.1071715
STEP 72 ================================
prereg loss 0.66352654 reg_l1 44.422417 reg_l2 26.66914
loss 5.105768
STEP 73 ================================
prereg loss 0.6629654 reg_l1 44.414104 reg_l2 26.662354
loss 5.104376
STEP 74 ================================
prereg loss 0.6624103 reg_l1 44.405804 reg_l2 26.655607
loss 5.1029906
STEP 75 ================================
prereg loss 0.6618582 reg_l1 44.39756 reg_l2 26.648884
loss 5.101614
STEP 76 ================================
prereg loss 0.66130614 reg_l1 44.389355 reg_l2 26.642218
loss 5.1002417
STEP 77 ================================
prereg loss 0.6607551 reg_l1 44.381176 reg_l2 26.635597
loss 5.0988727
STEP 78 ================================
prereg loss 0.6602001 reg_l1 44.37306 reg_l2 26.62903
loss 5.097506
STEP 79 ================================
prereg loss 0.6596456 reg_l1 44.364986 reg_l2 26.622509
loss 5.096144
STEP 80 ================================
prereg loss 0.6590847 reg_l1 44.35692 reg_l2 26.616037
loss 5.0947766
STEP 81 ================================
prereg loss 0.6585215 reg_l1 44.348923 reg_l2 26.609617
loss 5.093414
cutoff 0.1849603 network size 85
STEP 82 ================================
prereg loss 0.6579505 reg_l1 44.155964 reg_l2 26.569027
loss 5.073547
STEP 83 ================================
prereg loss 0.65737885 reg_l1 44.14885 reg_l2 26.562992
loss 5.0722637
STEP 84 ================================
prereg loss 0.6568039 reg_l1 44.141747 reg_l2 26.556992
loss 5.0709786
STEP 85 ================================
prereg loss 0.6562265 reg_l1 44.134655 reg_l2 26.551023
loss 5.069692
STEP 86 ================================
prereg loss 0.6556442 reg_l1 44.127575 reg_l2 26.545076
loss 5.0684013
STEP 87 ================================
prereg loss 0.65506274 reg_l1 44.120514 reg_l2 26.539156
loss 5.0671144
STEP 88 ================================
prereg loss 0.65448135 reg_l1 44.11347 reg_l2 26.533249
loss 5.0658283
STEP 89 ================================
prereg loss 0.6538999 reg_l1 44.106403 reg_l2 26.527346
loss 5.06454
STEP 90 ================================
prereg loss 0.65331954 reg_l1 44.099354 reg_l2 26.521458
loss 5.063255
STEP 91 ================================
prereg loss 0.65273905 reg_l1 44.092285 reg_l2 26.515575
loss 5.061968
STEP 92 ================================
prereg loss 0.65216374 reg_l1 44.085228 reg_l2 26.509687
loss 5.060687
STEP 93 ================================
prereg loss 0.65158975 reg_l1 44.078136 reg_l2 26.503809
loss 5.0594034
STEP 94 ================================
prereg loss 0.6510194 reg_l1 44.071053 reg_l2 26.49791
loss 5.058125
STEP 95 ================================
prereg loss 0.6504548 reg_l1 44.06394 reg_l2 26.492014
loss 5.056849
STEP 96 ================================
prereg loss 0.64989346 reg_l1 44.056805 reg_l2 26.486115
loss 5.055574
STEP 97 ================================
prereg loss 0.64933974 reg_l1 44.049686 reg_l2 26.480202
loss 5.0543084
STEP 98 ================================
prereg loss 0.64878654 reg_l1 44.04252 reg_l2 26.474283
loss 5.0530386
STEP 99 ================================
prereg loss 0.64824593 reg_l1 44.035366 reg_l2 26.468357
loss 5.0517826
STEP 100 ================================
prereg loss 0.6477057 reg_l1 44.028168 reg_l2 26.462433
loss 5.0505223
2022-07-20T15:29:11.138

julia> serialize("cf-85-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-85-parameters-opt.ser", opt)
```

Very good, but let's run 100 steps to regularize better and switch to
5 sparsifications per 200 steps.

```
julia> steps!(100)
2022-07-20T15:33:25.885
STEP 1 ================================
prereg loss 0.64717245 reg_l1 44.020973 reg_l2 26.456493
loss 5.0492697
STEP 2 ================================
prereg loss 0.64664626 reg_l1 44.013763 reg_l2 26.450563
loss 5.0480223
STEP 3 ================================
prereg loss 0.6461265 reg_l1 44.006557 reg_l2 26.44462
loss 5.0467825
STEP 4 ================================
prereg loss 0.64561146 reg_l1 43.999332 reg_l2 26.438696
loss 5.0455446
STEP 5 ================================
prereg loss 0.64510286 reg_l1 43.99209 reg_l2 26.432753
loss 5.044312
STEP 6 ================================
prereg loss 0.6445978 reg_l1 43.984837 reg_l2 26.42683
loss 5.0430818
STEP 7 ================================
prereg loss 0.6440995 reg_l1 43.977592 reg_l2 26.420914
loss 5.0418587
STEP 8 ================================
prereg loss 0.6436038 reg_l1 43.970352 reg_l2 26.415
loss 5.040639
STEP 9 ================================
prereg loss 0.6431123 reg_l1 43.963104 reg_l2 26.409098
loss 5.0394225
STEP 10 ================================
prereg loss 0.642625 reg_l1 43.95584 reg_l2 26.403198
loss 5.038209
STEP 11 ================================
prereg loss 0.642139 reg_l1 43.94858 reg_l2 26.397324
loss 5.0369973
STEP 12 ================================
prereg loss 0.6416567 reg_l1 43.941334 reg_l2 26.391445
loss 5.0357904
STEP 13 ================================
prereg loss 0.64117676 reg_l1 43.93408 reg_l2 26.385592
loss 5.0345845
STEP 14 ================================
prereg loss 0.64069754 reg_l1 43.926826 reg_l2 26.379753
loss 5.03338
STEP 15 ================================
prereg loss 0.6402231 reg_l1 43.91958 reg_l2 26.37391
loss 5.032181
STEP 16 ================================
prereg loss 0.6397484 reg_l1 43.91234 reg_l2 26.368097
loss 5.0309825
STEP 17 ================================
prereg loss 0.6392748 reg_l1 43.905083 reg_l2 26.362282
loss 5.029783
STEP 18 ================================
prereg loss 0.638802 reg_l1 43.897835 reg_l2 26.35649
loss 5.0285854
STEP 19 ================================
prereg loss 0.6383298 reg_l1 43.890594 reg_l2 26.350706
loss 5.0273895
STEP 20 ================================
prereg loss 0.6378579 reg_l1 43.88333 reg_l2 26.344933
loss 5.026191
STEP 21 ================================
prereg loss 0.63738865 reg_l1 43.87609 reg_l2 26.339174
loss 5.0249977
STEP 22 ================================
prereg loss 0.6369213 reg_l1 43.868843 reg_l2 26.333435
loss 5.0238056
STEP 23 ================================
prereg loss 0.636455 reg_l1 43.861588 reg_l2 26.327696
loss 5.022614
STEP 24 ================================
prereg loss 0.63598776 reg_l1 43.85434 reg_l2 26.32197
loss 5.021422
STEP 25 ================================
prereg loss 0.6355204 reg_l1 43.84708 reg_l2 26.31625
loss 5.0202284
STEP 26 ================================
prereg loss 0.6350574 reg_l1 43.839817 reg_l2 26.310535
loss 5.019039
STEP 27 ================================
prereg loss 0.6345939 reg_l1 43.832565 reg_l2 26.304838
loss 5.0178504
STEP 28 ================================
prereg loss 0.63413227 reg_l1 43.825302 reg_l2 26.29914
loss 5.0166626
STEP 29 ================================
prereg loss 0.63367516 reg_l1 43.818047 reg_l2 26.293465
loss 5.01548
STEP 30 ================================
prereg loss 0.6332176 reg_l1 43.81076 reg_l2 26.28778
loss 5.0142937
STEP 31 ================================
prereg loss 0.63276476 reg_l1 43.80349 reg_l2 26.28211
loss 5.013114
STEP 32 ================================
prereg loss 0.63231325 reg_l1 43.796215 reg_l2 26.276455
loss 5.0119348
STEP 33 ================================
prereg loss 0.63186395 reg_l1 43.788918 reg_l2 26.270794
loss 5.010756
STEP 34 ================================
prereg loss 0.6314179 reg_l1 43.78165 reg_l2 26.265146
loss 5.009583
STEP 35 ================================
prereg loss 0.63097525 reg_l1 43.774353 reg_l2 26.259508
loss 5.0084105
STEP 36 ================================
prereg loss 0.6305332 reg_l1 43.76707 reg_l2 26.253878
loss 5.0072403
STEP 37 ================================
prereg loss 0.6300966 reg_l1 43.75976 reg_l2 26.24825
loss 5.0060725
STEP 38 ================================
prereg loss 0.62966233 reg_l1 43.75246 reg_l2 26.242636
loss 5.0049086
STEP 39 ================================
prereg loss 0.6292307 reg_l1 43.74515 reg_l2 26.237024
loss 5.0037456
STEP 40 ================================
prereg loss 0.62880105 reg_l1 43.737846 reg_l2 26.231424
loss 5.0025854
STEP 41 ================================
prereg loss 0.6283755 reg_l1 43.730526 reg_l2 26.22583
loss 5.001428
STEP 42 ================================
prereg loss 0.627952 reg_l1 43.72322 reg_l2 26.220255
loss 5.000274
STEP 43 ================================
prereg loss 0.62752783 reg_l1 43.71591 reg_l2 26.21467
loss 4.999119
STEP 44 ================================
prereg loss 0.6271105 reg_l1 43.708584 reg_l2 26.209108
loss 4.997969
STEP 45 ================================
prereg loss 0.6266952 reg_l1 43.70126 reg_l2 26.203552
loss 4.9968214
STEP 46 ================================
prereg loss 0.6262797 reg_l1 43.693954 reg_l2 26.198004
loss 4.9956756
STEP 47 ================================
prereg loss 0.6258674 reg_l1 43.68661 reg_l2 26.192461
loss 4.994529
STEP 48 ================================
prereg loss 0.6254569 reg_l1 43.679283 reg_l2 26.186935
loss 4.9933853
STEP 49 ================================
prereg loss 0.6250482 reg_l1 43.67196 reg_l2 26.181417
loss 4.9922442
STEP 50 ================================
prereg loss 0.62464184 reg_l1 43.664608 reg_l2 26.175898
loss 4.9911027
STEP 51 ================================
prereg loss 0.62423724 reg_l1 43.657284 reg_l2 26.170395
loss 4.9899654
STEP 52 ================================
prereg loss 0.6238347 reg_l1 43.649944 reg_l2 26.164907
loss 4.988829
STEP 53 ================================
prereg loss 0.6234335 reg_l1 43.642605 reg_l2 26.15942
loss 4.9876943
STEP 54 ================================
prereg loss 0.6230312 reg_l1 43.635273 reg_l2 26.153954
loss 4.9865584
STEP 55 ================================
prereg loss 0.6226343 reg_l1 43.62792 reg_l2 26.14849
loss 4.9854264
STEP 56 ================================
prereg loss 0.62223774 reg_l1 43.620583 reg_l2 26.143032
loss 4.984296
STEP 57 ================================
prereg loss 0.6218423 reg_l1 43.613228 reg_l2 26.13759
loss 4.9831653
STEP 58 ================================
prereg loss 0.62144643 reg_l1 43.60587 reg_l2 26.13216
loss 4.9820337
STEP 59 ================================
prereg loss 0.6210572 reg_l1 43.59852 reg_l2 26.126732
loss 4.980909
STEP 60 ================================
prereg loss 0.62066454 reg_l1 43.591167 reg_l2 26.121313
loss 4.9797816
STEP 61 ================================
prereg loss 0.6202756 reg_l1 43.5838 reg_l2 26.115904
loss 4.978656
STEP 62 ================================
prereg loss 0.6198863 reg_l1 43.57645 reg_l2 26.110504
loss 4.9775314
STEP 63 ================================
prereg loss 0.61950046 reg_l1 43.569073 reg_l2 26.105124
loss 4.976408
STEP 64 ================================
prereg loss 0.6191182 reg_l1 43.561718 reg_l2 26.099743
loss 4.9752903
STEP 65 ================================
prereg loss 0.6187348 reg_l1 43.554344 reg_l2 26.094372
loss 4.9741693
STEP 66 ================================
prereg loss 0.61835206 reg_l1 43.546967 reg_l2 26.089012
loss 4.9730487
STEP 67 ================================
prereg loss 0.61797315 reg_l1 43.5396 reg_l2 26.083652
loss 4.9719334
STEP 68 ================================
prereg loss 0.6175946 reg_l1 43.53222 reg_l2 26.078314
loss 4.9708166
STEP 69 ================================
prereg loss 0.6172193 reg_l1 43.524853 reg_l2 26.072985
loss 4.9697046
STEP 70 ================================
prereg loss 0.6168451 reg_l1 43.51747 reg_l2 26.067661
loss 4.968592
STEP 71 ================================
prereg loss 0.61647433 reg_l1 43.510094 reg_l2 26.06235
loss 4.9674835
STEP 72 ================================
prereg loss 0.61610144 reg_l1 43.50271 reg_l2 26.057041
loss 4.966372
STEP 73 ================================
prereg loss 0.61573267 reg_l1 43.495304 reg_l2 26.051743
loss 4.9652634
STEP 74 ================================
prereg loss 0.6153674 reg_l1 43.48793 reg_l2 26.046463
loss 4.9641604
STEP 75 ================================
prereg loss 0.6150011 reg_l1 43.48053 reg_l2 26.041176
loss 4.963054
STEP 76 ================================
prereg loss 0.6146366 reg_l1 43.473125 reg_l2 26.035912
loss 4.961949
STEP 77 ================================
prereg loss 0.61427486 reg_l1 43.465736 reg_l2 26.030645
loss 4.960849
STEP 78 ================================
prereg loss 0.6139138 reg_l1 43.458324 reg_l2 26.025398
loss 4.9597464
STEP 79 ================================
prereg loss 0.6135553 reg_l1 43.450928 reg_l2 26.020157
loss 4.958648
STEP 80 ================================
prereg loss 0.613199 reg_l1 43.44352 reg_l2 26.014923
loss 4.957551
STEP 81 ================================
prereg loss 0.6128427 reg_l1 43.436115 reg_l2 26.009697
loss 4.9564543
STEP 82 ================================
prereg loss 0.61248726 reg_l1 43.428703 reg_l2 26.004484
loss 4.9553576
STEP 83 ================================
prereg loss 0.61213565 reg_l1 43.421272 reg_l2 25.999277
loss 4.9542627
STEP 84 ================================
prereg loss 0.6117832 reg_l1 43.413864 reg_l2 25.99409
loss 4.9531693
STEP 85 ================================
prereg loss 0.61143297 reg_l1 43.40645 reg_l2 25.988903
loss 4.952078
STEP 86 ================================
prereg loss 0.61108583 reg_l1 43.399017 reg_l2 25.983727
loss 4.950988
STEP 87 ================================
prereg loss 0.61073786 reg_l1 43.39159 reg_l2 25.978554
loss 4.949897
STEP 88 ================================
prereg loss 0.6103914 reg_l1 43.384182 reg_l2 25.973402
loss 4.9488096
STEP 89 ================================
prereg loss 0.61004627 reg_l1 43.376743 reg_l2 25.96825
loss 4.947721
STEP 90 ================================
prereg loss 0.6097034 reg_l1 43.36931 reg_l2 25.96311
loss 4.9466343
STEP 91 ================================
prereg loss 0.6093607 reg_l1 43.361874 reg_l2 25.957983
loss 4.945548
STEP 92 ================================
prereg loss 0.6090223 reg_l1 43.354427 reg_l2 25.952871
loss 4.944465
STEP 93 ================================
prereg loss 0.6086823 reg_l1 43.347004 reg_l2 25.94775
loss 4.9433827
STEP 94 ================================
prereg loss 0.6083451 reg_l1 43.33956 reg_l2 25.942648
loss 4.9423013
STEP 95 ================================
prereg loss 0.6080079 reg_l1 43.33213 reg_l2 25.937567
loss 4.941221
STEP 96 ================================
prereg loss 0.60767233 reg_l1 43.324684 reg_l2 25.932482
loss 4.9401407
STEP 97 ================================
prereg loss 0.6073394 reg_l1 43.31724 reg_l2 25.92741
loss 4.9390635
STEP 98 ================================
prereg loss 0.6070061 reg_l1 43.309776 reg_l2 25.922354
loss 4.937984
STEP 99 ================================
prereg loss 0.6066745 reg_l1 43.302338 reg_l2 25.917295
loss 4.9369087
STEP 100 ================================
prereg loss 0.6063444 reg_l1 43.29488 reg_l2 25.912252
loss 4.9358325
2022-07-20T15:40:11.225

julia> interleaving_steps!(200, 40)
2022-07-20T15:40:26.689
STEP 1 ================================
prereg loss 0.6060151 reg_l1 43.287415 reg_l2 25.907217
loss 4.9347568
cutoff 0.11008332 network size 84
STEP 2 ================================
prereg loss 0.6064158 reg_l1 43.169872 reg_l2 25.890074
loss 4.9234033
STEP 3 ================================
prereg loss 0.6060767 reg_l1 43.163433 reg_l2 25.885296
loss 4.92242
STEP 4 ================================
prereg loss 0.60573816 reg_l1 43.156994 reg_l2 25.880548
loss 4.9214377
STEP 5 ================================
prereg loss 0.60539716 reg_l1 43.150555 reg_l2 25.875822
loss 4.9204526
STEP 6 ================================
prereg loss 0.6050565 reg_l1 43.144127 reg_l2 25.871105
loss 4.919469
STEP 7 ================================
prereg loss 0.60471874 reg_l1 43.137707 reg_l2 25.86641
loss 4.9184895
STEP 8 ================================
prereg loss 0.6043811 reg_l1 43.131283 reg_l2 25.861725
loss 4.9175096
STEP 9 ================================
prereg loss 0.6040421 reg_l1 43.124863 reg_l2 25.857056
loss 4.916528
STEP 10 ================================
prereg loss 0.6037102 reg_l1 43.118423 reg_l2 25.852388
loss 4.9155526
STEP 11 ================================
prereg loss 0.60337925 reg_l1 43.11199 reg_l2 25.84773
loss 4.9145784
STEP 12 ================================
prereg loss 0.60304815 reg_l1 43.105553 reg_l2 25.843071
loss 4.913604
STEP 13 ================================
prereg loss 0.6027215 reg_l1 43.09911 reg_l2 25.838427
loss 4.912633
STEP 14 ================================
prereg loss 0.6023975 reg_l1 43.09266 reg_l2 25.833773
loss 4.9116635
STEP 15 ================================
prereg loss 0.6020803 reg_l1 43.08619 reg_l2 25.829124
loss 4.9106994
STEP 16 ================================
prereg loss 0.601765 reg_l1 43.079704 reg_l2 25.824478
loss 4.9097357
STEP 17 ================================
prereg loss 0.60145235 reg_l1 43.07322 reg_l2 25.819828
loss 4.9087744
STEP 18 ================================
prereg loss 0.6011432 reg_l1 43.066727 reg_l2 25.815182
loss 4.907816
STEP 19 ================================
prereg loss 0.60083854 reg_l1 43.060226 reg_l2 25.810534
loss 4.9068613
STEP 20 ================================
prereg loss 0.6005353 reg_l1 43.053715 reg_l2 25.805887
loss 4.905907
STEP 21 ================================
prereg loss 0.60023725 reg_l1 43.04721 reg_l2 25.80124
loss 4.9049587
STEP 22 ================================
prereg loss 0.59994024 reg_l1 43.040665 reg_l2 25.79659
loss 4.904007
STEP 23 ================================
prereg loss 0.59964657 reg_l1 43.03414 reg_l2 25.791962
loss 4.903061
STEP 24 ================================
prereg loss 0.5993548 reg_l1 43.027603 reg_l2 25.78732
loss 4.9021153
STEP 25 ================================
prereg loss 0.5990669 reg_l1 43.021065 reg_l2 25.782696
loss 4.901173
STEP 26 ================================
prereg loss 0.59877944 reg_l1 43.01452 reg_l2 25.77807
loss 4.9002314
STEP 27 ================================
prereg loss 0.5984935 reg_l1 43.00797 reg_l2 25.773445
loss 4.8992906
STEP 28 ================================
prereg loss 0.59820795 reg_l1 43.001423 reg_l2 25.768837
loss 4.8983502
STEP 29 ================================
prereg loss 0.59792477 reg_l1 42.994865 reg_l2 25.764236
loss 4.8974113
STEP 30 ================================
prereg loss 0.59764236 reg_l1 42.988323 reg_l2 25.759655
loss 4.896475
STEP 31 ================================
prereg loss 0.5973589 reg_l1 42.98177 reg_l2 25.75506
loss 4.895536
STEP 32 ================================
prereg loss 0.5970771 reg_l1 42.975204 reg_l2 25.750488
loss 4.8945975
STEP 33 ================================
prereg loss 0.5967958 reg_l1 42.96867 reg_l2 25.745922
loss 4.8936625
STEP 34 ================================
prereg loss 0.5965125 reg_l1 42.962116 reg_l2 25.741371
loss 4.892724
STEP 35 ================================
prereg loss 0.5962313 reg_l1 42.955574 reg_l2 25.736832
loss 4.891789
STEP 36 ================================
prereg loss 0.5959493 reg_l1 42.949005 reg_l2 25.73229
loss 4.8908496
STEP 37 ================================
prereg loss 0.5956669 reg_l1 42.94246 reg_l2 25.727772
loss 4.889913
STEP 38 ================================
prereg loss 0.5953863 reg_l1 42.93591 reg_l2 25.723248
loss 4.8889775
STEP 39 ================================
prereg loss 0.59510326 reg_l1 42.929356 reg_l2 25.718744
loss 4.888039
STEP 40 ================================
prereg loss 0.5948206 reg_l1 42.922806 reg_l2 25.71425
loss 4.887101
STEP 41 ================================
prereg loss 0.59453756 reg_l1 42.916252 reg_l2 25.709757
loss 4.886163
cutoff 0.07480596 network size 83
STEP 42 ================================
prereg loss 0.5942539 reg_l1 42.834892 reg_l2 25.69968
loss 4.8777432
STEP 43 ================================
prereg loss 0.59397423 reg_l1 42.829258 reg_l2 25.69535
loss 4.8769
STEP 44 ================================
prereg loss 0.5936933 reg_l1 42.823624 reg_l2 25.691021
loss 4.8760557
STEP 45 ================================
prereg loss 0.5934101 reg_l1 42.818016 reg_l2 25.686693
loss 4.8752117
STEP 46 ================================
prereg loss 0.5931331 reg_l1 42.812378 reg_l2 25.682377
loss 4.874371
STEP 47 ================================
prereg loss 0.59285146 reg_l1 42.80674 reg_l2 25.678066
loss 4.8735256
STEP 48 ================================
prereg loss 0.59257203 reg_l1 42.801098 reg_l2 25.67376
loss 4.872682
STEP 49 ================================
prereg loss 0.59229517 reg_l1 42.795467 reg_l2 25.669453
loss 4.871842
STEP 50 ================================
prereg loss 0.59201896 reg_l1 42.789825 reg_l2 25.665154
loss 4.8710017
STEP 51 ================================
prereg loss 0.59174335 reg_l1 42.784184 reg_l2 25.660868
loss 4.870162
STEP 52 ================================
prereg loss 0.5914686 reg_l1 42.778534 reg_l2 25.656576
loss 4.869322
STEP 53 ================================
prereg loss 0.59119457 reg_l1 42.77288 reg_l2 25.652288
loss 4.8684826
STEP 54 ================================
prereg loss 0.5909219 reg_l1 42.76723 reg_l2 25.648014
loss 4.8676453
STEP 55 ================================
prereg loss 0.5906512 reg_l1 42.76157 reg_l2 25.643742
loss 4.866808
STEP 56 ================================
prereg loss 0.5903802 reg_l1 42.75592 reg_l2 25.639473
loss 4.8659725
STEP 57 ================================
prereg loss 0.590109 reg_l1 42.750263 reg_l2 25.635195
loss 4.865135
STEP 58 ================================
prereg loss 0.5898412 reg_l1 42.744587 reg_l2 25.630941
loss 4.8643003
STEP 59 ================================
prereg loss 0.58957356 reg_l1 42.73893 reg_l2 25.62669
loss 4.8634663
STEP 60 ================================
prereg loss 0.58930695 reg_l1 42.733273 reg_l2 25.622444
loss 4.862634
STEP 61 ================================
prereg loss 0.58903897 reg_l1 42.72758 reg_l2 25.618196
loss 4.861797
STEP 62 ================================
prereg loss 0.58877426 reg_l1 42.721924 reg_l2 25.61395
loss 4.8609667
STEP 63 ================================
prereg loss 0.5885095 reg_l1 42.716244 reg_l2 25.609728
loss 4.860134
STEP 64 ================================
prereg loss 0.58824635 reg_l1 42.710567 reg_l2 25.6055
loss 4.859303
STEP 65 ================================
prereg loss 0.58798176 reg_l1 42.7049 reg_l2 25.601278
loss 4.858472
STEP 66 ================================
prereg loss 0.58771896 reg_l1 42.69921 reg_l2 25.597061
loss 4.8576403
STEP 67 ================================
prereg loss 0.5874559 reg_l1 42.69354 reg_l2 25.592854
loss 4.8568096
STEP 68 ================================
prereg loss 0.5871942 reg_l1 42.687847 reg_l2 25.588652
loss 4.855979
STEP 69 ================================
prereg loss 0.5869311 reg_l1 42.68217 reg_l2 25.58446
loss 4.8551483
STEP 70 ================================
prereg loss 0.5866705 reg_l1 42.676495 reg_l2 25.580275
loss 4.85432
STEP 71 ================================
prereg loss 0.58640903 reg_l1 42.670803 reg_l2 25.57609
loss 4.8534894
STEP 72 ================================
prereg loss 0.5861492 reg_l1 42.66512 reg_l2 25.571915
loss 4.852661
STEP 73 ================================
prereg loss 0.5858879 reg_l1 42.659435 reg_l2 25.567743
loss 4.8518314
STEP 74 ================================
prereg loss 0.58563024 reg_l1 42.65374 reg_l2 25.563583
loss 4.8510046
STEP 75 ================================
prereg loss 0.5853682 reg_l1 42.648056 reg_l2 25.559422
loss 4.850174
STEP 76 ================================
prereg loss 0.5851081 reg_l1 42.642372 reg_l2 25.555273
loss 4.8493457
STEP 77 ================================
prereg loss 0.5848518 reg_l1 42.636673 reg_l2 25.551134
loss 4.8485193
STEP 78 ================================
prereg loss 0.5845909 reg_l1 42.630978 reg_l2 25.546995
loss 4.8476887
STEP 79 ================================
prereg loss 0.584333 reg_l1 42.625282 reg_l2 25.542864
loss 4.8468614
STEP 80 ================================
prereg loss 0.5840749 reg_l1 42.619587 reg_l2 25.538744
loss 4.8460336
STEP 81 ================================
prereg loss 0.58381814 reg_l1 42.613895 reg_l2 25.53463
loss 4.8452077
cutoff 0.15453778 network size 82
STEP 82 ================================
prereg loss 0.5835608 reg_l1 42.453644 reg_l2 25.506624
loss 4.8289256
STEP 83 ================================
prereg loss 0.5833038 reg_l1 42.449165 reg_l2 25.502895
loss 4.8282204
STEP 84 ================================
prereg loss 0.58304834 reg_l1 42.444683 reg_l2 25.49917
loss 4.8275166
STEP 85 ================================
prereg loss 0.5827948 reg_l1 42.440197 reg_l2 25.495451
loss 4.8268147
STEP 86 ================================
prereg loss 0.5825377 reg_l1 42.435703 reg_l2 25.49173
loss 4.826108
STEP 87 ================================
prereg loss 0.58228165 reg_l1 42.4312 reg_l2 25.488016
loss 4.8254013
STEP 88 ================================
prereg loss 0.5820281 reg_l1 42.42672 reg_l2 25.484304
loss 4.8247
STEP 89 ================================
prereg loss 0.581774 reg_l1 42.422215 reg_l2 25.480585
loss 4.8239956
STEP 90 ================================
prereg loss 0.5815213 reg_l1 42.41772 reg_l2 25.476877
loss 4.8232937
STEP 91 ================================
prereg loss 0.5812653 reg_l1 42.413227 reg_l2 25.473171
loss 4.8225884
STEP 92 ================================
prereg loss 0.5810143 reg_l1 42.40873 reg_l2 25.469471
loss 4.821887
STEP 93 ================================
prereg loss 0.58076197 reg_l1 42.404232 reg_l2 25.465773
loss 4.821185
STEP 94 ================================
prereg loss 0.5805099 reg_l1 42.39973 reg_l2 25.462086
loss 4.820483
STEP 95 ================================
prereg loss 0.5802597 reg_l1 42.395226 reg_l2 25.458395
loss 4.8197823
STEP 96 ================================
prereg loss 0.580008 reg_l1 42.39072 reg_l2 25.4547
loss 4.8190804
STEP 97 ================================
prereg loss 0.5797572 reg_l1 42.38621 reg_l2 25.451023
loss 4.8183784
STEP 98 ================================
prereg loss 0.57950884 reg_l1 42.381695 reg_l2 25.447344
loss 4.8176785
STEP 99 ================================
prereg loss 0.57925934 reg_l1 42.377186 reg_l2 25.443665
loss 4.816978
STEP 100 ================================
prereg loss 0.5790092 reg_l1 42.372677 reg_l2 25.439995
loss 4.816277
STEP 101 ================================
prereg loss 0.5787594 reg_l1 42.368168 reg_l2 25.436321
loss 4.815576
STEP 102 ================================
prereg loss 0.5785092 reg_l1 42.36366 reg_l2 25.432653
loss 4.814875
STEP 103 ================================
prereg loss 0.5782617 reg_l1 42.359127 reg_l2 25.428995
loss 4.8141747
STEP 104 ================================
prereg loss 0.57801193 reg_l1 42.354607 reg_l2 25.425335
loss 4.8134727
STEP 105 ================================
prereg loss 0.5777639 reg_l1 42.350082 reg_l2 25.42169
loss 4.8127723
STEP 106 ================================
prereg loss 0.57751584 reg_l1 42.345562 reg_l2 25.418032
loss 4.812072
STEP 107 ================================
prereg loss 0.57726926 reg_l1 42.34104 reg_l2 25.414383
loss 4.811373
STEP 108 ================================
prereg loss 0.5770223 reg_l1 42.33653 reg_l2 25.410742
loss 4.8106756
STEP 109 ================================
prereg loss 0.57677436 reg_l1 42.331993 reg_l2 25.407106
loss 4.8099737
STEP 110 ================================
prereg loss 0.57652795 reg_l1 42.32746 reg_l2 25.40346
loss 4.809274
STEP 111 ================================
prereg loss 0.5762798 reg_l1 42.322937 reg_l2 25.399832
loss 4.8085732
STEP 112 ================================
prereg loss 0.57603383 reg_l1 42.318413 reg_l2 25.396202
loss 4.8078756
STEP 113 ================================
prereg loss 0.57578766 reg_l1 42.313873 reg_l2 25.392586
loss 4.807175
STEP 114 ================================
prereg loss 0.575541 reg_l1 42.30933 reg_l2 25.38896
loss 4.806474
STEP 115 ================================
prereg loss 0.5752961 reg_l1 42.30481 reg_l2 25.385338
loss 4.805777
STEP 116 ================================
prereg loss 0.5750498 reg_l1 42.30027 reg_l2 25.38173
loss 4.805077
STEP 117 ================================
prereg loss 0.57480603 reg_l1 42.295746 reg_l2 25.378122
loss 4.804381
STEP 118 ================================
prereg loss 0.57455975 reg_l1 42.29119 reg_l2 25.37452
loss 4.803679
STEP 119 ================================
prereg loss 0.57431245 reg_l1 42.286648 reg_l2 25.370918
loss 4.8029776
STEP 120 ================================
prereg loss 0.5740697 reg_l1 42.282116 reg_l2 25.367323
loss 4.8022814
STEP 121 ================================
prereg loss 0.5738249 reg_l1 42.277565 reg_l2 25.363724
loss 4.8015814
cutoff 0.20166622 network size 81
STEP 122 ================================
prereg loss 0.5735825 reg_l1 42.071354 reg_l2 25.319475
loss 4.7807183
STEP 123 ================================
prereg loss 0.57333744 reg_l1 42.06706 reg_l2 25.315983
loss 4.7800436
STEP 124 ================================
prereg loss 0.5730936 reg_l1 42.06277 reg_l2 25.312502
loss 4.779371
STEP 125 ================================
prereg loss 0.57284856 reg_l1 42.058456 reg_l2 25.309023
loss 4.778694
STEP 126 ================================
prereg loss 0.5726057 reg_l1 42.05416 reg_l2 25.30555
loss 4.778022
STEP 127 ================================
prereg loss 0.5723637 reg_l1 42.049843 reg_l2 25.302076
loss 4.777348
STEP 128 ================================
prereg loss 0.57212055 reg_l1 42.045547 reg_l2 25.298607
loss 4.7766757
STEP 129 ================================
prereg loss 0.57187647 reg_l1 42.041237 reg_l2 25.295145
loss 4.7760005
STEP 130 ================================
prereg loss 0.5716336 reg_l1 42.03693 reg_l2 25.291695
loss 4.7753267
STEP 131 ================================
prereg loss 0.571391 reg_l1 42.032627 reg_l2 25.288239
loss 4.774654
STEP 132 ================================
prereg loss 0.5711485 reg_l1 42.028313 reg_l2 25.284782
loss 4.7739797
STEP 133 ================================
prereg loss 0.5709076 reg_l1 42.02401 reg_l2 25.281338
loss 4.7733088
STEP 134 ================================
prereg loss 0.57066476 reg_l1 42.01968 reg_l2 25.27789
loss 4.772633
STEP 135 ================================
prereg loss 0.5704234 reg_l1 42.015377 reg_l2 25.274454
loss 4.771961
STEP 136 ================================
prereg loss 0.57018346 reg_l1 42.01106 reg_l2 25.271019
loss 4.7712893
STEP 137 ================================
prereg loss 0.56994104 reg_l1 42.00674 reg_l2 25.267582
loss 4.770615
STEP 138 ================================
prereg loss 0.5696981 reg_l1 42.002426 reg_l2 25.264158
loss 4.7699404
STEP 139 ================================
prereg loss 0.56945705 reg_l1 41.998108 reg_l2 25.26073
loss 4.769268
STEP 140 ================================
prereg loss 0.56921715 reg_l1 41.993782 reg_l2 25.257313
loss 4.7685957
STEP 141 ================================
prereg loss 0.5689751 reg_l1 41.989464 reg_l2 25.253891
loss 4.7679214
STEP 142 ================================
prereg loss 0.5687336 reg_l1 41.98514 reg_l2 25.25049
loss 4.767248
STEP 143 ================================
prereg loss 0.56849086 reg_l1 41.98082 reg_l2 25.247084
loss 4.766573
STEP 144 ================================
prereg loss 0.5682518 reg_l1 41.976494 reg_l2 25.243675
loss 4.765901
STEP 145 ================================
prereg loss 0.5680093 reg_l1 41.972157 reg_l2 25.24027
loss 4.765225
STEP 146 ================================
prereg loss 0.56777 reg_l1 41.96784 reg_l2 25.236877
loss 4.764554
STEP 147 ================================
prereg loss 0.5675298 reg_l1 41.963512 reg_l2 25.233492
loss 4.763881
STEP 148 ================================
prereg loss 0.5672884 reg_l1 41.959175 reg_l2 25.2301
loss 4.763206
STEP 149 ================================
prereg loss 0.5670458 reg_l1 41.954845 reg_l2 25.226711
loss 4.7625303
STEP 150 ================================
prereg loss 0.5668076 reg_l1 41.950516 reg_l2 25.223337
loss 4.7618594
STEP 151 ================================
prereg loss 0.56656694 reg_l1 41.946182 reg_l2 25.219963
loss 4.761185
STEP 152 ================================
prereg loss 0.56632686 reg_l1 41.941845 reg_l2 25.216587
loss 4.7605114
STEP 153 ================================
prereg loss 0.5660857 reg_l1 41.93751 reg_l2 25.21322
loss 4.759837
STEP 154 ================================
prereg loss 0.5658475 reg_l1 41.93317 reg_l2 25.20986
loss 4.7591643
STEP 155 ================================
prereg loss 0.56560725 reg_l1 41.92884 reg_l2 25.206505
loss 4.758491
STEP 156 ================================
prereg loss 0.5653666 reg_l1 41.924488 reg_l2 25.20315
loss 4.757816
STEP 157 ================================
prereg loss 0.5651266 reg_l1 41.920147 reg_l2 25.199791
loss 4.757141
STEP 158 ================================
prereg loss 0.56488633 reg_l1 41.915806 reg_l2 25.196447
loss 4.756467
STEP 159 ================================
prereg loss 0.5646068 reg_l1 41.911457 reg_l2 25.193102
loss 4.7557526
STEP 160 ================================
prereg loss 0.56429917 reg_l1 41.90723 reg_l2 25.18991
loss 4.755022
STEP 161 ================================
prereg loss 0.5639665 reg_l1 41.903103 reg_l2 25.186846
loss 4.754277
cutoff 0.21000245 network size 80
STEP 162 ================================
prereg loss 0.56361157 reg_l1 41.68908 reg_l2 25.139807
loss 4.7325196
STEP 163 ================================
prereg loss 0.5632399 reg_l1 41.68544 reg_l2 25.137114
loss 4.7317843
STEP 164 ================================
prereg loss 0.56285506 reg_l1 41.68188 reg_l2 25.134506
loss 4.7310433
STEP 165 ================================
prereg loss 0.56245476 reg_l1 41.678368 reg_l2 25.131996
loss 4.7302914
STEP 166 ================================
prereg loss 0.5620476 reg_l1 41.67493 reg_l2 25.129555
loss 4.729541
STEP 167 ================================
prereg loss 0.56162995 reg_l1 41.671528 reg_l2 25.127186
loss 4.7287827
STEP 168 ================================
prereg loss 0.5612076 reg_l1 41.66818 reg_l2 25.12488
loss 4.728026
STEP 169 ================================
prereg loss 0.56078297 reg_l1 41.664852 reg_l2 25.122616
loss 4.727268
STEP 170 ================================
prereg loss 0.56035596 reg_l1 41.661575 reg_l2 25.120403
loss 4.726514
STEP 171 ================================
prereg loss 0.55992806 reg_l1 41.658314 reg_l2 25.11823
loss 4.7257595
STEP 172 ================================
prereg loss 0.5595027 reg_l1 41.655064 reg_l2 25.116076
loss 4.725009
STEP 173 ================================
prereg loss 0.55907816 reg_l1 41.65183 reg_l2 25.113953
loss 4.7242613
STEP 174 ================================
prereg loss 0.5586592 reg_l1 41.648605 reg_l2 25.111843
loss 4.72352
STEP 175 ================================
prereg loss 0.5582425 reg_l1 41.645386 reg_l2 25.109747
loss 4.722781
STEP 176 ================================
prereg loss 0.55782944 reg_l1 41.64215 reg_l2 25.107664
loss 4.7220445
STEP 177 ================================
prereg loss 0.55742043 reg_l1 41.638927 reg_l2 25.10558
loss 4.721313
STEP 178 ================================
prereg loss 0.55701977 reg_l1 41.6357 reg_l2 25.103497
loss 4.7205896
STEP 179 ================================
prereg loss 0.55662113 reg_l1 41.63245 reg_l2 25.101412
loss 4.7198663
STEP 180 ================================
prereg loss 0.5562268 reg_l1 41.629177 reg_l2 25.099314
loss 4.7191443
STEP 181 ================================
prereg loss 0.5558414 reg_l1 41.625908 reg_l2 25.097204
loss 4.7184324
STEP 182 ================================
prereg loss 0.5554582 reg_l1 41.622597 reg_l2 25.095076
loss 4.7177176
STEP 183 ================================
prereg loss 0.5550807 reg_l1 41.61928 reg_l2 25.09293
loss 4.717009
STEP 184 ================================
prereg loss 0.5547107 reg_l1 41.615925 reg_l2 25.090761
loss 4.7163033
STEP 185 ================================
prereg loss 0.55434614 reg_l1 41.612564 reg_l2 25.088587
loss 4.7156024
STEP 186 ================================
prereg loss 0.5539864 reg_l1 41.609154 reg_l2 25.08637
loss 4.714902
STEP 187 ================================
prereg loss 0.55363274 reg_l1 41.60572 reg_l2 25.084133
loss 4.714205
STEP 188 ================================
prereg loss 0.5532847 reg_l1 41.602264 reg_l2 25.081873
loss 4.713511
STEP 189 ================================
prereg loss 0.5529403 reg_l1 41.598785 reg_l2 25.079586
loss 4.712819
STEP 190 ================================
prereg loss 0.5526034 reg_l1 41.59527 reg_l2 25.077272
loss 4.71213
STEP 191 ================================
prereg loss 0.55227137 reg_l1 41.591724 reg_l2 25.07493
loss 4.711444
STEP 192 ================================
prereg loss 0.5519481 reg_l1 41.58816 reg_l2 25.07256
loss 4.7107644
STEP 193 ================================
prereg loss 0.55162823 reg_l1 41.584553 reg_l2 25.07016
loss 4.7100835
STEP 194 ================================
prereg loss 0.55131316 reg_l1 41.58093 reg_l2 25.067738
loss 4.709406
STEP 195 ================================
prereg loss 0.5510061 reg_l1 41.577282 reg_l2 25.065296
loss 4.7087345
STEP 196 ================================
prereg loss 0.5507025 reg_l1 41.57358 reg_l2 25.062819
loss 4.7080607
STEP 197 ================================
prereg loss 0.5504047 reg_l1 41.569874 reg_l2 25.060314
loss 4.707392
STEP 198 ================================
prereg loss 0.5501136 reg_l1 41.56613 reg_l2 25.05778
loss 4.706727
STEP 199 ================================
prereg loss 0.54982466 reg_l1 41.56236 reg_l2 25.055223
loss 4.706061
STEP 200 ================================
prereg loss 0.54954284 reg_l1 41.558575 reg_l2 25.052643
loss 4.7054005
2022-07-20T15:53:35.661

julia> serialize("cf-80-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-80-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T15:59:21.357
STEP 1 ================================
prereg loss 0.54926574 reg_l1 41.55476 reg_l2 25.050053
loss 4.704742
cutoff 0.19855286 network size 79
STEP 2 ================================
prereg loss 0.6389486 reg_l1 41.352364 reg_l2 25.008001
loss 4.7741847
STEP 3 ================================
prereg loss 0.63425505 reg_l1 41.35076 reg_l2 25.007645
loss 4.769331
STEP 4 ================================
prereg loss 0.62622696 reg_l1 41.35074 reg_l2 25.009113
loss 4.761301
STEP 5 ================================
prereg loss 0.6158625 reg_l1 41.35207 reg_l2 25.012117
loss 4.7510695
STEP 6 ================================
prereg loss 0.6042095 reg_l1 41.354458 reg_l2 25.016348
loss 4.7396555
STEP 7 ================================
prereg loss 0.5922566 reg_l1 41.357643 reg_l2 25.021482
loss 4.728021
STEP 8 ================================
prereg loss 0.5808506 reg_l1 41.361355 reg_l2 25.027208
loss 4.716986
STEP 9 ================================
prereg loss 0.57062304 reg_l1 41.36534 reg_l2 25.033232
loss 4.707157
STEP 10 ================================
prereg loss 0.5619799 reg_l1 41.369354 reg_l2 25.03926
loss 4.6989155
STEP 11 ================================
prereg loss 0.5550976 reg_l1 41.373154 reg_l2 25.045044
loss 4.692413
STEP 12 ================================
prereg loss 0.5499438 reg_l1 41.37658 reg_l2 25.05036
loss 4.687602
STEP 13 ================================
prereg loss 0.5463349 reg_l1 41.37944 reg_l2 25.055006
loss 4.684279
STEP 14 ================================
prereg loss 0.54390234 reg_l1 41.38161 reg_l2 25.058842
loss 4.6820636
STEP 15 ================================
prereg loss 0.5396098 reg_l1 41.38301 reg_l2 25.06176
loss 4.6779113
STEP 16 ================================
prereg loss 0.5357149 reg_l1 41.384033 reg_l2 25.064167
loss 4.6741185
STEP 17 ================================
prereg loss 0.5322272 reg_l1 41.38459 reg_l2 25.065926
loss 4.6706862
STEP 18 ================================
prereg loss 0.52900773 reg_l1 41.38457 reg_l2 25.066986
loss 4.667465
STEP 19 ================================
prereg loss 0.5259197 reg_l1 41.383945 reg_l2 25.06728
loss 4.6643143
STEP 20 ================================
prereg loss 0.5227768 reg_l1 41.382706 reg_l2 25.066807
loss 4.6610475
STEP 21 ================================
prereg loss 0.519631 reg_l1 41.38083 reg_l2 25.065577
loss 4.657714
STEP 22 ================================
prereg loss 0.51645863 reg_l1 41.37835 reg_l2 25.063614
loss 4.6542935
STEP 23 ================================
prereg loss 0.5132834 reg_l1 41.375313 reg_l2 25.060976
loss 4.6508145
STEP 24 ================================
prereg loss 0.5101453 reg_l1 41.371758 reg_l2 25.057743
loss 4.647321
STEP 25 ================================
prereg loss 0.50710505 reg_l1 41.367767 reg_l2 25.053982
loss 4.643882
STEP 26 ================================
prereg loss 0.5042261 reg_l1 41.363377 reg_l2 25.049805
loss 4.640564
STEP 27 ================================
prereg loss 0.5015601 reg_l1 41.358704 reg_l2 25.045277
loss 4.6374307
STEP 28 ================================
prereg loss 0.49914876 reg_l1 41.353832 reg_l2 25.040531
loss 4.634532
STEP 29 ================================
prereg loss 0.49701336 reg_l1 41.348824 reg_l2 25.035652
loss 4.631896
STEP 30 ================================
prereg loss 0.49515688 reg_l1 41.343758 reg_l2 25.030739
loss 4.629533
STEP 31 ================================
prereg loss 0.4935545 reg_l1 41.338703 reg_l2 25.025858
loss 4.627425
STEP 32 ================================
prereg loss 0.49217537 reg_l1 41.333733 reg_l2 25.021091
loss 4.625549
STEP 33 ================================
prereg loss 0.4909779 reg_l1 41.328903 reg_l2 25.016506
loss 4.623868
STEP 34 ================================
prereg loss 0.4899101 reg_l1 41.32424 reg_l2 25.012136
loss 4.6223345
STEP 35 ================================
prereg loss 0.4889292 reg_l1 41.31977 reg_l2 25.008028
loss 4.6209064
STEP 36 ================================
prereg loss 0.48799926 reg_l1 41.315514 reg_l2 25.00419
loss 4.6195507
STEP 37 ================================
prereg loss 0.48708615 reg_l1 41.31151 reg_l2 25.000635
loss 4.618237
STEP 38 ================================
prereg loss 0.48617387 reg_l1 41.30772 reg_l2 24.997356
loss 4.616946
STEP 39 ================================
prereg loss 0.48525935 reg_l1 41.304142 reg_l2 24.994335
loss 4.615674
STEP 40 ================================
prereg loss 0.48434713 reg_l1 41.300747 reg_l2 24.991568
loss 4.6144223
STEP 41 ================================
prereg loss 0.48345074 reg_l1 41.297516 reg_l2 24.988983
loss 4.6132026
cutoff 0.20541437 network size 78
STEP 42 ================================
prereg loss 0.4807143 reg_l1 41.088985 reg_l2 24.944368
loss 4.589613
STEP 43 ================================
prereg loss 0.47991225 reg_l1 41.08626 reg_l2 24.942171
loss 4.5885386
STEP 44 ================================
prereg loss 0.47917786 reg_l1 41.08357 reg_l2 24.94004
loss 4.587535
STEP 45 ================================
prereg loss 0.47853294 reg_l1 41.08088 reg_l2 24.937922
loss 4.586621
STEP 46 ================================
prereg loss 0.4779876 reg_l1 41.078163 reg_l2 24.935774
loss 4.585804
STEP 47 ================================
prereg loss 0.4775497 reg_l1 41.07536 reg_l2 24.933556
loss 4.5850854
STEP 48 ================================
prereg loss 0.47721878 reg_l1 41.072445 reg_l2 24.931244
loss 4.584463
STEP 49 ================================
prereg loss 0.47694582 reg_l1 41.069427 reg_l2 24.928818
loss 4.5838885
STEP 50 ================================
prereg loss 0.47676706 reg_l1 41.066277 reg_l2 24.926239
loss 4.583395
STEP 51 ================================
prereg loss 0.4766821 reg_l1 41.062973 reg_l2 24.923506
loss 4.5829797
STEP 52 ================================
prereg loss 0.47667527 reg_l1 41.059505 reg_l2 24.920597
loss 4.5826263
STEP 53 ================================
prereg loss 0.47673497 reg_l1 41.05586 reg_l2 24.917488
loss 4.582321
STEP 54 ================================
prereg loss 0.47685063 reg_l1 41.05203 reg_l2 24.914198
loss 4.5820537
STEP 55 ================================
prereg loss 0.4770115 reg_l1 41.048027 reg_l2 24.910732
loss 4.5818143
STEP 56 ================================
prereg loss 0.47721025 reg_l1 41.04386 reg_l2 24.907076
loss 4.5815964
STEP 57 ================================
prereg loss 0.47743887 reg_l1 41.039543 reg_l2 24.903257
loss 4.5813932
STEP 58 ================================
prereg loss 0.4776931 reg_l1 41.03508 reg_l2 24.899305
loss 4.581201
STEP 59 ================================
prereg loss 0.47796544 reg_l1 41.030483 reg_l2 24.895214
loss 4.5810137
STEP 60 ================================
prereg loss 0.47825256 reg_l1 41.025784 reg_l2 24.89102
loss 4.580831
STEP 61 ================================
prereg loss 0.47855598 reg_l1 41.021004 reg_l2 24.886742
loss 4.5806565
STEP 62 ================================
prereg loss 0.47886547 reg_l1 41.016155 reg_l2 24.88241
loss 4.580481
STEP 63 ================================
prereg loss 0.4791829 reg_l1 41.011257 reg_l2 24.878033
loss 4.5803084
STEP 64 ================================
prereg loss 0.47950226 reg_l1 41.006348 reg_l2 24.873636
loss 4.5801373
STEP 65 ================================
prereg loss 0.4798198 reg_l1 41.00142 reg_l2 24.869246
loss 4.579962
STEP 66 ================================
prereg loss 0.48013264 reg_l1 40.99649 reg_l2 24.864883
loss 4.5797815
STEP 67 ================================
prereg loss 0.48043653 reg_l1 40.9916 reg_l2 24.86055
loss 4.5795965
STEP 68 ================================
prereg loss 0.48072267 reg_l1 40.986725 reg_l2 24.856262
loss 4.5793953
STEP 69 ================================
prereg loss 0.48099434 reg_l1 40.981873 reg_l2 24.85203
loss 4.5791817
STEP 70 ================================
prereg loss 0.4812466 reg_l1 40.97709 reg_l2 24.847874
loss 4.5789557
STEP 71 ================================
prereg loss 0.48147863 reg_l1 40.97234 reg_l2 24.843775
loss 4.578713
STEP 72 ================================
prereg loss 0.48169038 reg_l1 40.96766 reg_l2 24.839739
loss 4.5784564
STEP 73 ================================
prereg loss 0.4818805 reg_l1 40.963013 reg_l2 24.835783
loss 4.578182
STEP 74 ================================
prereg loss 0.48205012 reg_l1 40.958424 reg_l2 24.831882
loss 4.5778923
STEP 75 ================================
prereg loss 0.48219967 reg_l1 40.95386 reg_l2 24.828045
loss 4.5775857
STEP 76 ================================
prereg loss 0.48233703 reg_l1 40.949337 reg_l2 24.824257
loss 4.577271
STEP 77 ================================
prereg loss 0.4824581 reg_l1 40.944855 reg_l2 24.820517
loss 4.576944
STEP 78 ================================
prereg loss 0.48257056 reg_l1 40.940384 reg_l2 24.816805
loss 4.576609
STEP 79 ================================
prereg loss 0.48267266 reg_l1 40.935944 reg_l2 24.813137
loss 4.5762672
STEP 80 ================================
prereg loss 0.48276898 reg_l1 40.931515 reg_l2 24.809484
loss 4.5759206
STEP 81 ================================
prereg loss 0.4828635 reg_l1 40.927086 reg_l2 24.80585
loss 4.575572
cutoff 0.21850015 network size 77
STEP 82 ================================
prereg loss 0.5282839 reg_l1 40.704132 reg_l2 24.754484
loss 4.598697
STEP 83 ================================
prereg loss 0.52612096 reg_l1 40.701214 reg_l2 24.752417
loss 4.5962424
STEP 84 ================================
prereg loss 0.5224138 reg_l1 40.69931 reg_l2 24.751589
loss 4.5923448
STEP 85 ================================
prereg loss 0.5176517 reg_l1 40.698273 reg_l2 24.751785
loss 4.587479
STEP 86 ================================
prereg loss 0.5123051 reg_l1 40.697903 reg_l2 24.75279
loss 4.5820956
STEP 87 ================================
prereg loss 0.50574857 reg_l1 40.698048 reg_l2 24.75441
loss 4.5755534
STEP 88 ================================
prereg loss 0.49828702 reg_l1 40.6989 reg_l2 24.756859
loss 4.568177
STEP 89 ================================
prereg loss 0.49110115 reg_l1 40.700253 reg_l2 24.759857
loss 4.5611267
STEP 90 ================================
prereg loss 0.48462555 reg_l1 40.701843 reg_l2 24.76314
loss 4.5548096
STEP 91 ================================
prereg loss 0.4791291 reg_l1 40.703506 reg_l2 24.766468
loss 4.54948
STEP 92 ================================
prereg loss 0.47467822 reg_l1 40.70501 reg_l2 24.769608
loss 4.545179
STEP 93 ================================
prereg loss 0.47123316 reg_l1 40.706215 reg_l2 24.772371
loss 4.541855
STEP 94 ================================
prereg loss 0.46871886 reg_l1 40.706974 reg_l2 24.774597
loss 4.5394163
STEP 95 ================================
prereg loss 0.4669282 reg_l1 40.707153 reg_l2 24.776142
loss 4.5376434
STEP 96 ================================
prereg loss 0.4656323 reg_l1 40.706703 reg_l2 24.776926
loss 4.536303
STEP 97 ================================
prereg loss 0.464617 reg_l1 40.705563 reg_l2 24.77688
loss 4.5351734
STEP 98 ================================
prereg loss 0.46369827 reg_l1 40.703682 reg_l2 24.775982
loss 4.5340667
STEP 99 ================================
prereg loss 0.4627595 reg_l1 40.70108 reg_l2 24.774244
loss 4.5328674
STEP 100 ================================
prereg loss 0.46173802 reg_l1 40.697803 reg_l2 24.7717
loss 4.5315185
STEP 101 ================================
prereg loss 0.4606305 reg_l1 40.693882 reg_l2 24.768415
loss 4.530019
STEP 102 ================================
prereg loss 0.45947176 reg_l1 40.689407 reg_l2 24.764477
loss 4.5284123
STEP 103 ================================
prereg loss 0.4583311 reg_l1 40.684444 reg_l2 24.75998
loss 4.526776
STEP 104 ================================
prereg loss 0.45728216 reg_l1 40.679073 reg_l2 24.755045
loss 4.5251894
STEP 105 ================================
prereg loss 0.456398 reg_l1 40.673435 reg_l2 24.749771
loss 4.5237417
STEP 106 ================================
prereg loss 0.45572397 reg_l1 40.667603 reg_l2 24.744291
loss 4.5224843
STEP 107 ================================
prereg loss 0.45528358 reg_l1 40.661655 reg_l2 24.738718
loss 4.521449
STEP 108 ================================
prereg loss 0.45507872 reg_l1 40.655727 reg_l2 24.73314
loss 4.5206513
STEP 109 ================================
prereg loss 0.45507717 reg_l1 40.649845 reg_l2 24.727673
loss 4.520062
STEP 110 ================================
prereg loss 0.4552325 reg_l1 40.64413 reg_l2 24.722372
loss 4.5196457
STEP 111 ================================
prereg loss 0.455487 reg_l1 40.63858 reg_l2 24.717321
loss 4.5193453
STEP 112 ================================
prereg loss 0.45577472 reg_l1 40.633293 reg_l2 24.71256
loss 4.519104
STEP 113 ================================
prereg loss 0.45604452 reg_l1 40.62827 reg_l2 24.708128
loss 4.518872
STEP 114 ================================
prereg loss 0.45624223 reg_l1 40.62353 reg_l2 24.704033
loss 4.518595
STEP 115 ================================
prereg loss 0.4563431 reg_l1 40.619083 reg_l2 24.700293
loss 4.5182514
STEP 116 ================================
prereg loss 0.45632765 reg_l1 40.614906 reg_l2 24.696873
loss 4.517818
STEP 117 ================================
prereg loss 0.45619768 reg_l1 40.611 reg_l2 24.693779
loss 4.5172977
STEP 118 ================================
prereg loss 0.45596856 reg_l1 40.60733 reg_l2 24.690952
loss 4.5167017
STEP 119 ================================
prereg loss 0.4556644 reg_l1 40.60386 reg_l2 24.688372
loss 4.5160503
STEP 120 ================================
prereg loss 0.45531148 reg_l1 40.600536 reg_l2 24.685974
loss 4.515365
STEP 121 ================================
prereg loss 0.45494437 reg_l1 40.597336 reg_l2 24.683727
loss 4.514678
cutoff 0.20353049 network size 76
STEP 122 ================================
prereg loss 0.4545912 reg_l1 40.390694 reg_l2 24.64015
loss 4.493661
STEP 123 ================================
prereg loss 0.45427758 reg_l1 40.388294 reg_l2 24.638315
loss 4.493107
STEP 124 ================================
prereg loss 0.45402026 reg_l1 40.38587 reg_l2 24.636478
loss 4.492607
STEP 125 ================================
prereg loss 0.45383024 reg_l1 40.383423 reg_l2 24.634584
loss 4.4921727
STEP 126 ================================
prereg loss 0.45371363 reg_l1 40.38089 reg_l2 24.632607
loss 4.4918027
STEP 127 ================================
prereg loss 0.45366564 reg_l1 40.378242 reg_l2 24.630499
loss 4.49149
STEP 128 ================================
prereg loss 0.45368612 reg_l1 40.375473 reg_l2 24.628258
loss 4.491234
STEP 129 ================================
prereg loss 0.45376554 reg_l1 40.372547 reg_l2 24.62584
loss 4.49102
STEP 130 ================================
prereg loss 0.45389277 reg_l1 40.36948 reg_l2 24.623253
loss 4.490841
STEP 131 ================================
prereg loss 0.45406178 reg_l1 40.366234 reg_l2 24.620491
loss 4.4906855
STEP 132 ================================
prereg loss 0.45426467 reg_l1 40.36283 reg_l2 24.617558
loss 4.4905477
STEP 133 ================================
prereg loss 0.4544941 reg_l1 40.359303 reg_l2 24.614447
loss 4.490424
STEP 134 ================================
prereg loss 0.45474812 reg_l1 40.355633 reg_l2 24.611187
loss 4.4903116
STEP 135 ================================
prereg loss 0.45502168 reg_l1 40.351814 reg_l2 24.607796
loss 4.4902034
STEP 136 ================================
prereg loss 0.4553128 reg_l1 40.34792 reg_l2 24.604282
loss 4.4901047
STEP 137 ================================
prereg loss 0.45562115 reg_l1 40.343918 reg_l2 24.60068
loss 4.490013
STEP 138 ================================
prereg loss 0.45594278 reg_l1 40.339874 reg_l2 24.597012
loss 4.48993
STEP 139 ================================
prereg loss 0.45627537 reg_l1 40.33577 reg_l2 24.593285
loss 4.4898524
STEP 140 ================================
prereg loss 0.4566148 reg_l1 40.331646 reg_l2 24.589558
loss 4.4897795
STEP 141 ================================
prereg loss 0.45695835 reg_l1 40.327526 reg_l2 24.585829
loss 4.489711
STEP 142 ================================
prereg loss 0.45730346 reg_l1 40.323402 reg_l2 24.582111
loss 4.489644
STEP 143 ================================
prereg loss 0.4576419 reg_l1 40.319305 reg_l2 24.578424
loss 4.4895725
STEP 144 ================================
prereg loss 0.45797202 reg_l1 40.315266 reg_l2 24.574808
loss 4.4894986
STEP 145 ================================
prereg loss 0.4582851 reg_l1 40.311264 reg_l2 24.571241
loss 4.4894114
STEP 146 ================================
prereg loss 0.45858276 reg_l1 40.30732 reg_l2 24.567743
loss 4.489315
STEP 147 ================================
prereg loss 0.45886007 reg_l1 40.303425 reg_l2 24.564325
loss 4.4892025
STEP 148 ================================
prereg loss 0.4591169 reg_l1 40.2996 reg_l2 24.560978
loss 4.489077
STEP 149 ================================
prereg loss 0.45935252 reg_l1 40.295826 reg_l2 24.557707
loss 4.488935
STEP 150 ================================
prereg loss 0.45956522 reg_l1 40.292126 reg_l2 24.554512
loss 4.4887776
STEP 151 ================================
prereg loss 0.4597596 reg_l1 40.28845 reg_l2 24.551374
loss 4.4886045
STEP 152 ================================
prereg loss 0.4599352 reg_l1 40.284843 reg_l2 24.548302
loss 4.4884195
STEP 153 ================================
prereg loss 0.46009666 reg_l1 40.281258 reg_l2 24.545288
loss 4.4882226
STEP 154 ================================
prereg loss 0.46024582 reg_l1 40.27772 reg_l2 24.542303
loss 4.488018
STEP 155 ================================
prereg loss 0.46038496 reg_l1 40.27421 reg_l2 24.539356
loss 4.4878063
STEP 156 ================================
prereg loss 0.46051937 reg_l1 40.270714 reg_l2 24.53644
loss 4.487591
STEP 157 ================================
prereg loss 0.46064836 reg_l1 40.26721 reg_l2 24.533537
loss 4.4873695
STEP 158 ================================
prereg loss 0.46077535 reg_l1 40.263737 reg_l2 24.530647
loss 4.4871492
STEP 159 ================================
prereg loss 0.46090376 reg_l1 40.26023 reg_l2 24.527754
loss 4.486927
STEP 160 ================================
prereg loss 0.46103424 reg_l1 40.256725 reg_l2 24.524862
loss 4.4867067
STEP 161 ================================
prereg loss 0.4611682 reg_l1 40.25322 reg_l2 24.521961
loss 4.4864902
cutoff 0.21770723 network size 75
STEP 162 ================================
prereg loss 0.46130264 reg_l1 40.03198 reg_l2 24.47165
loss 4.464501
STEP 163 ================================
prereg loss 0.46144617 reg_l1 40.028625 reg_l2 24.468801
loss 4.4643087
STEP 164 ================================
prereg loss 0.46159187 reg_l1 40.025257 reg_l2 24.465956
loss 4.4641175
STEP 165 ================================
prereg loss 0.46173924 reg_l1 40.021866 reg_l2 24.463081
loss 4.463926
STEP 166 ================================
prereg loss 0.46189326 reg_l1 40.01847 reg_l2 24.460203
loss 4.4637403
STEP 167 ================================
prereg loss 0.46205306 reg_l1 40.015053 reg_l2 24.457317
loss 4.463558
STEP 168 ================================
prereg loss 0.46221384 reg_l1 40.011635 reg_l2 24.45442
loss 4.4633775
STEP 169 ================================
prereg loss 0.46237788 reg_l1 40.00822 reg_l2 24.451517
loss 4.4632
STEP 170 ================================
prereg loss 0.46254385 reg_l1 40.00478 reg_l2 24.448622
loss 4.463022
STEP 171 ================================
prereg loss 0.46271092 reg_l1 40.001347 reg_l2 24.445738
loss 4.462846
STEP 172 ================================
prereg loss 0.46287534 reg_l1 39.997917 reg_l2 24.442852
loss 4.462667
STEP 173 ================================
prereg loss 0.4630415 reg_l1 39.994503 reg_l2 24.43998
loss 4.462492
STEP 174 ================================
prereg loss 0.46320876 reg_l1 39.991085 reg_l2 24.437128
loss 4.4623175
STEP 175 ================================
prereg loss 0.4633679 reg_l1 39.987682 reg_l2 24.434288
loss 4.4621363
STEP 176 ================================
prereg loss 0.46352372 reg_l1 39.9843 reg_l2 24.431475
loss 4.4619536
STEP 177 ================================
prereg loss 0.46367726 reg_l1 39.980938 reg_l2 24.428684
loss 4.461771
STEP 178 ================================
prereg loss 0.4638237 reg_l1 39.977585 reg_l2 24.425924
loss 4.461582
STEP 179 ================================
prereg loss 0.46396556 reg_l1 39.974236 reg_l2 24.42319
loss 4.461389
STEP 180 ================================
prereg loss 0.4641 reg_l1 39.970936 reg_l2 24.420473
loss 4.4611936
STEP 181 ================================
prereg loss 0.46422774 reg_l1 39.967655 reg_l2 24.417791
loss 4.4609933
STEP 182 ================================
prereg loss 0.46434864 reg_l1 39.96438 reg_l2 24.415134
loss 4.4607863
STEP 183 ================================
prereg loss 0.46446258 reg_l1 39.961113 reg_l2 24.412498
loss 4.460574
STEP 184 ================================
prereg loss 0.4645718 reg_l1 39.957867 reg_l2 24.409887
loss 4.4603586
STEP 185 ================================
prereg loss 0.4646746 reg_l1 39.954636 reg_l2 24.4073
loss 4.4601383
STEP 186 ================================
prereg loss 0.46476963 reg_l1 39.951435 reg_l2 24.404734
loss 4.4599133
STEP 187 ================================
prereg loss 0.464862 reg_l1 39.948242 reg_l2 24.402185
loss 4.4596863
STEP 188 ================================
prereg loss 0.46494702 reg_l1 39.945057 reg_l2 24.399664
loss 4.4594526
STEP 189 ================================
prereg loss 0.4650174 reg_l1 39.941895 reg_l2 24.397167
loss 4.459207
STEP 190 ================================
prereg loss 0.46507853 reg_l1 39.93876 reg_l2 24.3947
loss 4.4589543
STEP 191 ================================
prereg loss 0.4651306 reg_l1 39.935627 reg_l2 24.392254
loss 4.4586935
STEP 192 ================================
prereg loss 0.46517745 reg_l1 39.93251 reg_l2 24.389832
loss 4.4584284
STEP 193 ================================
prereg loss 0.46521616 reg_l1 39.929413 reg_l2 24.387423
loss 4.4581575
STEP 194 ================================
prereg loss 0.4652502 reg_l1 39.92632 reg_l2 24.385029
loss 4.457882
STEP 195 ================================
prereg loss 0.46528208 reg_l1 39.92323 reg_l2 24.382639
loss 4.457605
STEP 196 ================================
prereg loss 0.4653119 reg_l1 39.92015 reg_l2 24.38026
loss 4.457327
STEP 197 ================================
prereg loss 0.46534452 reg_l1 39.917076 reg_l2 24.377888
loss 4.457052
STEP 198 ================================
prereg loss 0.46537447 reg_l1 39.913998 reg_l2 24.37551
loss 4.456774
STEP 199 ================================
prereg loss 0.46540415 reg_l1 39.91091 reg_l2 24.373148
loss 4.4564953
STEP 200 ================================
prereg loss 0.46543586 reg_l1 39.90783 reg_l2 24.37078
loss 4.4562187
2022-07-20T16:12:01.952

julia> serialize("cf-75-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-75-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T16:21:18.880
STEP 1 ================================
prereg loss 0.46546745 reg_l1 39.90475 reg_l2 24.368416
loss 4.4559426
cutoff 0.23269485 network size 74
STEP 2 ================================
prereg loss 0.46549985 reg_l1 39.66896 reg_l2 24.311901
loss 4.432396
STEP 3 ================================
prereg loss 0.4655335 reg_l1 39.66597 reg_l2 24.309586
loss 4.432131
STEP 4 ================================
prereg loss 0.46556684 reg_l1 39.662987 reg_l2 24.307264
loss 4.4318657
STEP 5 ================================
prereg loss 0.46559846 reg_l1 39.66001 reg_l2 24.304962
loss 4.4315996
STEP 6 ================================
prereg loss 0.46562997 reg_l1 39.657013 reg_l2 24.302649
loss 4.431331
STEP 7 ================================
prereg loss 0.46566013 reg_l1 39.654034 reg_l2 24.30035
loss 4.4310637
STEP 8 ================================
prereg loss 0.46568623 reg_l1 39.65104 reg_l2 24.29805
loss 4.43079
STEP 9 ================================
prereg loss 0.4657103 reg_l1 39.648056 reg_l2 24.295753
loss 4.430516
STEP 10 ================================
prereg loss 0.46572942 reg_l1 39.64508 reg_l2 24.293465
loss 4.4302373
STEP 11 ================================
prereg loss 0.46574697 reg_l1 39.642128 reg_l2 24.29119
loss 4.42996
STEP 12 ================================
prereg loss 0.465758 reg_l1 39.639137 reg_l2 24.288921
loss 4.429672
STEP 13 ================================
prereg loss 0.4657654 reg_l1 39.636173 reg_l2 24.28666
loss 4.429383
STEP 14 ================================
prereg loss 0.4657686 reg_l1 39.63323 reg_l2 24.28442
loss 4.4290915
STEP 15 ================================
prereg loss 0.46576545 reg_l1 39.63027 reg_l2 24.28218
loss 4.428792
STEP 16 ================================
prereg loss 0.4657576 reg_l1 39.62733 reg_l2 24.279959
loss 4.4284906
STEP 17 ================================
prereg loss 0.46574613 reg_l1 39.624393 reg_l2 24.27775
loss 4.4281855
STEP 18 ================================
prereg loss 0.46572813 reg_l1 39.62146 reg_l2 24.275545
loss 4.427874
STEP 19 ================================
prereg loss 0.4657073 reg_l1 39.618538 reg_l2 24.27336
loss 4.427561
STEP 20 ================================
prereg loss 0.46567756 reg_l1 39.615627 reg_l2 24.271189
loss 4.4272404
STEP 21 ================================
prereg loss 0.46564737 reg_l1 39.612717 reg_l2 24.26902
loss 4.426919
STEP 22 ================================
prereg loss 0.46561104 reg_l1 39.609818 reg_l2 24.26686
loss 4.426593
STEP 23 ================================
prereg loss 0.4655741 reg_l1 39.606915 reg_l2 24.264713
loss 4.4262657
STEP 24 ================================
prereg loss 0.46552968 reg_l1 39.604027 reg_l2 24.262571
loss 4.4259324
STEP 25 ================================
prereg loss 0.4654869 reg_l1 39.60114 reg_l2 24.260443
loss 4.425601
STEP 26 ================================
prereg loss 0.46543935 reg_l1 39.598248 reg_l2 24.258318
loss 4.4252644
STEP 27 ================================
prereg loss 0.46538803 reg_l1 39.59537 reg_l2 24.256195
loss 4.4249253
STEP 28 ================================
prereg loss 0.46533296 reg_l1 39.592487 reg_l2 24.25409
loss 4.4245815
STEP 29 ================================
prereg loss 0.46528053 reg_l1 39.58961 reg_l2 24.251976
loss 4.424242
STEP 30 ================================
prereg loss 0.46522003 reg_l1 39.58674 reg_l2 24.249872
loss 4.423894
STEP 31 ================================
prereg loss 0.46516147 reg_l1 39.58385 reg_l2 24.24778
loss 4.423547
STEP 32 ================================
prereg loss 0.46510014 reg_l1 39.580982 reg_l2 24.245684
loss 4.423198
STEP 33 ================================
prereg loss 0.46503487 reg_l1 39.578114 reg_l2 24.243587
loss 4.4228463
STEP 34 ================================
prereg loss 0.46496895 reg_l1 39.575226 reg_l2 24.2415
loss 4.4224916
STEP 35 ================================
prereg loss 0.46489984 reg_l1 39.57236 reg_l2 24.239418
loss 4.422136
STEP 36 ================================
prereg loss 0.46482927 reg_l1 39.569473 reg_l2 24.237337
loss 4.421777
STEP 37 ================================
prereg loss 0.4647549 reg_l1 39.56661 reg_l2 24.235252
loss 4.421416
STEP 38 ================================
prereg loss 0.46467936 reg_l1 39.56373 reg_l2 24.233175
loss 4.4210525
STEP 39 ================================
prereg loss 0.46459982 reg_l1 39.56086 reg_l2 24.231094
loss 4.420686
STEP 40 ================================
prereg loss 0.46451876 reg_l1 39.557976 reg_l2 24.229027
loss 4.420316
STEP 41 ================================
prereg loss 0.46443707 reg_l1 39.555103 reg_l2 24.226967
loss 4.4199476
cutoff 0.24703746 network size 73
STEP 42 ================================
prereg loss 0.46435007 reg_l1 39.305202 reg_l2 24.163874
loss 4.3948703
STEP 43 ================================
prereg loss 0.4642612 reg_l1 39.30246 reg_l2 24.161877
loss 4.3945074
STEP 44 ================================
prereg loss 0.46417013 reg_l1 39.2997 reg_l2 24.159885
loss 4.3941402
STEP 45 ================================
prereg loss 0.4640758 reg_l1 39.296967 reg_l2 24.157898
loss 4.3937726
STEP 46 ================================
prereg loss 0.4639789 reg_l1 39.294243 reg_l2 24.15592
loss 4.393403
STEP 47 ================================
prereg loss 0.46387854 reg_l1 39.291496 reg_l2 24.153946
loss 4.3930283
STEP 48 ================================
prereg loss 0.4637758 reg_l1 39.288757 reg_l2 24.151974
loss 4.3926516
STEP 49 ================================
prereg loss 0.4636703 reg_l1 39.286026 reg_l2 24.150002
loss 4.392273
STEP 50 ================================
prereg loss 0.463562 reg_l1 39.283295 reg_l2 24.148037
loss 4.3918915
STEP 51 ================================
prereg loss 0.4634522 reg_l1 39.28057 reg_l2 24.14608
loss 4.3915095
STEP 52 ================================
prereg loss 0.4633393 reg_l1 39.277832 reg_l2 24.144123
loss 4.3911223
STEP 53 ================================
prereg loss 0.4632238 reg_l1 39.27511 reg_l2 24.142178
loss 4.3907347
STEP 54 ================================
prereg loss 0.46310392 reg_l1 39.272385 reg_l2 24.140236
loss 4.390342
STEP 55 ================================
prereg loss 0.46298355 reg_l1 39.269676 reg_l2 24.13829
loss 4.389951
STEP 56 ================================
prereg loss 0.4628612 reg_l1 39.266945 reg_l2 24.136356
loss 4.389556
STEP 57 ================================
prereg loss 0.462734 reg_l1 39.26423 reg_l2 24.134415
loss 4.389157
STEP 58 ================================
prereg loss 0.4626057 reg_l1 39.261497 reg_l2 24.132498
loss 4.388756
STEP 59 ================================
prereg loss 0.46247813 reg_l1 39.25879 reg_l2 24.130566
loss 4.388357
STEP 60 ================================
prereg loss 0.4623449 reg_l1 39.256058 reg_l2 24.128647
loss 4.387951
STEP 61 ================================
prereg loss 0.46220845 reg_l1 39.253334 reg_l2 24.12672
loss 4.387542
STEP 62 ================================
prereg loss 0.46207288 reg_l1 39.250626 reg_l2 24.124807
loss 4.3871355
STEP 63 ================================
prereg loss 0.46193343 reg_l1 39.247902 reg_l2 24.12289
loss 4.3867235
STEP 64 ================================
prereg loss 0.4617927 reg_l1 39.24519 reg_l2 24.12098
loss 4.3863115
STEP 65 ================================
prereg loss 0.4616488 reg_l1 39.242474 reg_l2 24.119066
loss 4.385896
STEP 66 ================================
prereg loss 0.46150213 reg_l1 39.239746 reg_l2 24.117165
loss 4.385477
STEP 67 ================================
prereg loss 0.46135536 reg_l1 39.23703 reg_l2 24.115248
loss 4.3850584
STEP 68 ================================
prereg loss 0.46120563 reg_l1 39.234306 reg_l2 24.113346
loss 4.3846364
STEP 69 ================================
prereg loss 0.46105394 reg_l1 39.231583 reg_l2 24.111443
loss 4.3842125
STEP 70 ================================
prereg loss 0.46090114 reg_l1 39.22886 reg_l2 24.109545
loss 4.383787
STEP 71 ================================
prereg loss 0.46074542 reg_l1 39.226147 reg_l2 24.107653
loss 4.3833604
STEP 72 ================================
prereg loss 0.46058652 reg_l1 39.223434 reg_l2 24.105751
loss 4.38293
STEP 73 ================================
prereg loss 0.46042806 reg_l1 39.220695 reg_l2 24.103851
loss 4.382498
STEP 74 ================================
prereg loss 0.46026522 reg_l1 39.217983 reg_l2 24.101963
loss 4.382064
STEP 75 ================================
prereg loss 0.46010074 reg_l1 39.215256 reg_l2 24.100073
loss 4.3816266
STEP 76 ================================
prereg loss 0.45993438 reg_l1 39.212532 reg_l2 24.09818
loss 4.3811874
STEP 77 ================================
prereg loss 0.45976678 reg_l1 39.209816 reg_l2 24.096294
loss 4.3807483
STEP 78 ================================
prereg loss 0.45959917 reg_l1 39.20709 reg_l2 24.09441
loss 4.380308
STEP 79 ================================
prereg loss 0.45942503 reg_l1 39.20437 reg_l2 24.092527
loss 4.379862
STEP 80 ================================
prereg loss 0.45925292 reg_l1 39.201633 reg_l2 24.090641
loss 4.3794165
STEP 81 ================================
prereg loss 0.45907804 reg_l1 39.198917 reg_l2 24.088764
loss 4.3789697
cutoff 0.26827025 network size 72
STEP 82 ================================
prereg loss 0.45889872 reg_l1 38.92792 reg_l2 24.014915
loss 4.351691
STEP 83 ================================
prereg loss 0.4587195 reg_l1 38.92544 reg_l2 24.013182
loss 4.351264
STEP 84 ================================
prereg loss 0.45853838 reg_l1 38.92295 reg_l2 24.01143
loss 4.3508334
STEP 85 ================================
prereg loss 0.45835602 reg_l1 38.920483 reg_l2 24.009687
loss 4.3504043
STEP 86 ================================
prereg loss 0.45816866 reg_l1 38.917995 reg_l2 24.00795
loss 4.3499684
STEP 87 ================================
prereg loss 0.45798206 reg_l1 38.915512 reg_l2 24.006208
loss 4.349533
STEP 88 ================================
prereg loss 0.45779374 reg_l1 38.913033 reg_l2 24.004475
loss 4.3490973
STEP 89 ================================
prereg loss 0.4576025 reg_l1 38.910545 reg_l2 24.002739
loss 4.348657
STEP 90 ================================
prereg loss 0.4574115 reg_l1 38.908066 reg_l2 24.001001
loss 4.348218
STEP 91 ================================
prereg loss 0.45721766 reg_l1 38.905586 reg_l2 23.999266
loss 4.3477764
STEP 92 ================================
prereg loss 0.4570216 reg_l1 38.90309 reg_l2 23.997528
loss 4.3473306
STEP 93 ================================
prereg loss 0.45682484 reg_l1 38.90061 reg_l2 23.995798
loss 4.346886
STEP 94 ================================
prereg loss 0.45662487 reg_l1 38.898117 reg_l2 23.99407
loss 4.3464365
STEP 95 ================================
prereg loss 0.45642388 reg_l1 38.895634 reg_l2 23.992334
loss 4.3459873
STEP 96 ================================
prereg loss 0.45622125 reg_l1 38.893147 reg_l2 23.9906
loss 4.3455358
STEP 97 ================================
prereg loss 0.45601776 reg_l1 38.890648 reg_l2 23.98887
loss 4.3450828
STEP 98 ================================
prereg loss 0.45581186 reg_l1 38.888153 reg_l2 23.987135
loss 4.3446274
STEP 99 ================================
prereg loss 0.4556044 reg_l1 38.88566 reg_l2 23.985403
loss 4.34417
STEP 100 ================================
prereg loss 0.45539474 reg_l1 38.883167 reg_l2 23.983673
loss 4.343712
STEP 101 ================================
prereg loss 0.45518488 reg_l1 38.88067 reg_l2 23.981941
loss 4.3432517
STEP 102 ================================
prereg loss 0.45497254 reg_l1 38.878166 reg_l2 23.980211
loss 4.342789
STEP 103 ================================
prereg loss 0.45475706 reg_l1 38.875668 reg_l2 23.978483
loss 4.342324
STEP 104 ================================
prereg loss 0.4545427 reg_l1 38.87316 reg_l2 23.97675
loss 4.341859
STEP 105 ================================
prereg loss 0.45432788 reg_l1 38.87066 reg_l2 23.975018
loss 4.341394
STEP 106 ================================
prereg loss 0.45410842 reg_l1 38.868153 reg_l2 23.973278
loss 4.340924
STEP 107 ================================
prereg loss 0.45388907 reg_l1 38.86565 reg_l2 23.971554
loss 4.340454
STEP 108 ================================
prereg loss 0.4536693 reg_l1 38.86314 reg_l2 23.96982
loss 4.3399835
STEP 109 ================================
prereg loss 0.45344648 reg_l1 38.860622 reg_l2 23.96809
loss 4.339509
STEP 110 ================================
prereg loss 0.45322123 reg_l1 38.858112 reg_l2 23.96636
loss 4.3390326
STEP 111 ================================
prereg loss 0.4529967 reg_l1 38.85559 reg_l2 23.964626
loss 4.338556
STEP 112 ================================
prereg loss 0.45277053 reg_l1 38.85309 reg_l2 23.962894
loss 4.3380795
STEP 113 ================================
prereg loss 0.45254046 reg_l1 38.850567 reg_l2 23.961159
loss 4.3375974
STEP 114 ================================
prereg loss 0.4523121 reg_l1 38.84804 reg_l2 23.959433
loss 4.3371162
STEP 115 ================================
prereg loss 0.4520814 reg_l1 38.84552 reg_l2 23.957693
loss 4.336633
STEP 116 ================================
prereg loss 0.4518476 reg_l1 38.843014 reg_l2 23.955956
loss 4.336149
STEP 117 ================================
prereg loss 0.45161554 reg_l1 38.840485 reg_l2 23.954222
loss 4.335664
STEP 118 ================================
prereg loss 0.45138124 reg_l1 38.83795 reg_l2 23.952484
loss 4.3351765
STEP 119 ================================
prereg loss 0.45114604 reg_l1 38.835426 reg_l2 23.95075
loss 4.3346887
STEP 120 ================================
prereg loss 0.45090857 reg_l1 38.83289 reg_l2 23.949007
loss 4.3341975
STEP 121 ================================
prereg loss 0.45066735 reg_l1 38.830353 reg_l2 23.947273
loss 4.3337026
cutoff 0.2659197 network size 71
STEP 122 ================================
prereg loss 0.45042846 reg_l1 38.561897 reg_l2 23.874813
loss 4.306618
STEP 123 ================================
prereg loss 0.45018908 reg_l1 38.559586 reg_l2 23.873194
loss 4.3061476
STEP 124 ================================
prereg loss 0.44994766 reg_l1 38.557278 reg_l2 23.87157
loss 4.3056755
STEP 125 ================================
prereg loss 0.4497028 reg_l1 38.554966 reg_l2 23.869953
loss 4.3051996
STEP 126 ================================
prereg loss 0.44945908 reg_l1 38.55264 reg_l2 23.868326
loss 4.304723
STEP 127 ================================
prereg loss 0.44921437 reg_l1 38.550327 reg_l2 23.866707
loss 4.304247
STEP 128 ================================
prereg loss 0.44896665 reg_l1 38.548004 reg_l2 23.865082
loss 4.303767
STEP 129 ================================
prereg loss 0.44871783 reg_l1 38.545673 reg_l2 23.86345
loss 4.303285
STEP 130 ================================
prereg loss 0.4484707 reg_l1 38.54335 reg_l2 23.861824
loss 4.302806
STEP 131 ================================
prereg loss 0.44822013 reg_l1 38.541027 reg_l2 23.860188
loss 4.302323
STEP 132 ================================
prereg loss 0.44796944 reg_l1 38.538685 reg_l2 23.858562
loss 4.301838
STEP 133 ================================
prereg loss 0.4477182 reg_l1 38.53635 reg_l2 23.856922
loss 4.3013535
STEP 134 ================================
prereg loss 0.4474638 reg_l1 38.534016 reg_l2 23.855291
loss 4.3008657
STEP 135 ================================
prereg loss 0.44720966 reg_l1 38.531673 reg_l2 23.85365
loss 4.300377
STEP 136 ================================
prereg loss 0.44695547 reg_l1 38.52933 reg_l2 23.852016
loss 4.2998886
STEP 137 ================================
prereg loss 0.4466985 reg_l1 38.52699 reg_l2 23.850378
loss 4.2993975
STEP 138 ================================
prereg loss 0.44644183 reg_l1 38.52463 reg_l2 23.848736
loss 4.298905
STEP 139 ================================
prereg loss 0.44618452 reg_l1 38.522293 reg_l2 23.84709
loss 4.2984138
STEP 140 ================================
prereg loss 0.4459243 reg_l1 38.51993 reg_l2 23.845446
loss 4.297918
STEP 141 ================================
prereg loss 0.44566482 reg_l1 38.517574 reg_l2 23.843798
loss 4.2974224
STEP 142 ================================
prereg loss 0.44540438 reg_l1 38.51522 reg_l2 23.842144
loss 4.2969265
STEP 143 ================================
prereg loss 0.44514114 reg_l1 38.51286 reg_l2 23.840504
loss 4.2964272
STEP 144 ================================
prereg loss 0.44487908 reg_l1 38.510494 reg_l2 23.838839
loss 4.2959285
STEP 145 ================================
prereg loss 0.44461536 reg_l1 38.508137 reg_l2 23.83719
loss 4.295429
STEP 146 ================================
prereg loss 0.4443518 reg_l1 38.505764 reg_l2 23.83553
loss 4.294928
STEP 147 ================================
prereg loss 0.44408572 reg_l1 38.503384 reg_l2 23.833872
loss 4.294424
STEP 148 ================================
prereg loss 0.4438209 reg_l1 38.501015 reg_l2 23.832209
loss 4.2939224
STEP 149 ================================
prereg loss 0.44355413 reg_l1 38.498627 reg_l2 23.830542
loss 4.293417
STEP 150 ================================
prereg loss 0.44328767 reg_l1 38.496254 reg_l2 23.828882
loss 4.292913
STEP 151 ================================
prereg loss 0.44301975 reg_l1 38.493866 reg_l2 23.827208
loss 4.2924066
STEP 152 ================================
prereg loss 0.44275022 reg_l1 38.49148 reg_l2 23.825535
loss 4.2918987
STEP 153 ================================
prereg loss 0.4424801 reg_l1 38.48909 reg_l2 23.82386
loss 4.2913895
STEP 154 ================================
prereg loss 0.44220966 reg_l1 38.486694 reg_l2 23.822187
loss 4.2908792
STEP 155 ================================
prereg loss 0.4419384 reg_l1 38.4843 reg_l2 23.820509
loss 4.290368
STEP 156 ================================
prereg loss 0.4416686 reg_l1 38.481895 reg_l2 23.818825
loss 4.2898583
STEP 157 ================================
prereg loss 0.44139427 reg_l1 38.47948 reg_l2 23.817139
loss 4.2893424
STEP 158 ================================
prereg loss 0.44112206 reg_l1 38.477085 reg_l2 23.815458
loss 4.2888308
STEP 159 ================================
prereg loss 0.4408475 reg_l1 38.474678 reg_l2 23.813766
loss 4.2883153
STEP 160 ================================
prereg loss 0.44057307 reg_l1 38.47226 reg_l2 23.812073
loss 4.287799
STEP 161 ================================
prereg loss 0.44029918 reg_l1 38.46985 reg_l2 23.810377
loss 4.287284
cutoff 0.26515263 network size 70
STEP 162 ================================
prereg loss 0.44002488 reg_l1 38.20227 reg_l2 23.738375
loss 4.260252
STEP 163 ================================
prereg loss 0.43974805 reg_l1 38.199913 reg_l2 23.736713
loss 4.2597394
STEP 164 ================================
prereg loss 0.43947256 reg_l1 38.19755 reg_l2 23.735048
loss 4.2592278
STEP 165 ================================
prereg loss 0.43919495 reg_l1 38.195187 reg_l2 23.73337
loss 4.2587137
STEP 166 ================================
prereg loss 0.43891588 reg_l1 38.19283 reg_l2 23.731703
loss 4.2581987
STEP 167 ================================
prereg loss 0.43863857 reg_l1 38.190453 reg_l2 23.730022
loss 4.2576838
STEP 168 ================================
prereg loss 0.4383593 reg_l1 38.18808 reg_l2 23.728344
loss 4.2571673
STEP 169 ================================
prereg loss 0.43808034 reg_l1 38.185707 reg_l2 23.726665
loss 4.2566514
STEP 170 ================================
prereg loss 0.43779886 reg_l1 38.183323 reg_l2 23.724974
loss 4.256131
STEP 171 ================================
prereg loss 0.43752107 reg_l1 38.180935 reg_l2 23.723291
loss 4.2556148
STEP 172 ================================
prereg loss 0.43724096 reg_l1 38.17856 reg_l2 23.721598
loss 4.255097
STEP 173 ================================
prereg loss 0.43696412 reg_l1 38.176144 reg_l2 23.719881
loss 4.2545786
STEP 174 ================================
prereg loss 0.43669164 reg_l1 38.173714 reg_l2 23.71816
loss 4.254063
STEP 175 ================================
prereg loss 0.43642122 reg_l1 38.17129 reg_l2 23.71642
loss 4.2535505
STEP 176 ================================
prereg loss 0.43615454 reg_l1 38.16885 reg_l2 23.71467
loss 4.2530394
STEP 177 ================================
prereg loss 0.43588915 reg_l1 38.1664 reg_l2 23.71291
loss 4.252529
STEP 178 ================================
prereg loss 0.4356266 reg_l1 38.16394 reg_l2 23.711145
loss 4.252021
STEP 179 ================================
prereg loss 0.43536246 reg_l1 38.16147 reg_l2 23.709377
loss 4.251509
STEP 180 ================================
prereg loss 0.43509707 reg_l1 38.158997 reg_l2 23.707602
loss 4.2509966
STEP 181 ================================
prereg loss 0.43483198 reg_l1 38.156525 reg_l2 23.70582
loss 4.2504845
STEP 182 ================================
prereg loss 0.4345636 reg_l1 38.154057 reg_l2 23.704048
loss 4.249969
STEP 183 ================================
prereg loss 0.43429467 reg_l1 38.15159 reg_l2 23.702278
loss 4.2494535
STEP 184 ================================
prereg loss 0.43402448 reg_l1 38.14911 reg_l2 23.700508
loss 4.248935
STEP 185 ================================
prereg loss 0.43374893 reg_l1 38.146637 reg_l2 23.698744
loss 4.2484126
STEP 186 ================================
prereg loss 0.43347317 reg_l1 38.144165 reg_l2 23.696981
loss 4.24789
STEP 187 ================================
prereg loss 0.43319377 reg_l1 38.141693 reg_l2 23.695215
loss 4.247363
STEP 188 ================================
prereg loss 0.43291497 reg_l1 38.13923 reg_l2 23.693466
loss 4.2468376
STEP 189 ================================
prereg loss 0.43263158 reg_l1 38.13676 reg_l2 23.69171
loss 4.246308
STEP 190 ================================
prereg loss 0.43234843 reg_l1 38.134308 reg_l2 23.689957
loss 4.245779
STEP 191 ================================
prereg loss 0.43206525 reg_l1 38.13185 reg_l2 23.688215
loss 4.2452507
STEP 192 ================================
prereg loss 0.4317826 reg_l1 38.12938 reg_l2 23.68646
loss 4.2447205
STEP 193 ================================
prereg loss 0.4314972 reg_l1 38.126926 reg_l2 23.684713
loss 4.2441897
STEP 194 ================================
prereg loss 0.43121216 reg_l1 38.124454 reg_l2 23.682966
loss 4.2436576
STEP 195 ================================
prereg loss 0.43092808 reg_l1 38.12199 reg_l2 23.681213
loss 4.2431273
STEP 196 ================================
prereg loss 0.43064466 reg_l1 38.11951 reg_l2 23.679455
loss 4.2425957
STEP 197 ================================
prereg loss 0.4303629 reg_l1 38.117035 reg_l2 23.677696
loss 4.2420664
STEP 198 ================================
prereg loss 0.43008128 reg_l1 38.114555 reg_l2 23.675934
loss 4.2415366
STEP 199 ================================
prereg loss 0.42980072 reg_l1 38.112072 reg_l2 23.67416
loss 4.241008
STEP 200 ================================
prereg loss 0.42952177 reg_l1 38.109585 reg_l2 23.672392
loss 4.2404804
2022-07-20T16:33:36.394

julia> serialize("cf-70-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-70-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T16:39:58.033
STEP 1 ================================
prereg loss 0.4292412 reg_l1 38.107082 reg_l2 23.670609
loss 4.2399497
cutoff 0.27343735 network size 69
STEP 2 ================================
prereg loss 0.43149018 reg_l1 37.831142 reg_l2 23.594048
loss 4.2146044
STEP 3 ================================
prereg loss 0.43110317 reg_l1 37.829006 reg_l2 23.592625
loss 4.2140036
STEP 4 ================================
prereg loss 0.43062416 reg_l1 37.82709 reg_l2 23.591442
loss 4.213333
STEP 5 ================================
prereg loss 0.43007973 reg_l1 37.82536 reg_l2 23.590483
loss 4.212616
STEP 6 ================================
prereg loss 0.4294936 reg_l1 37.823746 reg_l2 23.589666
loss 4.2118683
STEP 7 ================================
prereg loss 0.42889038 reg_l1 37.822235 reg_l2 23.588974
loss 4.211114
STEP 8 ================================
prereg loss 0.42828614 reg_l1 37.82078 reg_l2 23.588352
loss 4.2103643
STEP 9 ================================
prereg loss 0.4277042 reg_l1 37.819355 reg_l2 23.587759
loss 4.2096395
STEP 10 ================================
prereg loss 0.42715028 reg_l1 37.81793 reg_l2 23.587145
loss 4.2089434
STEP 11 ================================
prereg loss 0.42662874 reg_l1 37.816444 reg_l2 23.586494
loss 4.2082734
STEP 12 ================================
prereg loss 0.4261445 reg_l1 37.8149 reg_l2 23.585756
loss 4.2076344
STEP 13 ================================
prereg loss 0.42569658 reg_l1 37.813274 reg_l2 23.58493
loss 4.207024
STEP 14 ================================
prereg loss 0.42528412 reg_l1 37.811546 reg_l2 23.583994
loss 4.2064385
STEP 15 ================================
prereg loss 0.42490005 reg_l1 37.80972 reg_l2 23.582914
loss 4.205872
STEP 16 ================================
prereg loss 0.42454332 reg_l1 37.807762 reg_l2 23.581715
loss 4.2053194
STEP 17 ================================
prereg loss 0.42420867 reg_l1 37.805714 reg_l2 23.580372
loss 4.20478
STEP 18 ================================
prereg loss 0.4238927 reg_l1 37.803543 reg_l2 23.578917
loss 4.204247
STEP 19 ================================
prereg loss 0.42359316 reg_l1 37.80128 reg_l2 23.577347
loss 4.2037215
STEP 20 ================================
prereg loss 0.42330906 reg_l1 37.798923 reg_l2 23.575665
loss 4.2032013
STEP 21 ================================
prereg loss 0.4230385 reg_l1 37.79649 reg_l2 23.573906
loss 4.2026877
STEP 22 ================================
prereg loss 0.42277575 reg_l1 37.793983 reg_l2 23.572086
loss 4.202174
STEP 23 ================================
prereg loss 0.4225214 reg_l1 37.791443 reg_l2 23.570204
loss 4.201666
STEP 24 ================================
prereg loss 0.42227617 reg_l1 37.78888 reg_l2 23.5683
loss 4.2011642
STEP 25 ================================
prereg loss 0.42203116 reg_l1 37.786297 reg_l2 23.566372
loss 4.2006607
STEP 26 ================================
prereg loss 0.42178458 reg_l1 37.78372 reg_l2 23.56445
loss 4.200156
STEP 27 ================================
prereg loss 0.4215356 reg_l1 37.78115 reg_l2 23.562548
loss 4.199651
STEP 28 ================================
prereg loss 0.42128032 reg_l1 37.77859 reg_l2 23.560667
loss 4.1991396
STEP 29 ================================
prereg loss 0.42101955 reg_l1 37.77607 reg_l2 23.558828
loss 4.1986265
STEP 30 ================================
prereg loss 0.42074534 reg_l1 37.773586 reg_l2 23.55703
loss 4.198104
STEP 31 ================================
prereg loss 0.4204621 reg_l1 37.77114 reg_l2 23.555273
loss 4.197576
STEP 32 ================================
prereg loss 0.42016992 reg_l1 37.768738 reg_l2 23.55356
loss 4.197044
STEP 33 ================================
prereg loss 0.4198658 reg_l1 37.76637 reg_l2 23.551893
loss 4.1965027
STEP 34 ================================
prereg loss 0.41955373 reg_l1 37.764034 reg_l2 23.550272
loss 4.195957
STEP 35 ================================
prereg loss 0.41923407 reg_l1 37.76174 reg_l2 23.548677
loss 4.1954083
STEP 36 ================================
prereg loss 0.4189109 reg_l1 37.75946 reg_l2 23.547115
loss 4.194857
STEP 37 ================================
prereg loss 0.41858357 reg_l1 37.757202 reg_l2 23.545563
loss 4.194304
STEP 38 ================================
prereg loss 0.4182559 reg_l1 37.754948 reg_l2 23.544012
loss 4.193751
STEP 39 ================================
prereg loss 0.4179288 reg_l1 37.752697 reg_l2 23.54247
loss 4.1931987
STEP 40 ================================
prereg loss 0.41760778 reg_l1 37.750435 reg_l2 23.540915
loss 4.1926513
STEP 41 ================================
prereg loss 0.41728923 reg_l1 37.748165 reg_l2 23.539337
loss 4.192106
cutoff 0.27021608 network size 68
STEP 42 ================================
prereg loss 0.41711357 reg_l1 37.475636 reg_l2 23.464718
loss 4.164677
STEP 43 ================================
prereg loss 0.4166874 reg_l1 37.473587 reg_l2 23.463354
loss 4.1640463
STEP 44 ================================
prereg loss 0.41619462 reg_l1 37.471687 reg_l2 23.462154
loss 4.1633635
STEP 45 ================================
prereg loss 0.415658 reg_l1 37.469864 reg_l2 23.461075
loss 4.1626444
STEP 46 ================================
prereg loss 0.41510406 reg_l1 37.468124 reg_l2 23.460075
loss 4.1619167
STEP 47 ================================
prereg loss 0.41455305 reg_l1 37.46642 reg_l2 23.459106
loss 4.161195
STEP 48 ================================
prereg loss 0.41401887 reg_l1 37.46471 reg_l2 23.458147
loss 4.16049
STEP 49 ================================
prereg loss 0.41351503 reg_l1 37.462975 reg_l2 23.457153
loss 4.1598125
STEP 50 ================================
prereg loss 0.4130443 reg_l1 37.461197 reg_l2 23.456093
loss 4.159164
STEP 51 ================================
prereg loss 0.4126122 reg_l1 37.45934 reg_l2 23.454948
loss 4.1585464
STEP 52 ================================
prereg loss 0.41221508 reg_l1 37.457386 reg_l2 23.45369
loss 4.1579537
STEP 53 ================================
prereg loss 0.4118498 reg_l1 37.455338 reg_l2 23.452301
loss 4.1573834
STEP 54 ================================
prereg loss 0.4115124 reg_l1 37.453175 reg_l2 23.450777
loss 4.15683
STEP 55 ================================
prereg loss 0.41120118 reg_l1 37.450897 reg_l2 23.449114
loss 4.156291
STEP 56 ================================
prereg loss 0.41090745 reg_l1 37.448494 reg_l2 23.447313
loss 4.155757
STEP 57 ================================
prereg loss 0.41063303 reg_l1 37.44599 reg_l2 23.445375
loss 4.155232
STEP 58 ================================
prereg loss 0.4103742 reg_l1 37.44338 reg_l2 23.443314
loss 4.154712
STEP 59 ================================
prereg loss 0.41012615 reg_l1 37.440666 reg_l2 23.441154
loss 4.154193
STEP 60 ================================
prereg loss 0.4098898 reg_l1 37.437885 reg_l2 23.438904
loss 4.1536784
STEP 61 ================================
prereg loss 0.4096618 reg_l1 37.435036 reg_l2 23.436577
loss 4.1531653
STEP 62 ================================
prereg loss 0.40944204 reg_l1 37.432133 reg_l2 23.43419
loss 4.1526556
STEP 63 ================================
prereg loss 0.40922284 reg_l1 37.429188 reg_l2 23.431763
loss 4.1521416
STEP 64 ================================
prereg loss 0.4090049 reg_l1 37.426243 reg_l2 23.429337
loss 4.151629
STEP 65 ================================
prereg loss 0.40878856 reg_l1 37.42328 reg_l2 23.4269
loss 4.1511164
STEP 66 ================================
prereg loss 0.4085667 reg_l1 37.42035 reg_l2 23.424486
loss 4.1506014
STEP 67 ================================
prereg loss 0.4083374 reg_l1 37.417427 reg_l2 23.422104
loss 4.15008
STEP 68 ================================
prereg loss 0.408102 reg_l1 37.414536 reg_l2 23.419765
loss 4.1495557
STEP 69 ================================
prereg loss 0.40785334 reg_l1 37.41169 reg_l2 23.417477
loss 4.149022
STEP 70 ================================
prereg loss 0.4075957 reg_l1 37.40889 reg_l2 23.415237
loss 4.1484847
STEP 71 ================================
prereg loss 0.40732732 reg_l1 37.406147 reg_l2 23.413057
loss 4.147942
STEP 72 ================================
prereg loss 0.40704718 reg_l1 37.403435 reg_l2 23.410933
loss 4.147391
STEP 73 ================================
prereg loss 0.4067599 reg_l1 37.400787 reg_l2 23.40886
loss 4.1468387
STEP 74 ================================
prereg loss 0.40646386 reg_l1 37.39817 reg_l2 23.406832
loss 4.1462812
STEP 75 ================================
prereg loss 0.40616232 reg_l1 37.39559 reg_l2 23.40485
loss 4.1457214
STEP 76 ================================
prereg loss 0.4058593 reg_l1 37.39304 reg_l2 23.402897
loss 4.1451635
STEP 77 ================================
prereg loss 0.40555564 reg_l1 37.390514 reg_l2 23.400959
loss 4.144607
STEP 78 ================================
prereg loss 0.40525132 reg_l1 37.387993 reg_l2 23.399036
loss 4.1440506
STEP 79 ================================
prereg loss 0.4049529 reg_l1 37.38548 reg_l2 23.39712
loss 4.143501
STEP 80 ================================
prereg loss 0.40465644 reg_l1 37.382957 reg_l2 23.395182
loss 4.1429524
STEP 81 ================================
prereg loss 0.40436643 reg_l1 37.38042 reg_l2 23.393242
loss 4.1424084
cutoff 0.2697117 network size 67
STEP 82 ================================
prereg loss 0.4296295 reg_l1 37.108143 reg_l2 23.31853
loss 4.140444
STEP 83 ================================
prereg loss 0.42932737 reg_l1 37.105846 reg_l2 23.316664
loss 4.139912
STEP 84 ================================
prereg loss 0.429108 reg_l1 37.103554 reg_l2 23.314768
loss 4.1394634
STEP 85 ================================
prereg loss 0.42905596 reg_l1 37.101227 reg_l2 23.312843
loss 4.1391788
STEP 86 ================================
prereg loss 0.4292503 reg_l1 37.09888 reg_l2 23.310852
loss 4.1391387
STEP 87 ================================
prereg loss 0.42971674 reg_l1 37.09644 reg_l2 23.308796
loss 4.139361
STEP 88 ================================
prereg loss 0.43043083 reg_l1 37.093906 reg_l2 23.306664
loss 4.1398215
STEP 89 ================================
prereg loss 0.43131745 reg_l1 37.091263 reg_l2 23.30444
loss 4.140444
STEP 90 ================================
prereg loss 0.4322672 reg_l1 37.08847 reg_l2 23.30213
loss 4.141114
STEP 91 ================================
prereg loss 0.4331693 reg_l1 37.085564 reg_l2 23.29974
loss 4.1417255
STEP 92 ================================
prereg loss 0.4339279 reg_l1 37.08255 reg_l2 23.297297
loss 4.142183
STEP 93 ================================
prereg loss 0.4344774 reg_l1 37.07942 reg_l2 23.294802
loss 4.1424193
STEP 94 ================================
prereg loss 0.43479902 reg_l1 37.076202 reg_l2 23.292282
loss 4.1424193
STEP 95 ================================
prereg loss 0.43490088 reg_l1 37.07294 reg_l2 23.289768
loss 4.142195
STEP 96 ================================
prereg loss 0.43484095 reg_l1 37.069656 reg_l2 23.28729
loss 4.1418066
STEP 97 ================================
prereg loss 0.43467432 reg_l1 37.06638 reg_l2 23.284866
loss 4.1413126
STEP 98 ================================
prereg loss 0.43446442 reg_l1 37.06314 reg_l2 23.282528
loss 4.1407785
STEP 99 ================================
prereg loss 0.43426093 reg_l1 37.059975 reg_l2 23.280296
loss 4.1402583
STEP 100 ================================
prereg loss 0.43410358 reg_l1 37.056915 reg_l2 23.278166
loss 4.1397953
STEP 101 ================================
prereg loss 0.43400782 reg_l1 37.053955 reg_l2 23.276167
loss 4.1394033
STEP 102 ================================
prereg loss 0.43397805 reg_l1 37.051105 reg_l2 23.274298
loss 4.1390886
STEP 103 ================================
prereg loss 0.43401462 reg_l1 37.04837 reg_l2 23.272532
loss 4.1388516
STEP 104 ================================
prereg loss 0.43410984 reg_l1 37.04574 reg_l2 23.270874
loss 4.138684
STEP 105 ================================
prereg loss 0.43425977 reg_l1 37.04321 reg_l2 23.269293
loss 4.138581
STEP 106 ================================
prereg loss 0.4344694 reg_l1 37.040752 reg_l2 23.26778
loss 4.1385446
STEP 107 ================================
prereg loss 0.43473548 reg_l1 37.038345 reg_l2 23.266306
loss 4.13857
STEP 108 ================================
prereg loss 0.43506894 reg_l1 37.03597 reg_l2 23.264835
loss 4.1386657
STEP 109 ================================
prereg loss 0.4354721 reg_l1 37.03356 reg_l2 23.26336
loss 4.1388283
STEP 110 ================================
prereg loss 0.43594575 reg_l1 37.031143 reg_l2 23.261835
loss 4.13906
STEP 111 ================================
prereg loss 0.43648383 reg_l1 37.028664 reg_l2 23.260258
loss 4.13935
STEP 112 ================================
prereg loss 0.43706277 reg_l1 37.026108 reg_l2 23.25859
loss 4.1396737
STEP 113 ================================
prereg loss 0.43767807 reg_l1 37.023476 reg_l2 23.256845
loss 4.1400256
STEP 114 ================================
prereg loss 0.4382968 reg_l1 37.020744 reg_l2 23.255005
loss 4.1403713
STEP 115 ================================
prereg loss 0.43890545 reg_l1 37.0179 reg_l2 23.25307
loss 4.140695
STEP 116 ================================
prereg loss 0.43948492 reg_l1 37.01497 reg_l2 23.251038
loss 4.1409817
STEP 117 ================================
prereg loss 0.44002378 reg_l1 37.01194 reg_l2 23.248934
loss 4.1412177
STEP 118 ================================
prereg loss 0.44051698 reg_l1 37.00884 reg_l2 23.246756
loss 4.141401
STEP 119 ================================
prereg loss 0.44096714 reg_l1 37.00567 reg_l2 23.244518
loss 4.1415343
STEP 120 ================================
prereg loss 0.44138277 reg_l1 37.00245 reg_l2 23.242254
loss 4.141628
STEP 121 ================================
prereg loss 0.44176537 reg_l1 36.9992 reg_l2 23.239986
loss 4.1416855
cutoff 0.28444636 network size 66
STEP 122 ================================
prereg loss 0.4421285 reg_l1 36.711502 reg_l2 23.156801
loss 4.113279
STEP 123 ================================
prereg loss 0.44248584 reg_l1 36.708622 reg_l2 23.154755
loss 4.113348
STEP 124 ================================
prereg loss 0.44283783 reg_l1 36.70578 reg_l2 23.152748
loss 4.1134157
STEP 125 ================================
prereg loss 0.4431945 reg_l1 36.702953 reg_l2 23.150787
loss 4.11349
STEP 126 ================================
prereg loss 0.44355857 reg_l1 36.70022 reg_l2 23.148897
loss 4.11358
STEP 127 ================================
prereg loss 0.4439318 reg_l1 36.697517 reg_l2 23.147062
loss 4.1136837
STEP 128 ================================
prereg loss 0.4443184 reg_l1 36.694885 reg_l2 23.145298
loss 4.113807
STEP 129 ================================
prereg loss 0.44471344 reg_l1 36.692333 reg_l2 23.14361
loss 4.113947
STEP 130 ================================
prereg loss 0.4451231 reg_l1 36.68983 reg_l2 23.14198
loss 4.114106
STEP 131 ================================
prereg loss 0.44554234 reg_l1 36.687386 reg_l2 23.14042
loss 4.1142807
STEP 132 ================================
prereg loss 0.44597137 reg_l1 36.68499 reg_l2 23.138899
loss 4.1144705
STEP 133 ================================
prereg loss 0.44641244 reg_l1 36.682632 reg_l2 23.13742
loss 4.1146755
STEP 134 ================================
prereg loss 0.44686452 reg_l1 36.680313 reg_l2 23.135983
loss 4.114896
STEP 135 ================================
prereg loss 0.44732088 reg_l1 36.677998 reg_l2 23.134552
loss 4.1151204
STEP 136 ================================
prereg loss 0.44777945 reg_l1 36.675705 reg_l2 23.133148
loss 4.1153502
STEP 137 ================================
prereg loss 0.44824186 reg_l1 36.673405 reg_l2 23.13174
loss 4.1155825
STEP 138 ================================
prereg loss 0.44869894 reg_l1 36.6711 reg_l2 23.130323
loss 4.115809
STEP 139 ================================
prereg loss 0.449151 reg_l1 36.668774 reg_l2 23.128902
loss 4.1160283
STEP 140 ================================
prereg loss 0.44959238 reg_l1 36.666428 reg_l2 23.127466
loss 4.1162353
STEP 141 ================================
prereg loss 0.45001975 reg_l1 36.66407 reg_l2 23.126009
loss 4.116427
STEP 142 ================================
prereg loss 0.4504388 reg_l1 36.661697 reg_l2 23.124542
loss 4.1166086
STEP 143 ================================
prereg loss 0.45084378 reg_l1 36.659294 reg_l2 23.123064
loss 4.116773
STEP 144 ================================
prereg loss 0.45123693 reg_l1 36.656883 reg_l2 23.121578
loss 4.1169252
STEP 145 ================================
prereg loss 0.45161778 reg_l1 36.654457 reg_l2 23.120087
loss 4.1170635
STEP 146 ================================
prereg loss 0.45199418 reg_l1 36.65203 reg_l2 23.118587
loss 4.117197
STEP 147 ================================
prereg loss 0.4523623 reg_l1 36.649605 reg_l2 23.117096
loss 4.117323
STEP 148 ================================
prereg loss 0.4527266 reg_l1 36.647175 reg_l2 23.11562
loss 4.117444
STEP 149 ================================
prereg loss 0.45309162 reg_l1 36.644768 reg_l2 23.114153
loss 4.1175685
STEP 150 ================================
prereg loss 0.45345554 reg_l1 36.642384 reg_l2 23.112709
loss 4.117694
STEP 151 ================================
prereg loss 0.45381764 reg_l1 36.640015 reg_l2 23.11129
loss 4.1178193
STEP 152 ================================
prereg loss 0.45418137 reg_l1 36.63767 reg_l2 23.109896
loss 4.117948
STEP 153 ================================
prereg loss 0.4545484 reg_l1 36.635338 reg_l2 23.108524
loss 4.1180825
STEP 154 ================================
prereg loss 0.45491517 reg_l1 36.63305 reg_l2 23.107193
loss 4.1182203
STEP 155 ================================
prereg loss 0.45528254 reg_l1 36.630783 reg_l2 23.105888
loss 4.118361
STEP 156 ================================
prereg loss 0.45564753 reg_l1 36.628536 reg_l2 23.104624
loss 4.118501
STEP 157 ================================
prereg loss 0.456013 reg_l1 36.626328 reg_l2 23.103376
loss 4.1186457
STEP 158 ================================
prereg loss 0.45637307 reg_l1 36.624138 reg_l2 23.102152
loss 4.118787
STEP 159 ================================
prereg loss 0.45673093 reg_l1 36.621964 reg_l2 23.100964
loss 4.1189275
STEP 160 ================================
prereg loss 0.45708537 reg_l1 36.619812 reg_l2 23.099787
loss 4.1190667
STEP 161 ================================
prereg loss 0.45743164 reg_l1 36.61767 reg_l2 23.098639
loss 4.1191983
cutoff 0.2902945 network size 65
STEP 162 ================================
prereg loss 0.45777357 reg_l1 36.325256 reg_l2 23.013224
loss 4.090299
STEP 163 ================================
prereg loss 0.45810446 reg_l1 36.323284 reg_l2 23.012186
loss 4.090433
STEP 164 ================================
prereg loss 0.45843115 reg_l1 36.321327 reg_l2 23.011164
loss 4.090564
STEP 165 ================================
prereg loss 0.4587453 reg_l1 36.319378 reg_l2 23.010155
loss 4.090683
STEP 166 ================================
prereg loss 0.45905727 reg_l1 36.317432 reg_l2 23.009144
loss 4.0908003
STEP 167 ================================
prereg loss 0.45935714 reg_l1 36.31549 reg_l2 23.008148
loss 4.090906
STEP 168 ================================
prereg loss 0.45964953 reg_l1 36.31355 reg_l2 23.007158
loss 4.0910044
STEP 169 ================================
prereg loss 0.4599366 reg_l1 36.311607 reg_l2 23.006178
loss 4.0910974
STEP 170 ================================
prereg loss 0.46021467 reg_l1 36.309692 reg_l2 23.0052
loss 4.091184
STEP 171 ================================
prereg loss 0.4604923 reg_l1 36.307766 reg_l2 23.004229
loss 4.091269
STEP 172 ================================
prereg loss 0.46076193 reg_l1 36.305836 reg_l2 23.003275
loss 4.0913453
STEP 173 ================================
prereg loss 0.4610296 reg_l1 36.30393 reg_l2 23.002327
loss 4.0914226
STEP 174 ================================
prereg loss 0.46129158 reg_l1 36.30202 reg_l2 23.001387
loss 4.0914936
STEP 175 ================================
prereg loss 0.46155253 reg_l1 36.300133 reg_l2 23.000458
loss 4.0915656
STEP 176 ================================
prereg loss 0.4618098 reg_l1 36.298252 reg_l2 22.99954
loss 4.091635
STEP 177 ================================
prereg loss 0.4620632 reg_l1 36.29637 reg_l2 22.998646
loss 4.0917006
STEP 178 ================================
prereg loss 0.46231323 reg_l1 36.2945 reg_l2 22.997753
loss 4.091763
STEP 179 ================================
prereg loss 0.46256173 reg_l1 36.292652 reg_l2 22.996878
loss 4.091827
STEP 180 ================================
prereg loss 0.46280345 reg_l1 36.2908 reg_l2 22.996016
loss 4.091883
STEP 181 ================================
prereg loss 0.4630461 reg_l1 36.288967 reg_l2 22.995161
loss 4.091943
STEP 182 ================================
prereg loss 0.46328118 reg_l1 36.28715 reg_l2 22.994324
loss 4.0919967
STEP 183 ================================
prereg loss 0.4635123 reg_l1 36.28533 reg_l2 22.993498
loss 4.0920453
STEP 184 ================================
prereg loss 0.46373683 reg_l1 36.283524 reg_l2 22.99269
loss 4.092089
STEP 185 ================================
prereg loss 0.46395832 reg_l1 36.281727 reg_l2 22.99189
loss 4.092131
STEP 186 ================================
prereg loss 0.46416938 reg_l1 36.279957 reg_l2 22.991096
loss 4.092165
STEP 187 ================================
prereg loss 0.46437877 reg_l1 36.278175 reg_l2 22.990316
loss 4.0921965
STEP 188 ================================
prereg loss 0.4645765 reg_l1 36.2764 reg_l2 22.989557
loss 4.092217
STEP 189 ================================
prereg loss 0.46477178 reg_l1 36.274635 reg_l2 22.988792
loss 4.0922356
STEP 190 ================================
prereg loss 0.46495834 reg_l1 36.272892 reg_l2 22.988045
loss 4.0922475
STEP 191 ================================
prereg loss 0.46513897 reg_l1 36.271133 reg_l2 22.987312
loss 4.0922523
STEP 192 ================================
prereg loss 0.46531194 reg_l1 36.269394 reg_l2 22.986584
loss 4.0922513
STEP 193 ================================
prereg loss 0.4654801 reg_l1 36.267666 reg_l2 22.985859
loss 4.092247
STEP 194 ================================
prereg loss 0.46563923 reg_l1 36.26593 reg_l2 22.985151
loss 4.092232
STEP 195 ================================
prereg loss 0.4657959 reg_l1 36.264206 reg_l2 22.98445
loss 4.0922165
STEP 196 ================================
prereg loss 0.46594682 reg_l1 36.262493 reg_l2 22.983753
loss 4.092196
STEP 197 ================================
prereg loss 0.46609086 reg_l1 36.260788 reg_l2 22.983067
loss 4.09217
STEP 198 ================================
prereg loss 0.4662297 reg_l1 36.259087 reg_l2 22.982391
loss 4.0921383
STEP 199 ================================
prereg loss 0.46636468 reg_l1 36.25739 reg_l2 22.981718
loss 4.0921035
STEP 200 ================================
prereg loss 0.46649265 reg_l1 36.255695 reg_l2 22.98106
loss 4.092062
2022-07-20T16:51:54.890

julia> serialize("cf-65-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-65-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T16:52:46.827
STEP 1 ================================
prereg loss 0.46661785 reg_l1 36.254005 reg_l2 22.980406
loss 4.0920186
cutoff 0.30853218 network size 64
STEP 2 ================================
prereg loss 4.695739 reg_l1 35.943798 reg_l2 22.884565
loss 8.290119
STEP 3 ================================
prereg loss 4.5706177 reg_l1 35.929478 reg_l2 22.869781
loss 8.163566
STEP 4 ================================
prereg loss 4.3580747 reg_l1 35.90474 reg_l2 22.843369
loss 7.9485483
STEP 5 ================================
prereg loss 4.102583 reg_l1 35.87186 reg_l2 22.807974
loss 7.689769
STEP 6 ================================
prereg loss 3.8449886 reg_l1 35.833103 reg_l2 22.766224
loss 7.428299
STEP 7 ================================
prereg loss 3.6159894 reg_l1 35.790596 reg_l2 22.7206
loss 7.1950493
STEP 8 ================================
prereg loss 3.4340243 reg_l1 35.746284 reg_l2 22.673351
loss 7.0086527
STEP 9 ================================
prereg loss 3.305662 reg_l1 35.70189 reg_l2 22.626478
loss 6.8758507
STEP 10 ================================
prereg loss 3.2605975 reg_l1 35.65891 reg_l2 22.581638
loss 6.8264885
STEP 11 ================================
prereg loss 3.2596962 reg_l1 35.62021 reg_l2 22.542032
loss 6.8217173
STEP 12 ================================
prereg loss 3.275718 reg_l1 35.586475 reg_l2 22.50834
loss 6.834366
STEP 13 ================================
prereg loss 3.2956052 reg_l1 35.558083 reg_l2 22.48098
loss 6.8514137
STEP 14 ================================
prereg loss 3.3088515 reg_l1 35.53526 reg_l2 22.46013
loss 6.862377
STEP 15 ================================
prereg loss 3.3080337 reg_l1 35.51801 reg_l2 22.44577
loss 6.8598347
STEP 16 ================================
prereg loss 3.2888486 reg_l1 35.506237 reg_l2 22.437696
loss 6.8394723
STEP 17 ================================
prereg loss 3.2497487 reg_l1 35.499664 reg_l2 22.435574
loss 6.799715
STEP 18 ================================
prereg loss 3.1915884 reg_l1 35.497936 reg_l2 22.43895
loss 6.741382
STEP 19 ================================
prereg loss 3.1169538 reg_l1 35.50061 reg_l2 22.447296
loss 6.667015
STEP 20 ================================
prereg loss 3.0295863 reg_l1 35.507183 reg_l2 22.460001
loss 6.5803046
STEP 21 ================================
prereg loss 2.9339275 reg_l1 35.51707 reg_l2 22.476402
loss 6.485635
STEP 22 ================================
prereg loss 2.8345468 reg_l1 35.529705 reg_l2 22.495808
loss 6.3875175
STEP 23 ================================
prereg loss 2.735814 reg_l1 35.544437 reg_l2 22.517525
loss 6.290258
STEP 24 ================================
prereg loss 2.6627028 reg_l1 35.56066 reg_l2 22.540823
loss 6.218769
STEP 25 ================================
prereg loss 2.6049833 reg_l1 35.576332 reg_l2 22.563496
loss 6.1626167
STEP 26 ================================
prereg loss 2.5562727 reg_l1 35.591015 reg_l2 22.585016
loss 6.1153746
STEP 27 ================================
prereg loss 2.5154972 reg_l1 35.604298 reg_l2 22.604929
loss 6.075927
STEP 28 ================================
prereg loss 2.4810636 reg_l1 35.61585 reg_l2 22.62285
loss 6.0426483
STEP 29 ================================
prereg loss 2.4510999 reg_l1 35.62541 reg_l2 22.638527
loss 6.0136404
STEP 30 ================================
prereg loss 2.4237432 reg_l1 35.6328 reg_l2 22.651741
loss 5.9870234
STEP 31 ================================
prereg loss 2.3973804 reg_l1 35.637955 reg_l2 22.662407
loss 5.961176
STEP 32 ================================
prereg loss 2.3707917 reg_l1 35.640842 reg_l2 22.670536
loss 5.934876
STEP 33 ================================
prereg loss 2.343196 reg_l1 35.64154 reg_l2 22.676208
loss 5.90735
STEP 34 ================================
prereg loss 2.3143263 reg_l1 35.640163 reg_l2 22.679586
loss 5.8783426
STEP 35 ================================
prereg loss 2.2842913 reg_l1 35.63693 reg_l2 22.680893
loss 5.8479843
STEP 36 ================================
prereg loss 2.253542 reg_l1 35.632065 reg_l2 22.680412
loss 5.8167486
STEP 37 ================================
prereg loss 2.2226539 reg_l1 35.625847 reg_l2 22.67845
loss 5.7852383
STEP 38 ================================
prereg loss 2.1922827 reg_l1 35.61856 reg_l2 22.67534
loss 5.754139
STEP 39 ================================
prereg loss 2.1630306 reg_l1 35.61052 reg_l2 22.671425
loss 5.724083
STEP 40 ================================
prereg loss 2.1353571 reg_l1 35.60201 reg_l2 22.667034
loss 5.695558
STEP 41 ================================
prereg loss 2.1095252 reg_l1 35.593327 reg_l2 22.662487
loss 5.6688576
cutoff 0.24533342 network size 63
STEP 42 ================================
prereg loss 1.9068984 reg_l1 35.339413 reg_l2 22.597887
loss 5.44084
STEP 43 ================================
prereg loss 1.8962617 reg_l1 35.335648 reg_l2 22.597755
loss 5.4298267
STEP 44 ================================
prereg loss 1.8788729 reg_l1 35.33456 reg_l2 22.600729
loss 5.4123287
STEP 45 ================================
prereg loss 1.855653 reg_l1 35.335945 reg_l2 22.606533
loss 5.389248
STEP 46 ================================
prereg loss 1.8279868 reg_l1 35.33954 reg_l2 22.614845
loss 5.361941
STEP 47 ================================
prereg loss 1.7974261 reg_l1 35.345078 reg_l2 22.62532
loss 5.331934
STEP 48 ================================
prereg loss 1.7655295 reg_l1 35.352253 reg_l2 22.637579
loss 5.300755
STEP 49 ================================
prereg loss 1.7337255 reg_l1 35.360744 reg_l2 22.651243
loss 5.2698
STEP 50 ================================
prereg loss 1.7032114 reg_l1 35.370228 reg_l2 22.665905
loss 5.2402344
STEP 51 ================================
prereg loss 1.674926 reg_l1 35.38035 reg_l2 22.681181
loss 5.212961
STEP 52 ================================
prereg loss 1.6495478 reg_l1 35.39079 reg_l2 22.696686
loss 5.188627
STEP 53 ================================
prereg loss 1.6274762 reg_l1 35.401222 reg_l2 22.71205
loss 5.1675987
STEP 54 ================================
prereg loss 1.6088628 reg_l1 35.411343 reg_l2 22.726952
loss 5.149997
STEP 55 ================================
prereg loss 1.5935987 reg_l1 35.420887 reg_l2 22.7411
loss 5.1356874
STEP 56 ================================
prereg loss 1.5813806 reg_l1 35.429596 reg_l2 22.754244
loss 5.12434
STEP 57 ================================
prereg loss 1.5717185 reg_l1 35.437275 reg_l2 22.76618
loss 5.115446
STEP 58 ================================
prereg loss 1.5639788 reg_l1 35.443768 reg_l2 22.77675
loss 5.1083555
STEP 59 ================================
prereg loss 1.5575091 reg_l1 35.448963 reg_l2 22.785877
loss 5.1024055
STEP 60 ================================
prereg loss 1.5516528 reg_l1 35.45284 reg_l2 22.793509
loss 5.0969367
STEP 61 ================================
prereg loss 1.5458641 reg_l1 35.455345 reg_l2 22.79967
loss 5.0913987
STEP 62 ================================
prereg loss 1.5397218 reg_l1 35.456547 reg_l2 22.804426
loss 5.0853767
STEP 63 ================================
prereg loss 1.5329844 reg_l1 35.45652 reg_l2 22.80787
loss 5.0786366
STEP 64 ================================
prereg loss 1.5256002 reg_l1 35.455383 reg_l2 22.810146
loss 5.0711384
STEP 65 ================================
prereg loss 1.5176703 reg_l1 35.453278 reg_l2 22.81143
loss 5.0629983
STEP 66 ================================
prereg loss 1.5094181 reg_l1 35.45039 reg_l2 22.811913
loss 5.054457
STEP 67 ================================
prereg loss 1.501119 reg_l1 35.446884 reg_l2 22.811783
loss 5.0458074
STEP 68 ================================
prereg loss 1.4930576 reg_l1 35.442963 reg_l2 22.811243
loss 5.037354
STEP 69 ================================
prereg loss 1.4854714 reg_l1 35.438793 reg_l2 22.81049
loss 5.0293508
STEP 70 ================================
prereg loss 1.4785246 reg_l1 35.43457 reg_l2 22.809721
loss 5.0219817
STEP 71 ================================
prereg loss 1.4722743 reg_l1 35.43044 reg_l2 22.80908
loss 5.015318
STEP 72 ================================
prereg loss 1.4666942 reg_l1 35.42655 reg_l2 22.808727
loss 5.0093493
STEP 73 ================================
prereg loss 1.4616815 reg_l1 35.423023 reg_l2 22.808775
loss 5.003984
STEP 74 ================================
prereg loss 1.4570867 reg_l1 35.419952 reg_l2 22.809315
loss 4.999082
STEP 75 ================================
prereg loss 1.4527296 reg_l1 35.417385 reg_l2 22.810413
loss 4.994468
STEP 76 ================================
prereg loss 1.4484521 reg_l1 35.415367 reg_l2 22.812105
loss 4.989989
STEP 77 ================================
prereg loss 1.4441336 reg_l1 35.413925 reg_l2 22.814407
loss 4.985526
STEP 78 ================================
prereg loss 1.4396789 reg_l1 35.413055 reg_l2 22.817312
loss 4.9809847
STEP 79 ================================
prereg loss 1.4350498 reg_l1 35.4127 reg_l2 22.820787
loss 4.97632
STEP 80 ================================
prereg loss 1.4302446 reg_l1 35.412838 reg_l2 22.824795
loss 4.9715285
STEP 81 ================================
prereg loss 1.4252925 reg_l1 35.413403 reg_l2 22.829254
loss 4.966633
cutoff 0.27057976 network size 62
STEP 82 ================================
prereg loss 1.5812334 reg_l1 35.14373 reg_l2 22.760902
loss 5.0956063
STEP 83 ================================
prereg loss 1.5485709 reg_l1 35.149986 reg_l2 22.771473
loss 5.0635695
STEP 84 ================================
prereg loss 1.4975737 reg_l1 35.159996 reg_l2 22.78653
loss 5.013573
STEP 85 ================================
prereg loss 1.4342469 reg_l1 35.173183 reg_l2 22.805346
loss 4.9515653
STEP 86 ================================
prereg loss 1.363654 reg_l1 35.188953 reg_l2 22.827194
loss 4.8825493
STEP 87 ================================
prereg loss 1.2892991 reg_l1 35.206978 reg_l2 22.851713
loss 4.809997
STEP 88 ================================
prereg loss 1.217844 reg_l1 35.226707 reg_l2 22.878193
loss 4.7405148
STEP 89 ================================
prereg loss 1.1536502 reg_l1 35.24758 reg_l2 22.90594
loss 4.678408
STEP 90 ================================
prereg loss 1.1050819 reg_l1 35.26875 reg_l2 22.93392
loss 4.631957
STEP 91 ================================
prereg loss 1.0676183 reg_l1 35.289707 reg_l2 22.961521
loss 4.596589
STEP 92 ================================
prereg loss 1.0410975 reg_l1 35.309982 reg_l2 22.988174
loss 4.572096
STEP 93 ================================
prereg loss 1.0245594 reg_l1 35.329147 reg_l2 23.013367
loss 4.557474
STEP 94 ================================
prereg loss 1.0165128 reg_l1 35.3468 reg_l2 23.036636
loss 4.5511928
STEP 95 ================================
prereg loss 1.0151488 reg_l1 35.36262 reg_l2 23.057632
loss 4.551411
STEP 96 ================================
prereg loss 1.0185319 reg_l1 35.37634 reg_l2 23.076067
loss 4.5561657
STEP 97 ================================
prereg loss 1.0247811 reg_l1 35.387753 reg_l2 23.091736
loss 4.563556
STEP 98 ================================
prereg loss 1.0321478 reg_l1 35.39674 reg_l2 23.104542
loss 4.5718217
STEP 99 ================================
prereg loss 1.0391445 reg_l1 35.403255 reg_l2 23.114487
loss 4.57947
STEP 100 ================================
prereg loss 1.044706 reg_l1 35.407333 reg_l2 23.121634
loss 4.585439
STEP 101 ================================
prereg loss 1.0479789 reg_l1 35.409054 reg_l2 23.126148
loss 4.5888844
STEP 102 ================================
prereg loss 1.048642 reg_l1 35.408592 reg_l2 23.12824
loss 4.5895014
STEP 103 ================================
prereg loss 1.0467802 reg_l1 35.406178 reg_l2 23.128218
loss 4.587398
STEP 104 ================================
prereg loss 1.0428468 reg_l1 35.40209 reg_l2 23.126411
loss 4.5830555
STEP 105 ================================
prereg loss 1.0375648 reg_l1 35.396626 reg_l2 23.123188
loss 4.5772276
STEP 106 ================================
prereg loss 1.0318087 reg_l1 35.39012 reg_l2 23.11891
loss 4.570821
STEP 107 ================================
prereg loss 1.0264305 reg_l1 35.382927 reg_l2 23.113964
loss 4.564723
STEP 108 ================================
prereg loss 1.022168 reg_l1 35.37535 reg_l2 23.108717
loss 4.5597034
STEP 109 ================================
prereg loss 1.0195271 reg_l1 35.36771 reg_l2 23.103485
loss 4.5562983
STEP 110 ================================
prereg loss 1.0187256 reg_l1 35.36029 reg_l2 23.098576
loss 4.5547547
STEP 111 ================================
prereg loss 1.0196972 reg_l1 35.353344 reg_l2 23.094248
loss 4.555032
STEP 112 ================================
prereg loss 1.0221301 reg_l1 35.34709 reg_l2 23.090717
loss 4.5568395
STEP 113 ================================
prereg loss 1.0255383 reg_l1 35.341682 reg_l2 23.088129
loss 4.5597067
STEP 114 ================================
prereg loss 1.029361 reg_l1 35.337223 reg_l2 23.086607
loss 4.5630836
STEP 115 ================================
prereg loss 1.0330628 reg_l1 35.333805 reg_l2 23.086216
loss 4.5664434
STEP 116 ================================
prereg loss 1.0361974 reg_l1 35.331448 reg_l2 23.086967
loss 4.569342
STEP 117 ================================
prereg loss 1.0384686 reg_l1 35.330128 reg_l2 23.088854
loss 4.5714817
STEP 118 ================================
prereg loss 1.0397502 reg_l1 35.329807 reg_l2 23.091805
loss 4.572731
STEP 119 ================================
prereg loss 1.0400572 reg_l1 35.33039 reg_l2 23.095728
loss 4.5730963
STEP 120 ================================
prereg loss 1.0395269 reg_l1 35.33178 reg_l2 23.100525
loss 4.5727053
STEP 121 ================================
prereg loss 1.0383741 reg_l1 35.333855 reg_l2 23.106054
loss 4.5717597
cutoff 0.32622364 network size 61
STEP 122 ================================
prereg loss 1.6031952 reg_l1 35.01023 reg_l2 23.00576
loss 5.1042185
STEP 123 ================================
prereg loss 1.52656 reg_l1 35.015907 reg_l2 23.01403
loss 5.0281506
STEP 124 ================================
prereg loss 1.4143935 reg_l1 35.023983 reg_l2 23.023966
loss 4.916792
STEP 125 ================================
prereg loss 1.3009094 reg_l1 35.033768 reg_l2 23.035095
loss 4.804286
STEP 126 ================================
prereg loss 1.2158622 reg_l1 35.04454 reg_l2 23.046942
loss 4.7203164
STEP 127 ================================
prereg loss 1.1775194 reg_l1 35.0556 reg_l2 23.05902
loss 4.6830792
STEP 128 ================================
prereg loss 1.1894609 reg_l1 35.066254 reg_l2 23.070852
loss 4.6960864
STEP 129 ================================
prereg loss 1.2419295 reg_l1 35.075882 reg_l2 23.081997
loss 4.7495174
STEP 130 ================================
prereg loss 1.3162966 reg_l1 35.083996 reg_l2 23.092077
loss 4.8246965
STEP 131 ================================
prereg loss 1.3910478 reg_l1 35.090218 reg_l2 23.10079
loss 4.9000697
STEP 132 ================================
prereg loss 1.4499546 reg_l1 35.094322 reg_l2 23.107998
loss 4.959387
STEP 133 ================================
prereg loss 1.4767846 reg_l1 35.095486 reg_l2 23.112654
loss 4.9863334
STEP 134 ================================
prereg loss 1.4666922 reg_l1 35.093887 reg_l2 23.11496
loss 4.976081
STEP 135 ================================
prereg loss 1.4269971 reg_l1 35.0899 reg_l2 23.115234
loss 4.935987
STEP 136 ================================
prereg loss 1.3734461 reg_l1 35.084785 reg_l2 23.114822
loss 4.8819246
STEP 137 ================================
prereg loss 1.3153985 reg_l1 35.078945 reg_l2 23.114012
loss 4.8232927
STEP 138 ================================
prereg loss 1.2632769 reg_l1 35.072792 reg_l2 23.113106
loss 4.770556
STEP 139 ================================
prereg loss 1.2239426 reg_l1 35.066708 reg_l2 23.112383
loss 4.730613
STEP 140 ================================
prereg loss 1.1999109 reg_l1 35.06109 reg_l2 23.11209
loss 4.70602
STEP 141 ================================
prereg loss 1.1897763 reg_l1 35.056206 reg_l2 23.112434
loss 4.695397
STEP 142 ================================
prereg loss 1.1895021 reg_l1 35.052284 reg_l2 23.113567
loss 4.6947308
STEP 143 ================================
prereg loss 1.1941136 reg_l1 35.049522 reg_l2 23.115582
loss 4.6990657
STEP 144 ================================
prereg loss 1.1992344 reg_l1 35.04794 reg_l2 23.118505
loss 4.704028
STEP 145 ================================
prereg loss 1.202101 reg_l1 35.04758 reg_l2 23.122345
loss 4.706859
STEP 146 ================================
prereg loss 1.2019415 reg_l1 35.04835 reg_l2 23.127033
loss 4.7067766
STEP 147 ================================
prereg loss 1.1997043 reg_l1 35.050117 reg_l2 23.13248
loss 4.704716
STEP 148 ================================
prereg loss 1.1973671 reg_l1 35.052734 reg_l2 23.138577
loss 4.7026405
STEP 149 ================================
prereg loss 1.197045 reg_l1 35.05598 reg_l2 23.145168
loss 4.702643
STEP 150 ================================
prereg loss 1.2002678 reg_l1 35.05964 reg_l2 23.152128
loss 4.706232
STEP 151 ================================
prereg loss 1.2075224 reg_l1 35.063534 reg_l2 23.15929
loss 4.713876
STEP 152 ================================
prereg loss 1.2181307 reg_l1 35.06742 reg_l2 23.166529
loss 4.724873
STEP 153 ================================
prereg loss 1.2305213 reg_l1 35.071175 reg_l2 23.173723
loss 4.737639
STEP 154 ================================
prereg loss 1.2426629 reg_l1 35.074627 reg_l2 23.180769
loss 4.750126
STEP 155 ================================
prereg loss 1.252592 reg_l1 35.077694 reg_l2 23.187588
loss 4.760361
STEP 156 ================================
prereg loss 1.2588664 reg_l1 35.080322 reg_l2 23.194138
loss 4.7668986
STEP 157 ================================
prereg loss 1.2608064 reg_l1 35.08249 reg_l2 23.200396
loss 4.7690554
STEP 158 ================================
prereg loss 1.258561 reg_l1 35.084225 reg_l2 23.206383
loss 4.7669835
STEP 159 ================================
prereg loss 1.2529634 reg_l1 35.085567 reg_l2 23.212122
loss 4.7615204
STEP 160 ================================
prereg loss 1.2452395 reg_l1 35.086617 reg_l2 23.217653
loss 4.7539015
STEP 161 ================================
prereg loss 1.2367319 reg_l1 35.087452 reg_l2 23.22304
loss 4.745477
cutoff 0.32536834 network size 60
STEP 162 ================================
prereg loss 1.2285898 reg_l1 34.762806 reg_l2 23.122475
loss 4.70487
STEP 163 ================================
prereg loss 1.221638 reg_l1 34.763615 reg_l2 23.127829
loss 4.6979995
STEP 164 ================================
prereg loss 1.2162836 reg_l1 34.764515 reg_l2 23.133215
loss 4.692735
STEP 165 ================================
prereg loss 1.2125982 reg_l1 34.76554 reg_l2 23.138683
loss 4.6891522
STEP 166 ================================
prereg loss 1.2103968 reg_l1 34.76679 reg_l2 23.144272
loss 4.6870756
STEP 167 ================================
prereg loss 1.2094088 reg_l1 34.76827 reg_l2 23.15
loss 4.6862354
STEP 168 ================================
prereg loss 1.209363 reg_l1 34.769997 reg_l2 23.155884
loss 4.6863627
STEP 169 ================================
prereg loss 1.2100751 reg_l1 34.771984 reg_l2 23.16192
loss 4.6872735
STEP 170 ================================
prereg loss 1.211446 reg_l1 34.774204 reg_l2 23.16811
loss 4.6888666
STEP 171 ================================
prereg loss 1.2134675 reg_l1 34.776627 reg_l2 23.174423
loss 4.69113
STEP 172 ================================
prereg loss 1.21612 reg_l1 34.779186 reg_l2 23.180836
loss 4.6940384
STEP 173 ================================
prereg loss 1.2193499 reg_l1 34.781857 reg_l2 23.187325
loss 4.6975355
STEP 174 ================================
prereg loss 1.2230227 reg_l1 34.78459 reg_l2 23.193861
loss 4.701482
STEP 175 ================================
prereg loss 1.2269207 reg_l1 34.787308 reg_l2 23.200417
loss 4.7056518
STEP 176 ================================
prereg loss 1.2307497 reg_l1 34.78999 reg_l2 23.206953
loss 4.7097487
STEP 177 ================================
prereg loss 1.2342066 reg_l1 34.792576 reg_l2 23.213469
loss 4.7134643
STEP 178 ================================
prereg loss 1.2370143 reg_l1 34.79508 reg_l2 23.219936
loss 4.716522
STEP 179 ================================
prereg loss 1.2389752 reg_l1 34.79745 reg_l2 23.226355
loss 4.7187204
STEP 180 ================================
prereg loss 1.2400035 reg_l1 34.799717 reg_l2 23.232716
loss 4.719975
STEP 181 ================================
prereg loss 1.2401175 reg_l1 34.80187 reg_l2 23.239033
loss 4.7203045
STEP 182 ================================
prereg loss 1.2394445 reg_l1 34.80391 reg_l2 23.245306
loss 4.7198353
STEP 183 ================================
prereg loss 1.2381767 reg_l1 34.805878 reg_l2 23.251554
loss 4.7187643
STEP 184 ================================
prereg loss 1.2366881 reg_l1 34.807796 reg_l2 23.257786
loss 4.717468
STEP 185 ================================
prereg loss 1.2348341 reg_l1 34.80896 reg_l2 23.263168
loss 4.71573
STEP 186 ================================
prereg loss 1.2332084 reg_l1 34.81023 reg_l2 23.268673
loss 4.7142315
STEP 187 ================================
prereg loss 1.2317865 reg_l1 34.811626 reg_l2 23.274324
loss 4.7129493
STEP 188 ================================
prereg loss 1.2306492 reg_l1 34.813183 reg_l2 23.280138
loss 4.7119675
STEP 189 ================================
prereg loss 1.22983 reg_l1 34.814907 reg_l2 23.28613
loss 4.711321
STEP 190 ================================
prereg loss 1.2293268 reg_l1 34.81681 reg_l2 23.292288
loss 4.711008
STEP 191 ================================
prereg loss 1.2291297 reg_l1 34.81889 reg_l2 23.298607
loss 4.7110186
STEP 192 ================================
prereg loss 1.2292162 reg_l1 34.821125 reg_l2 23.305086
loss 4.711329
STEP 193 ================================
prereg loss 1.2295609 reg_l1 34.823513 reg_l2 23.311703
loss 4.711912
STEP 194 ================================
prereg loss 1.2301195 reg_l1 34.826023 reg_l2 23.318438
loss 4.712722
STEP 195 ================================
prereg loss 1.2314695 reg_l1 34.828625 reg_l2 23.325256
loss 4.714332
STEP 196 ================================
prereg loss 1.232236 reg_l1 34.830585 reg_l2 23.33131
loss 4.715295
STEP 197 ================================
prereg loss 1.2330225 reg_l1 34.83196 reg_l2 23.336676
loss 4.7162185
STEP 198 ================================
prereg loss 1.2341087 reg_l1 34.83352 reg_l2 23.342287
loss 4.7174606
STEP 199 ================================
prereg loss 1.2350385 reg_l1 34.835274 reg_l2 23.348124
loss 4.718566
STEP 200 ================================
prereg loss 1.2357553 reg_l1 34.83721 reg_l2 23.354181
loss 4.7194767
2022-07-20T17:04:20.991

julia> serialize("cf-60-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-60-parameters-opt.ser", opt)
```

Continuing: successful sparsification until a good model of 50 parameters,
a somewhat good model of 45 parameters, then divergence.

Need to backtrack to 45 or 50 parameters and continue, and/or redo more systematically:

```
julia> interleaving_steps!(200, 40)
2022-07-20T17:11:46.511
STEP 1 ================================
prereg loss 1.2362021 reg_l1 34.839283 reg_l2 23.360443
loss 4.7201304
cutoff 0.31680486 network size 59
STEP 2 ================================
prereg loss 1.2960678 reg_l1 34.524704 reg_l2 23.266527
loss 4.7485385
STEP 3 ================================
prereg loss 1.2926652 reg_l1 34.527367 reg_l2 23.272896
loss 4.745402
STEP 4 ================================
prereg loss 1.2885875 reg_l1 34.530083 reg_l2 23.278992
loss 4.7415957
STEP 5 ================================
prereg loss 1.2862659 reg_l1 34.5328 reg_l2 23.284834
loss 4.739546
STEP 6 ================================
prereg loss 1.2876332 reg_l1 34.535442 reg_l2 23.290438
loss 4.7411776
STEP 7 ================================
prereg loss 1.293872 reg_l1 34.537926 reg_l2 23.295807
loss 4.7476645
STEP 8 ================================
prereg loss 1.3049934 reg_l1 34.54089 reg_l2 23.301792
loss 4.7590823
STEP 9 ================================
prereg loss 1.3184874 reg_l1 34.544212 reg_l2 23.30831
loss 4.7729087
STEP 10 ================================
prereg loss 1.3320233 reg_l1 34.547752 reg_l2 23.315262
loss 4.7867985
STEP 11 ================================
prereg loss 1.3431693 reg_l1 34.551414 reg_l2 23.322577
loss 4.7983108
STEP 12 ================================
prereg loss 1.3499765 reg_l1 34.555088 reg_l2 23.330177
loss 4.8054857
STEP 13 ================================
prereg loss 1.3514372 reg_l1 34.558743 reg_l2 23.338003
loss 4.8073115
STEP 14 ================================
prereg loss 1.3475734 reg_l1 34.562332 reg_l2 23.34601
loss 4.803807
STEP 15 ================================
prereg loss 1.3393284 reg_l1 34.56588 reg_l2 23.354156
loss 4.7959166
STEP 16 ================================
prereg loss 1.3282195 reg_l1 34.569374 reg_l2 23.362427
loss 4.785157
STEP 17 ================================
prereg loss 1.315952 reg_l1 34.572857 reg_l2 23.370802
loss 4.7732377
STEP 18 ================================
prereg loss 1.3040103 reg_l1 34.576363 reg_l2 23.37926
loss 4.7616467
STEP 19 ================================
prereg loss 1.2934676 reg_l1 34.57992 reg_l2 23.387812
loss 4.75146
STEP 20 ================================
prereg loss 1.284869 reg_l1 34.583565 reg_l2 23.39642
loss 4.7432256
STEP 21 ================================
prereg loss 1.2783043 reg_l1 34.587307 reg_l2 23.405079
loss 4.737035
STEP 22 ================================
prereg loss 1.2735647 reg_l1 34.591175 reg_l2 23.41379
loss 4.732682
STEP 23 ================================
prereg loss 1.2703261 reg_l1 34.59516 reg_l2 23.422512
loss 4.729842
STEP 24 ================================
prereg loss 1.2683015 reg_l1 34.599247 reg_l2 23.431248
loss 4.728226
STEP 25 ================================
prereg loss 1.2673088 reg_l1 34.60343 reg_l2 23.43995
loss 4.727652
STEP 26 ================================
prereg loss 1.2673074 reg_l1 34.607655 reg_l2 23.448624
loss 4.728073
STEP 27 ================================
prereg loss 1.2682945 reg_l1 34.611885 reg_l2 23.457212
loss 4.729483
STEP 28 ================================
prereg loss 1.2702641 reg_l1 34.616085 reg_l2 23.465714
loss 4.7318726
STEP 29 ================================
prereg loss 1.2737182 reg_l1 34.6202 reg_l2 23.474104
loss 4.7357383
STEP 30 ================================
prereg loss 1.2777308 reg_l1 34.623512 reg_l2 23.481546
loss 4.7400823
STEP 31 ================================
prereg loss 1.2811921 reg_l1 34.626057 reg_l2 23.488144
loss 4.743798
STEP 32 ================================
prereg loss 1.2842915 reg_l1 34.6279 reg_l2 23.493988
loss 4.7470818
STEP 33 ================================
prereg loss 1.287347 reg_l1 34.629852 reg_l2 23.50002
loss 4.7503324
STEP 34 ================================
prereg loss 1.2894391 reg_l1 34.63188 reg_l2 23.50623
loss 4.7526274
STEP 35 ================================
prereg loss 1.2903777 reg_l1 34.634033 reg_l2 23.51266
loss 4.7537813
STEP 36 ================================
prereg loss 1.2901305 reg_l1 34.636307 reg_l2 23.519302
loss 4.7537613
STEP 37 ================================
prereg loss 1.2887801 reg_l1 34.638702 reg_l2 23.526175
loss 4.7526503
STEP 38 ================================
prereg loss 1.2865345 reg_l1 34.64125 reg_l2 23.533278
loss 4.75066
STEP 39 ================================
prereg loss 1.2836435 reg_l1 34.64397 reg_l2 23.540613
loss 4.7480407
STEP 40 ================================
prereg loss 1.280389 reg_l1 34.646847 reg_l2 23.54818
loss 4.745074
STEP 41 ================================
prereg loss 1.2769963 reg_l1 34.649902 reg_l2 23.555956
loss 4.7419868
cutoff 0.32591334 network size 58
STEP 42 ================================
prereg loss 1.453487 reg_l1 34.327232 reg_l2 23.457722
loss 4.8862104
STEP 43 ================================
prereg loss 1.4412516 reg_l1 34.33085 reg_l2 23.465458
loss 4.8743367
STEP 44 ================================
prereg loss 1.419904 reg_l1 34.33475 reg_l2 23.472967
loss 4.8533792
STEP 45 ================================
prereg loss 1.3985102 reg_l1 34.338837 reg_l2 23.480253
loss 4.8323936
STEP 46 ================================
prereg loss 1.3855686 reg_l1 34.342987 reg_l2 23.487354
loss 4.819867
STEP 47 ================================
prereg loss 1.3862555 reg_l1 34.347095 reg_l2 23.49428
loss 4.820965
STEP 48 ================================
prereg loss 1.401133 reg_l1 34.35107 reg_l2 23.501074
loss 4.83624
STEP 49 ================================
prereg loss 1.4264035 reg_l1 34.354782 reg_l2 23.507738
loss 4.8618817
STEP 50 ================================
prereg loss 1.4554017 reg_l1 34.35818 reg_l2 23.514294
loss 4.89122
STEP 51 ================================
prereg loss 1.480708 reg_l1 34.36122 reg_l2 23.520775
loss 4.91683
STEP 52 ================================
prereg loss 1.496215 reg_l1 34.363895 reg_l2 23.527195
loss 4.932605
STEP 53 ================================
prereg loss 1.5009829 reg_l1 34.36622 reg_l2 23.533594
loss 4.937605
STEP 54 ================================
prereg loss 1.493752 reg_l1 34.368935 reg_l2 23.540794
loss 4.9306455
STEP 55 ================================
prereg loss 1.4760396 reg_l1 34.37201 reg_l2 23.54874
loss 4.9132404
STEP 56 ================================
prereg loss 1.451749 reg_l1 34.375443 reg_l2 23.557352
loss 4.889293
STEP 57 ================================
prereg loss 1.4252659 reg_l1 34.37924 reg_l2 23.56656
loss 4.8631897
STEP 58 ================================
prereg loss 1.4003396 reg_l1 34.3834 reg_l2 23.576298
loss 4.8386793
STEP 59 ================================
prereg loss 1.3794032 reg_l1 34.38792 reg_l2 23.586466
loss 4.8181953
STEP 60 ================================
prereg loss 1.3633779 reg_l1 34.392807 reg_l2 23.597
loss 4.8026586
STEP 61 ================================
prereg loss 1.351897 reg_l1 34.398006 reg_l2 23.607805
loss 4.7916975
STEP 62 ================================
prereg loss 1.3438513 reg_l1 34.40351 reg_l2 23.618809
loss 4.7842026
STEP 63 ================================
prereg loss 1.3379714 reg_l1 34.409264 reg_l2 23.629921
loss 4.778898
STEP 64 ================================
prereg loss 1.3333211 reg_l1 34.415222 reg_l2 23.64107
loss 4.774843
STEP 65 ================================
prereg loss 1.3295401 reg_l1 34.421314 reg_l2 23.652191
loss 4.771672
STEP 66 ================================
prereg loss 1.3268014 reg_l1 34.427452 reg_l2 23.66322
loss 4.7695465
STEP 67 ================================
prereg loss 1.3255858 reg_l1 34.433598 reg_l2 23.674116
loss 4.7689457
STEP 68 ================================
prereg loss 1.3263763 reg_l1 34.439648 reg_l2 23.684828
loss 4.770341
STEP 69 ================================
prereg loss 1.3293465 reg_l1 34.44554 reg_l2 23.695318
loss 4.7739005
STEP 70 ================================
prereg loss 1.3342003 reg_l1 34.451218 reg_l2 23.70557
loss 4.779322
STEP 71 ================================
prereg loss 1.3401893 reg_l1 34.45663 reg_l2 23.71558
loss 4.7858524
STEP 72 ================================
prereg loss 1.3462437 reg_l1 34.461754 reg_l2 23.725328
loss 4.792419
STEP 73 ================================
prereg loss 1.3512396 reg_l1 34.46655 reg_l2 23.734821
loss 4.7978945
STEP 74 ================================
prereg loss 1.3542281 reg_l1 34.471035 reg_l2 23.744066
loss 4.8013315
STEP 75 ================================
prereg loss 1.3553787 reg_l1 34.475197 reg_l2 23.753086
loss 4.8028984
STEP 76 ================================
prereg loss 1.3532089 reg_l1 34.47838 reg_l2 23.761122
loss 4.801047
STEP 77 ================================
prereg loss 1.3478225 reg_l1 34.480736 reg_l2 23.768288
loss 4.795896
STEP 78 ================================
prereg loss 1.3416704 reg_l1 34.482365 reg_l2 23.774738
loss 4.789907
STEP 79 ================================
prereg loss 1.3348851 reg_l1 34.484108 reg_l2 23.781372
loss 4.7832956
STEP 80 ================================
prereg loss 1.3281088 reg_l1 34.48603 reg_l2 23.788242
loss 4.776712
STEP 81 ================================
prereg loss 1.3218368 reg_l1 34.488167 reg_l2 23.795374
loss 4.7706537
cutoff 0.33085847 network size 57
STEP 82 ================================
prereg loss 4.0300045 reg_l1 34.159706 reg_l2 23.693335
loss 7.4459753
STEP 83 ================================
prereg loss 3.9488144 reg_l1 34.152687 reg_l2 23.689667
loss 7.3640833
STEP 84 ================================
prereg loss 3.8223248 reg_l1 34.137615 reg_l2 23.676582
loss 7.2360864
STEP 85 ================================
prereg loss 3.6696146 reg_l1 34.11595 reg_l2 23.655855
loss 7.0812097
STEP 86 ================================
prereg loss 3.5227416 reg_l1 34.08882 reg_l2 23.62882
loss 6.9316235
STEP 87 ================================
prereg loss 3.3231983 reg_l1 34.053032 reg_l2 23.592625
loss 6.7285013
STEP 88 ================================
prereg loss 3.174445 reg_l1 34.01388 reg_l2 23.55239
loss 6.5758333
STEP 89 ================================
prereg loss 3.0633857 reg_l1 33.97329 reg_l2 23.510586
loss 6.460715
STEP 90 ================================
prereg loss 2.990513 reg_l1 33.932545 reg_l2 23.468801
loss 6.3837676
STEP 91 ================================
prereg loss 2.9492314 reg_l1 33.893257 reg_l2 23.429054
loss 6.3385572
STEP 92 ================================
prereg loss 2.9259858 reg_l1 33.8567 reg_l2 23.3929
loss 6.311656
STEP 93 ================================
prereg loss 2.9096487 reg_l1 33.823883 reg_l2 23.36151
loss 6.292037
STEP 94 ================================
prereg loss 2.8934762 reg_l1 33.7955 reg_l2 23.335646
loss 6.2730265
STEP 95 ================================
prereg loss 2.8747861 reg_l1 33.772034 reg_l2 23.315727
loss 6.2519894
STEP 96 ================================
prereg loss 2.853251 reg_l1 33.75367 reg_l2 23.301878
loss 6.2286177
STEP 97 ================================
prereg loss 2.8289375 reg_l1 33.74044 reg_l2 23.293947
loss 6.202982
STEP 98 ================================
prereg loss 2.800909 reg_l1 33.732166 reg_l2 23.29159
loss 6.1741257
STEP 99 ================================
prereg loss 2.7668898 reg_l1 33.728535 reg_l2 23.294304
loss 6.1397433
STEP 100 ================================
prereg loss 2.7238173 reg_l1 33.729084 reg_l2 23.301445
loss 6.0967255
STEP 101 ================================
prereg loss 2.6691067 reg_l1 33.73328 reg_l2 23.312311
loss 6.0424347
STEP 102 ================================
prereg loss 2.6018453 reg_l1 33.740505 reg_l2 23.326149
loss 5.975896
STEP 103 ================================
prereg loss 2.5235703 reg_l1 33.7501 reg_l2 23.342176
loss 5.8985806
STEP 104 ================================
prereg loss 2.438388 reg_l1 33.761375 reg_l2 23.359648
loss 5.8145256
STEP 105 ================================
prereg loss 2.352172 reg_l1 33.77368 reg_l2 23.377842
loss 5.72954
STEP 106 ================================
prereg loss 2.2713823 reg_l1 33.786373 reg_l2 23.39611
loss 5.6500196
STEP 107 ================================
prereg loss 2.2016606 reg_l1 33.79886 reg_l2 23.413841
loss 5.581547
STEP 108 ================================
prereg loss 2.1466026 reg_l1 33.810608 reg_l2 23.430532
loss 5.527663
STEP 109 ================================
prereg loss 2.1071036 reg_l1 33.821167 reg_l2 23.445736
loss 5.4892206
STEP 110 ================================
prereg loss 2.0813587 reg_l1 33.830162 reg_l2 23.45913
loss 5.464375
STEP 111 ================================
prereg loss 2.065409 reg_l1 33.837296 reg_l2 23.470436
loss 5.4491386
STEP 112 ================================
prereg loss 2.0542617 reg_l1 33.842392 reg_l2 23.479496
loss 5.438501
STEP 113 ================================
prereg loss 2.0429537 reg_l1 33.845333 reg_l2 23.486237
loss 5.4274874
STEP 114 ================================
prereg loss 2.0276525 reg_l1 33.846123 reg_l2 23.490664
loss 5.412265
STEP 115 ================================
prereg loss 2.0061786 reg_l1 33.84482 reg_l2 23.492851
loss 5.3906603
STEP 116 ================================
prereg loss 1.9782486 reg_l1 33.84157 reg_l2 23.49295
loss 5.362406
STEP 117 ================================
prereg loss 1.9451224 reg_l1 33.836563 reg_l2 23.49116
loss 5.3287787
STEP 118 ================================
prereg loss 1.9090341 reg_l1 33.830044 reg_l2 23.487709
loss 5.2920384
STEP 119 ================================
prereg loss 1.872551 reg_l1 33.822315 reg_l2 23.48289
loss 5.2547827
STEP 120 ================================
prereg loss 1.8379408 reg_l1 33.81364 reg_l2 23.476982
loss 5.219305
STEP 121 ================================
prereg loss 1.8067379 reg_l1 33.804325 reg_l2 23.470278
loss 5.1871705
cutoff 0.33085304 network size 56
STEP 122 ================================
prereg loss 1.797418 reg_l1 33.463818 reg_l2 23.353619
loss 5.1438
STEP 123 ================================
prereg loss 1.7772914 reg_l1 33.454315 reg_l2 23.34627
loss 5.122723
STEP 124 ================================
prereg loss 1.7592255 reg_l1 33.445 reg_l2 23.3389
loss 5.1037254
STEP 125 ================================
prereg loss 1.7422692 reg_l1 33.436092 reg_l2 23.331781
loss 5.0858784
STEP 126 ================================
prereg loss 1.7259204 reg_l1 33.42776 reg_l2 23.325134
loss 5.0686965
STEP 127 ================================
prereg loss 1.7099926 reg_l1 33.42016 reg_l2 23.31917
loss 5.0520086
STEP 128 ================================
prereg loss 1.6944673 reg_l1 33.413387 reg_l2 23.314047
loss 5.035806
STEP 129 ================================
prereg loss 1.6793092 reg_l1 33.40751 reg_l2 23.309872
loss 5.02006
STEP 130 ================================
prereg loss 1.6643633 reg_l1 33.40255 reg_l2 23.30672
loss 5.004618
STEP 131 ================================
prereg loss 1.6493711 reg_l1 33.398502 reg_l2 23.304613
loss 4.9892216
STEP 132 ================================
prereg loss 1.6340339 reg_l1 33.395336 reg_l2 23.30353
loss 4.9735675
STEP 133 ================================
prereg loss 1.6181281 reg_l1 33.39297 reg_l2 23.303423
loss 4.957425
STEP 134 ================================
prereg loss 1.6015868 reg_l1 33.391335 reg_l2 23.304178
loss 4.9407206
STEP 135 ================================
prereg loss 1.5845118 reg_l1 33.39031 reg_l2 23.305674
loss 4.923543
STEP 136 ================================
prereg loss 1.5671864 reg_l1 33.389782 reg_l2 23.307764
loss 4.9061646
STEP 137 ================================
prereg loss 1.5499511 reg_l1 33.389633 reg_l2 23.310287
loss 4.888914
STEP 138 ================================
prereg loss 1.533159 reg_l1 33.389736 reg_l2 23.31308
loss 4.872133
STEP 139 ================================
prereg loss 1.5170664 reg_l1 33.38997 reg_l2 23.315973
loss 4.8560634
STEP 140 ================================
prereg loss 1.5017961 reg_l1 33.39021 reg_l2 23.318808
loss 4.840817
STEP 141 ================================
prereg loss 1.4873387 reg_l1 33.390354 reg_l2 23.32145
loss 4.826374
STEP 142 ================================
prereg loss 1.4735866 reg_l1 33.390312 reg_l2 23.323763
loss 4.812618
STEP 143 ================================
prereg loss 1.4603837 reg_l1 33.39002 reg_l2 23.325651
loss 4.7993855
STEP 144 ================================
prereg loss 1.4475832 reg_l1 33.389385 reg_l2 23.327045
loss 4.786522
STEP 145 ================================
prereg loss 1.4350815 reg_l1 33.388382 reg_l2 23.32788
loss 4.77392
STEP 146 ================================
prereg loss 1.4228187 reg_l1 33.386986 reg_l2 23.328152
loss 4.7615175
STEP 147 ================================
prereg loss 1.4107897 reg_l1 33.385174 reg_l2 23.327843
loss 4.749307
STEP 148 ================================
prereg loss 1.3989991 reg_l1 33.38296 reg_l2 23.326965
loss 4.737295
STEP 149 ================================
prereg loss 1.387459 reg_l1 33.380363 reg_l2 23.32556
loss 4.7254953
STEP 150 ================================
prereg loss 1.3761668 reg_l1 33.3774 reg_l2 23.32367
loss 4.713907
STEP 151 ================================
prereg loss 1.3651103 reg_l1 33.3741 reg_l2 23.321346
loss 4.7025204
STEP 152 ================================
prereg loss 1.3542606 reg_l1 33.370514 reg_l2 23.318651
loss 4.691312
STEP 153 ================================
prereg loss 1.3436173 reg_l1 33.366695 reg_l2 23.31564
loss 4.680287
STEP 154 ================================
prereg loss 1.3331903 reg_l1 33.362682 reg_l2 23.312357
loss 4.6694584
STEP 155 ================================
prereg loss 1.323008 reg_l1 33.35851 reg_l2 23.308876
loss 4.658859
STEP 156 ================================
prereg loss 1.3131315 reg_l1 33.35423 reg_l2 23.305223
loss 4.6485543
STEP 157 ================================
prereg loss 1.3036078 reg_l1 33.349888 reg_l2 23.301445
loss 4.6385965
STEP 158 ================================
prereg loss 1.2944605 reg_l1 33.34552 reg_l2 23.297577
loss 4.6290126
STEP 159 ================================
prereg loss 1.2856992 reg_l1 33.34114 reg_l2 23.293648
loss 4.6198134
STEP 160 ================================
prereg loss 1.2772822 reg_l1 33.33679 reg_l2 23.289656
loss 4.610961
STEP 161 ================================
prereg loss 1.269143 reg_l1 33.332474 reg_l2 23.285625
loss 4.6023903
cutoff 0.29100543 network size 55
STEP 162 ================================
prereg loss 1.1470919 reg_l1 33.037205 reg_l2 23.196886
loss 4.4508123
STEP 163 ================================
prereg loss 1.1350822 reg_l1 33.036568 reg_l2 23.19667
loss 4.438739
STEP 164 ================================
prereg loss 1.1204784 reg_l1 33.038307 reg_l2 23.199291
loss 4.4243093
STEP 165 ================================
prereg loss 1.1061748 reg_l1 33.041996 reg_l2 23.204206
loss 4.4103746
STEP 166 ================================
prereg loss 1.094274 reg_l1 33.04722 reg_l2 23.210804
loss 4.398996
STEP 167 ================================
prereg loss 1.0855829 reg_l1 33.053497 reg_l2 23.218498
loss 4.3909326
STEP 168 ================================
prereg loss 1.079641 reg_l1 33.060417 reg_l2 23.226702
loss 4.3856826
STEP 169 ================================
prereg loss 1.0751833 reg_l1 33.067535 reg_l2 23.23485
loss 4.381937
STEP 170 ================================
prereg loss 1.0707862 reg_l1 33.074444 reg_l2 23.242428
loss 4.3782306
STEP 171 ================================
prereg loss 1.0654615 reg_l1 33.08078 reg_l2 23.249012
loss 4.3735394
STEP 172 ================================
prereg loss 1.0589424 reg_l1 33.086227 reg_l2 23.25424
loss 4.367565
STEP 173 ================================
prereg loss 1.0516729 reg_l1 33.090557 reg_l2 23.257837
loss 4.3607287
STEP 174 ================================
prereg loss 1.04447 reg_l1 33.09354 reg_l2 23.259623
loss 4.353824
STEP 175 ================================
prereg loss 1.0381259 reg_l1 33.095066 reg_l2 23.25949
loss 4.3476324
STEP 176 ================================
prereg loss 1.033043 reg_l1 33.095074 reg_l2 23.257431
loss 4.3425503
STEP 177 ================================
prereg loss 1.0291111 reg_l1 33.09352 reg_l2 23.25346
loss 4.3384633
STEP 178 ================================
prereg loss 1.0258018 reg_l1 33.090477 reg_l2 23.247696
loss 4.3348494
STEP 179 ================================
prereg loss 1.022443 reg_l1 33.08601 reg_l2 23.240263
loss 4.331044
STEP 180 ================================
prereg loss 1.0185475 reg_l1 33.080257 reg_l2 23.231348
loss 4.3265734
STEP 181 ================================
prereg loss 1.01405 reg_l1 33.07337 reg_l2 23.22114
loss 4.3213873
STEP 182 ================================
prereg loss 1.0093641 reg_l1 33.065525 reg_l2 23.209839
loss 4.315917
STEP 183 ================================
prereg loss 1.0052603 reg_l1 33.056915 reg_l2 23.197659
loss 4.3109517
STEP 184 ================================
prereg loss 1.0026051 reg_l1 33.047733 reg_l2 23.1848
loss 4.3073783
STEP 185 ================================
prereg loss 1.0020627 reg_l1 33.038147 reg_l2 23.171465
loss 4.305877
STEP 186 ================================
prereg loss 1.0038782 reg_l1 33.028362 reg_l2 23.157812
loss 4.3067145
STEP 187 ================================
prereg loss 1.007787 reg_l1 33.018528 reg_l2 23.144032
loss 4.30964
STEP 188 ================================
prereg loss 1.0130818 reg_l1 33.008797 reg_l2 23.130238
loss 4.3139615
STEP 189 ================================
prereg loss 1.0188049 reg_l1 32.99928 reg_l2 23.116554
loss 4.3187327
STEP 190 ================================
prereg loss 1.0239811 reg_l1 32.990074 reg_l2 23.103083
loss 4.3229885
STEP 191 ================================
prereg loss 1.0278574 reg_l1 32.981228 reg_l2 23.089874
loss 4.32598
STEP 192 ================================
prereg loss 1.0300279 reg_l1 32.9728 reg_l2 23.076998
loss 4.3273077
STEP 193 ================================
prereg loss 1.030483 reg_l1 32.964794 reg_l2 23.064468
loss 4.3269625
STEP 194 ================================
prereg loss 1.029556 reg_l1 32.957188 reg_l2 23.05228
loss 4.325275
STEP 195 ================================
prereg loss 1.0277963 reg_l1 32.94996 reg_l2 23.040428
loss 4.322792
STEP 196 ================================
prereg loss 1.0258409 reg_l1 32.943054 reg_l2 23.028864
loss 4.3201466
STEP 197 ================================
prereg loss 1.0242893 reg_l1 32.936413 reg_l2 23.01754
loss 4.3179307
STEP 198 ================================
prereg loss 1.0236315 reg_l1 32.929947 reg_l2 23.00638
loss 4.316626
STEP 199 ================================
prereg loss 1.0242101 reg_l1 32.92358 reg_l2 22.995308
loss 4.3165684
STEP 200 ================================
prereg loss 1.0262127 reg_l1 32.917244 reg_l2 22.98423
loss 4.317937
2022-07-20T17:23:01.034

julia> serialize("cf-55-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-55-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T17:23:35.272
STEP 1 ================================
prereg loss 1.0297087 reg_l1 32.910828 reg_l2 22.97306
loss 4.3207917
cutoff 0.28175586 network size 54
STEP 2 ================================
prereg loss 1.1882534 reg_l1 32.622517 reg_l2 22.882313
loss 4.4505053
STEP 3 ================================
prereg loss 1.1755695 reg_l1 32.620747 reg_l2 22.876722
loss 4.437644
STEP 4 ================================
prereg loss 1.1615709 reg_l1 32.622475 reg_l2 22.8757
loss 4.4238186
STEP 5 ================================
prereg loss 1.1542269 reg_l1 32.62705 reg_l2 22.878305
loss 4.4169316
STEP 6 ================================
prereg loss 1.1591773 reg_l1 32.633804 reg_l2 22.883581
loss 4.422558
STEP 7 ================================
prereg loss 1.1778705 reg_l1 32.642014 reg_l2 22.890543
loss 4.442072
STEP 8 ================================
prereg loss 1.2074395 reg_l1 32.651024 reg_l2 22.898233
loss 4.472542
STEP 9 ================================
prereg loss 1.242066 reg_l1 32.66014 reg_l2 22.905714
loss 4.50808
STEP 10 ================================
prereg loss 1.2751524 reg_l1 32.668728 reg_l2 22.912153
loss 4.5420256
STEP 11 ================================
prereg loss 1.30145 reg_l1 32.67621 reg_l2 22.916794
loss 4.569071
STEP 12 ================================
prereg loss 1.3184052 reg_l1 32.682064 reg_l2 22.919025
loss 4.5866117
STEP 13 ================================
prereg loss 1.3264065 reg_l1 32.685883 reg_l2 22.918375
loss 4.5949945
STEP 14 ================================
prereg loss 1.3279634 reg_l1 32.687344 reg_l2 22.914524
loss 4.596698
STEP 15 ================================
prereg loss 1.326294 reg_l1 32.68625 reg_l2 22.90731
loss 4.594919
STEP 16 ================================
prereg loss 1.3221563 reg_l1 32.682484 reg_l2 22.896706
loss 4.5904045
STEP 17 ================================
prereg loss 1.3186699 reg_l1 32.67659 reg_l2 22.883574
loss 4.586329
STEP 18 ================================
prereg loss 1.3167942 reg_l1 32.6686 reg_l2 22.86803
loss 4.583654
STEP 19 ================================
prereg loss 1.3151648 reg_l1 32.65859 reg_l2 22.850248
loss 4.5810237
STEP 20 ================================
prereg loss 1.3124636 reg_l1 32.64674 reg_l2 22.830462
loss 4.5771375
STEP 21 ================================
prereg loss 1.3082309 reg_l1 32.63325 reg_l2 22.80893
loss 4.571556
STEP 22 ================================
prereg loss 1.3032042 reg_l1 32.618385 reg_l2 22.78596
loss 4.565043
STEP 23 ================================
prereg loss 1.2990547 reg_l1 32.602406 reg_l2 22.76186
loss 4.559295
STEP 24 ================================
prereg loss 1.2977355 reg_l1 32.58562 reg_l2 22.736954
loss 4.5562973
STEP 25 ================================
prereg loss 1.3007306 reg_l1 32.56832 reg_l2 22.711542
loss 4.557563
STEP 26 ================================
prereg loss 1.3084613 reg_l1 32.550785 reg_l2 22.685925
loss 4.56354
STEP 27 ================================
prereg loss 1.3201251 reg_l1 32.533276 reg_l2 22.660364
loss 4.573453
STEP 28 ================================
prereg loss 1.3338856 reg_l1 32.516033 reg_l2 22.635082
loss 4.585489
STEP 29 ================================
prereg loss 1.3470875 reg_l1 32.49923 reg_l2 22.610285
loss 4.5970106
STEP 30 ================================
prereg loss 1.3576252 reg_l1 32.484238 reg_l2 22.587872
loss 4.606049
STEP 31 ================================
prereg loss 1.3659872 reg_l1 32.47102 reg_l2 22.567734
loss 4.613089
STEP 32 ================================
prereg loss 1.3706396 reg_l1 32.459496 reg_l2 22.549742
loss 4.616589
STEP 33 ================================
prereg loss 1.3708115 reg_l1 32.449547 reg_l2 22.533731
loss 4.6157665
STEP 34 ================================
prereg loss 1.3665332 reg_l1 32.441032 reg_l2 22.51952
loss 4.610636
STEP 35 ================================
prereg loss 1.3585457 reg_l1 32.433792 reg_l2 22.506926
loss 4.601925
STEP 36 ================================
prereg loss 1.3480568 reg_l1 32.427635 reg_l2 22.495726
loss 4.5908203
STEP 37 ================================
prereg loss 1.3364592 reg_l1 32.422386 reg_l2 22.485714
loss 4.578698
STEP 38 ================================
prereg loss 1.3250259 reg_l1 32.417847 reg_l2 22.476664
loss 4.5668106
STEP 39 ================================
prereg loss 1.3147525 reg_l1 32.41382 reg_l2 22.468369
loss 4.556134
STEP 40 ================================
prereg loss 1.3062559 reg_l1 32.410126 reg_l2 22.460634
loss 4.5472684
STEP 41 ================================
prereg loss 1.2997869 reg_l1 32.4066 reg_l2 22.453262
loss 4.540447
cutoff 0.29905817 network size 53
STEP 42 ================================
prereg loss 1.2952955 reg_l1 32.10404 reg_l2 22.356628
loss 4.505699
STEP 43 ================================
prereg loss 1.292529 reg_l1 32.100517 reg_l2 22.349531
loss 4.5025806
STEP 44 ================================
prereg loss 1.2911216 reg_l1 32.096786 reg_l2 22.34232
loss 4.5008
STEP 45 ================================
prereg loss 1.2906574 reg_l1 32.09276 reg_l2 22.3349
loss 4.4999332
STEP 46 ================================
prereg loss 1.2907025 reg_l1 32.08836 reg_l2 22.327175
loss 4.4995384
STEP 47 ================================
prereg loss 1.2908359 reg_l1 32.083565 reg_l2 22.319077
loss 4.499192
STEP 48 ================================
prereg loss 1.2906297 reg_l1 32.078342 reg_l2 22.310572
loss 4.498464
STEP 49 ================================
prereg loss 1.2896622 reg_l1 32.072678 reg_l2 22.30161
loss 4.49693
STEP 50 ================================
prereg loss 1.287591 reg_l1 32.066586 reg_l2 22.292204
loss 4.4942493
STEP 51 ================================
prereg loss 1.2841122 reg_l1 32.06006 reg_l2 22.282347
loss 4.490118
STEP 52 ================================
prereg loss 1.2790291 reg_l1 32.05314 reg_l2 22.272053
loss 4.484343
STEP 53 ================================
prereg loss 1.2722744 reg_l1 32.04584 reg_l2 22.261345
loss 4.4768586
STEP 54 ================================
prereg loss 1.2638983 reg_l1 32.038197 reg_l2 22.25026
loss 4.467718
STEP 55 ================================
prereg loss 1.2540784 reg_l1 32.03023 reg_l2 22.238834
loss 4.457102
STEP 56 ================================
prereg loss 1.2430704 reg_l1 32.02198 reg_l2 22.227102
loss 4.4452686
STEP 57 ================================
prereg loss 1.2311772 reg_l1 32.01347 reg_l2 22.21511
loss 4.432524
STEP 58 ================================
prereg loss 1.2177699 reg_l1 32.00475 reg_l2 22.202908
loss 4.418245
STEP 59 ================================
prereg loss 1.2039883 reg_l1 31.995974 reg_l2 22.190731
loss 4.4035854
STEP 60 ================================
prereg loss 1.1902934 reg_l1 31.98718 reg_l2 22.178621
loss 4.3890114
STEP 61 ================================
prereg loss 1.1768937 reg_l1 31.978392 reg_l2 22.166603
loss 4.374733
STEP 62 ================================
prereg loss 1.1639278 reg_l1 31.969648 reg_l2 22.154722
loss 4.360893
STEP 63 ================================
prereg loss 1.1514616 reg_l1 31.960955 reg_l2 22.142994
loss 4.347557
STEP 64 ================================
prereg loss 1.1395061 reg_l1 31.95234 reg_l2 22.131441
loss 4.33474
STEP 65 ================================
prereg loss 1.1280296 reg_l1 31.943829 reg_l2 22.120094
loss 4.3224125
STEP 66 ================================
prereg loss 1.116971 reg_l1 31.93544 reg_l2 22.108967
loss 4.3105154
STEP 67 ================================
prereg loss 1.1062311 reg_l1 31.927181 reg_l2 22.098064
loss 4.2989492
STEP 68 ================================
prereg loss 1.0956945 reg_l1 31.919067 reg_l2 22.087385
loss 4.2876015
STEP 69 ================================
prereg loss 1.0852523 reg_l1 31.9111 reg_l2 22.076939
loss 4.2763624
STEP 70 ================================
prereg loss 1.0747854 reg_l1 31.903276 reg_l2 22.066715
loss 4.265113
STEP 71 ================================
prereg loss 1.0642028 reg_l1 31.895597 reg_l2 22.056705
loss 4.2537622
STEP 72 ================================
prereg loss 1.0534216 reg_l1 31.88805 reg_l2 22.04688
loss 4.2422266
STEP 73 ================================
prereg loss 1.0424095 reg_l1 31.880615 reg_l2 22.037233
loss 4.230471
STEP 74 ================================
prereg loss 1.0311612 reg_l1 31.873276 reg_l2 22.027739
loss 4.2184887
STEP 75 ================================
prereg loss 1.0196902 reg_l1 31.866016 reg_l2 22.018366
loss 4.2062917
STEP 76 ================================
prereg loss 1.0080527 reg_l1 31.858795 reg_l2 22.00909
loss 4.193932
STEP 77 ================================
prereg loss 0.9962971 reg_l1 31.851612 reg_l2 21.99988
loss 4.1814585
STEP 78 ================================
prereg loss 0.9844912 reg_l1 31.844429 reg_l2 21.99071
loss 4.1689343
STEP 79 ================================
prereg loss 0.9726913 reg_l1 31.837221 reg_l2 21.981552
loss 4.1564136
STEP 80 ================================
prereg loss 0.9609557 reg_l1 31.829971 reg_l2 21.972395
loss 4.143953
STEP 81 ================================
prereg loss 0.94929916 reg_l1 31.82267 reg_l2 21.963203
loss 4.131566
cutoff 0.31717697 network size 52
STEP 82 ================================
prereg loss 0.9733196 reg_l1 31.498116 reg_l2 21.853373
loss 4.1231313
STEP 83 ================================
prereg loss 0.9584305 reg_l1 31.493523 reg_l2 21.847155
loss 4.107783
STEP 84 ================================
prereg loss 0.9409483 reg_l1 31.490526 reg_l2 21.84303
loss 4.090001
STEP 85 ================================
prereg loss 0.9220999 reg_l1 31.488861 reg_l2 21.840658
loss 4.0709863
STEP 86 ================================
prereg loss 0.903092 reg_l1 31.488283 reg_l2 21.8397
loss 4.0519204
STEP 87 ================================
prereg loss 0.88491964 reg_l1 31.48853 reg_l2 21.839815
loss 4.0337725
STEP 88 ================================
prereg loss 0.8682323 reg_l1 31.489347 reg_l2 21.84066
loss 4.017167
STEP 89 ================================
prereg loss 0.85331655 reg_l1 31.490482 reg_l2 21.841892
loss 4.002365
STEP 90 ================================
prereg loss 0.840166 reg_l1 31.49168 reg_l2 21.843197
loss 3.989334
STEP 91 ================================
prereg loss 0.8285946 reg_l1 31.492712 reg_l2 21.844273
loss 3.977866
STEP 92 ================================
prereg loss 0.8183149 reg_l1 31.493355 reg_l2 21.84484
loss 3.9676504
STEP 93 ================================
prereg loss 0.8090194 reg_l1 31.493404 reg_l2 21.844645
loss 3.9583597
STEP 94 ================================
prereg loss 0.8003902 reg_l1 31.4927 reg_l2 21.843504
loss 3.9496603
STEP 95 ================================
prereg loss 0.7921042 reg_l1 31.491077 reg_l2 21.841225
loss 3.941212
STEP 96 ================================
prereg loss 0.7838189 reg_l1 31.488445 reg_l2 21.837692
loss 3.9326634
STEP 97 ================================
prereg loss 0.7751659 reg_l1 31.484715 reg_l2 21.832838
loss 3.9236374
STEP 98 ================================
prereg loss 0.7657853 reg_l1 31.479855 reg_l2 21.826633
loss 3.9137707
STEP 99 ================================
prereg loss 0.75368124 reg_l1 31.473871 reg_l2 21.819084
loss 3.9010684
STEP 100 ================================
prereg loss 0.71315914 reg_l1 31.468319 reg_l2 21.81175
loss 3.859991
STEP 101 ================================
prereg loss 0.66274285 reg_l1 31.462667 reg_l2 21.80416
loss 3.8090096
STEP 102 ================================
prereg loss 0.61008644 reg_l1 31.456753 reg_l2 21.796204
loss 3.7557619
STEP 103 ================================
prereg loss 0.5607428 reg_l1 31.450558 reg_l2 21.787941
loss 3.7057986
STEP 104 ================================
prereg loss 0.5186501 reg_l1 31.444136 reg_l2 21.779488
loss 3.6630638
STEP 105 ================================
prereg loss 0.48627505 reg_l1 31.43755 reg_l2 21.770964
loss 3.63003
STEP 106 ================================
prereg loss 0.4646942 reg_l1 31.430885 reg_l2 21.7625
loss 3.6077828
STEP 107 ================================
prereg loss 0.4536729 reg_l1 31.42421 reg_l2 21.754192
loss 3.596094
STEP 108 ================================
prereg loss 0.45185244 reg_l1 31.417568 reg_l2 21.746117
loss 3.5936093
STEP 109 ================================
prereg loss 0.45700032 reg_l1 31.41099 reg_l2 21.73833
loss 3.5980992
STEP 110 ================================
prereg loss 0.46640638 reg_l1 31.404512 reg_l2 21.73086
loss 3.6068575
STEP 111 ================================
prereg loss 0.47732088 reg_l1 31.398172 reg_l2 21.723732
loss 3.6171381
STEP 112 ================================
prereg loss 0.48736277 reg_l1 31.391998 reg_l2 21.716984
loss 3.6265628
STEP 113 ================================
prereg loss 0.4948036 reg_l1 31.386036 reg_l2 21.71065
loss 3.6334074
STEP 114 ================================
prereg loss 0.498665 reg_l1 31.380323 reg_l2 21.704775
loss 3.6366975
STEP 115 ================================
prereg loss 0.4986832 reg_l1 31.374897 reg_l2 21.699383
loss 3.636173
STEP 116 ================================
prereg loss 0.49514732 reg_l1 31.369797 reg_l2 21.694487
loss 3.632127
STEP 117 ================================
prereg loss 0.48873243 reg_l1 31.365013 reg_l2 21.690098
loss 3.6252337
STEP 118 ================================
prereg loss 0.48032144 reg_l1 31.360565 reg_l2 21.686201
loss 3.616378
STEP 119 ================================
prereg loss 0.47084907 reg_l1 31.356432 reg_l2 21.68277
loss 3.6064923
STEP 120 ================================
prereg loss 0.4611918 reg_l1 31.352581 reg_l2 21.679747
loss 3.59645
STEP 121 ================================
prereg loss 0.45208856 reg_l1 31.348991 reg_l2 21.677092
loss 3.5869877
cutoff 0.31502646 network size 51
STEP 122 ================================
prereg loss 16.350023 reg_l1 31.030598 reg_l2 21.575504
loss 19.453083
STEP 123 ================================
prereg loss 16.027798 reg_l1 31.037968 reg_l2 21.586388
loss 19.131594
STEP 124 ================================
prereg loss 15.474342 reg_l1 31.054363 reg_l2 21.60838
loss 18.579779
STEP 125 ================================
prereg loss 14.748634 reg_l1 31.078352 reg_l2 21.639723
loss 17.85647
STEP 126 ================================
prereg loss 13.901005 reg_l1 31.108774 reg_l2 21.679028
loss 17.011883
STEP 127 ================================
prereg loss 12.972246 reg_l1 31.1447 reg_l2 21.725103
loss 16.086716
STEP 128 ================================
prereg loss 11.994106 reg_l1 31.18531 reg_l2 21.776937
loss 15.1126375
STEP 129 ================================
prereg loss 10.990871 reg_l1 31.229881 reg_l2 21.833593
loss 14.113859
STEP 130 ================================
prereg loss 9.981732 reg_l1 31.277718 reg_l2 21.89421
loss 13.109505
STEP 131 ================================
prereg loss 8.983172 reg_l1 31.328178 reg_l2 21.957933
loss 12.115991
STEP 132 ================================
prereg loss 8.010866 reg_l1 31.380573 reg_l2 22.023943
loss 11.148924
STEP 133 ================================
prereg loss 7.0807557 reg_l1 31.434238 reg_l2 22.091385
loss 10.224179
STEP 134 ================================
prereg loss 6.20909 reg_l1 31.488455 reg_l2 22.159412
loss 9.357936
STEP 135 ================================
prereg loss 5.4116335 reg_l1 31.542515 reg_l2 22.227139
loss 8.565886
STEP 136 ================================
prereg loss 4.702355 reg_l1 31.59567 reg_l2 22.293682
loss 7.8619223
STEP 137 ================================
prereg loss 4.091954 reg_l1 31.647188 reg_l2 22.358139
loss 7.256673
STEP 138 ================================
prereg loss 3.5842683 reg_l1 31.696327 reg_l2 22.419617
loss 6.753901
STEP 139 ================================
prereg loss 3.1743717 reg_l1 31.743221 reg_l2 22.478485
loss 6.348694
STEP 140 ================================
prereg loss 2.8652503 reg_l1 31.787224 reg_l2 22.533955
loss 6.043973
STEP 141 ================================
prereg loss 2.6476753 reg_l1 31.82775 reg_l2 22.585264
loss 5.83045
STEP 142 ================================
prereg loss 2.509459 reg_l1 31.86431 reg_l2 22.63177
loss 5.69589
STEP 143 ================================
prereg loss 2.4366937 reg_l1 31.896498 reg_l2 22.672901
loss 5.6263437
STEP 144 ================================
prereg loss 2.4148142 reg_l1 31.92399 reg_l2 22.708214
loss 5.607213
STEP 145 ================================
prereg loss 2.4294696 reg_l1 31.946598 reg_l2 22.737383
loss 5.6241293
STEP 146 ================================
prereg loss 2.467225 reg_l1 31.96419 reg_l2 22.760197
loss 5.663644
STEP 147 ================================
prereg loss 2.5160992 reg_l1 31.97678 reg_l2 22.776594
loss 5.7137775
STEP 148 ================================
prereg loss 2.5660243 reg_l1 31.984423 reg_l2 22.786621
loss 5.7644663
STEP 149 ================================
prereg loss 2.609133 reg_l1 31.987299 reg_l2 22.790466
loss 5.807863
STEP 150 ================================
prereg loss 2.6399271 reg_l1 31.985636 reg_l2 22.7884
loss 5.8384905
STEP 151 ================================
prereg loss 2.6552682 reg_l1 31.979744 reg_l2 22.780806
loss 5.853243
STEP 152 ================================
prereg loss 2.6541567 reg_l1 31.969973 reg_l2 22.768133
loss 5.8511543
STEP 153 ================================
prereg loss 2.6373675 reg_l1 31.956707 reg_l2 22.750902
loss 5.8330383
STEP 154 ================================
prereg loss 2.6070013 reg_l1 31.940357 reg_l2 22.72967
loss 5.801037
STEP 155 ================================
prereg loss 2.565954 reg_l1 31.921362 reg_l2 22.705019
loss 5.75809
STEP 156 ================================
prereg loss 2.5174413 reg_l1 31.900152 reg_l2 22.677559
loss 5.7074566
STEP 157 ================================
prereg loss 2.4561818 reg_l1 31.877161 reg_l2 22.647884
loss 5.643898
STEP 158 ================================
prereg loss 2.3854213 reg_l1 31.8516 reg_l2 22.614737
loss 5.5705814
STEP 159 ================================
prereg loss 2.3179204 reg_l1 31.82405 reg_l2 22.578987
loss 5.5003257
STEP 160 ================================
prereg loss 2.257351 reg_l1 31.795057 reg_l2 22.54141
loss 5.4368567
STEP 161 ================================
prereg loss 2.2055078 reg_l1 31.76511 reg_l2 22.502739
loss 5.382019
cutoff 0.32327148 network size 50
STEP 162 ================================
prereg loss 2.1967845 reg_l1 31.411379 reg_l2 22.359116
loss 5.337922
STEP 163 ================================
prereg loss 2.1578693 reg_l1 31.380632 reg_l2 22.319895
loss 5.295933
STEP 164 ================================
prereg loss 2.1252215 reg_l1 31.35 reg_l2 22.281118
loss 5.2602215
STEP 165 ================================
prereg loss 2.0969598 reg_l1 31.31983 reg_l2 22.243258
loss 5.228943
STEP 166 ================================
prereg loss 2.071423 reg_l1 31.290413 reg_l2 22.206703
loss 5.2004642
STEP 167 ================================
prereg loss 2.0474594 reg_l1 31.262012 reg_l2 22.17178
loss 5.1736608
STEP 168 ================================
prereg loss 2.0244417 reg_l1 31.23485 reg_l2 22.138754
loss 5.147927
STEP 169 ================================
prereg loss 2.0021846 reg_l1 31.209105 reg_l2 22.107822
loss 5.123095
STEP 170 ================================
prereg loss 1.9807373 reg_l1 31.184935 reg_l2 22.07914
loss 5.099231
STEP 171 ================================
prereg loss 1.9602059 reg_l1 31.16243 reg_l2 22.052797
loss 5.076449
STEP 172 ================================
prereg loss 1.9405751 reg_l1 31.141682 reg_l2 22.028849
loss 5.0547433
STEP 173 ================================
prereg loss 1.9216666 reg_l1 31.12271 reg_l2 22.007288
loss 5.0339375
STEP 174 ================================
prereg loss 1.9031775 reg_l1 31.105524 reg_l2 21.988071
loss 5.01373
STEP 175 ================================
prereg loss 1.8847156 reg_l1 31.090097 reg_l2 21.971142
loss 4.9937253
STEP 176 ================================
prereg loss 1.8659594 reg_l1 31.07637 reg_l2 21.956388
loss 4.9735966
STEP 177 ================================
prereg loss 1.8467187 reg_l1 31.06425 reg_l2 21.943672
loss 4.9531436
STEP 178 ================================
prereg loss 1.8270019 reg_l1 31.05363 reg_l2 21.932861
loss 4.932365
STEP 179 ================================
prereg loss 1.807019 reg_l1 31.044397 reg_l2 21.92378
loss 4.911459
STEP 180 ================================
prereg loss 1.7871318 reg_l1 31.036411 reg_l2 21.916252
loss 4.890773
STEP 181 ================================
prereg loss 1.7677642 reg_l1 31.029522 reg_l2 21.9101
loss 4.8707166
STEP 182 ================================
prereg loss 1.7493354 reg_l1 31.02358 reg_l2 21.905138
loss 4.8516936
STEP 183 ================================
prereg loss 1.7321637 reg_l1 31.01844 reg_l2 21.901186
loss 4.8340077
STEP 184 ================================
prereg loss 1.7164346 reg_l1 31.013948 reg_l2 21.89807
loss 4.8178296
STEP 185 ================================
prereg loss 1.7021868 reg_l1 31.009968 reg_l2 21.895622
loss 4.8031836
STEP 186 ================================
prereg loss 1.6893378 reg_l1 31.006388 reg_l2 21.893688
loss 4.7899766
STEP 187 ================================
prereg loss 1.6777167 reg_l1 31.003063 reg_l2 21.892124
loss 4.778023
STEP 188 ================================
prereg loss 1.6671385 reg_l1 30.999926 reg_l2 21.890808
loss 4.767131
STEP 189 ================================
prereg loss 1.6574215 reg_l1 30.996843 reg_l2 21.88963
loss 4.757106
STEP 190 ================================
prereg loss 1.6484427 reg_l1 30.993782 reg_l2 21.888493
loss 4.747821
STEP 191 ================================
prereg loss 1.6401461 reg_l1 30.990658 reg_l2 21.887312
loss 4.739212
STEP 192 ================================
prereg loss 1.6325309 reg_l1 30.987427 reg_l2 21.88602
loss 4.7312737
STEP 193 ================================
prereg loss 1.6256443 reg_l1 30.984066 reg_l2 21.884577
loss 4.724051
STEP 194 ================================
prereg loss 1.6195576 reg_l1 30.980541 reg_l2 21.882946
loss 4.717612
STEP 195 ================================
prereg loss 1.6143476 reg_l1 30.976843 reg_l2 21.881088
loss 4.712032
STEP 196 ================================
prereg loss 1.6100719 reg_l1 30.972954 reg_l2 21.878998
loss 4.7073674
STEP 197 ================================
prereg loss 1.6067666 reg_l1 30.968868 reg_l2 21.876673
loss 4.7036533
STEP 198 ================================
prereg loss 1.6044371 reg_l1 30.9646 reg_l2 21.8741
loss 4.700897
STEP 199 ================================
prereg loss 1.6030633 reg_l1 30.960161 reg_l2 21.871298
loss 4.6990795
STEP 200 ================================
prereg loss 1.602601 reg_l1 30.95555 reg_l2 21.868286
loss 4.6981564
2022-07-20T17:34:30.551

julia> serialize("cf-50-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-50-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T17:46:57.618
STEP 1 ================================
prereg loss 1.6029834 reg_l1 30.950773 reg_l2 21.865065
loss 4.6980605
cutoff 0.32157332 network size 49
STEP 2 ================================
prereg loss 1.6041331 reg_l1 30.624283 reg_l2 21.75826
loss 4.666561
STEP 3 ================================
prereg loss 1.6059537 reg_l1 30.619293 reg_l2 21.754753
loss 4.667883
STEP 4 ================================
prereg loss 1.6083353 reg_l1 30.6142 reg_l2 21.751112
loss 4.6697555
STEP 5 ================================
prereg loss 1.6111599 reg_l1 30.608994 reg_l2 21.747374
loss 4.6720595
STEP 6 ================================
prereg loss 1.6142994 reg_l1 30.603718 reg_l2 21.743565
loss 4.674671
STEP 7 ================================
prereg loss 1.6176226 reg_l1 30.598377 reg_l2 21.739721
loss 4.6774607
STEP 8 ================================
prereg loss 1.6209964 reg_l1 30.592997 reg_l2 21.73586
loss 4.680296
STEP 9 ================================
prereg loss 1.624304 reg_l1 30.58759 reg_l2 21.732014
loss 4.683063
STEP 10 ================================
prereg loss 1.6274313 reg_l1 30.582184 reg_l2 21.728235
loss 4.68565
STEP 11 ================================
prereg loss 1.6302999 reg_l1 30.576782 reg_l2 21.724524
loss 4.6879783
STEP 12 ================================
prereg loss 1.6328473 reg_l1 30.571428 reg_l2 21.720917
loss 4.68999
STEP 13 ================================
prereg loss 1.6350366 reg_l1 30.566132 reg_l2 21.717442
loss 4.69165
STEP 14 ================================
prereg loss 1.636861 reg_l1 30.560904 reg_l2 21.714123
loss 4.692951
STEP 15 ================================
prereg loss 1.638341 reg_l1 30.55578 reg_l2 21.710972
loss 4.693919
STEP 16 ================================
prereg loss 1.6395016 reg_l1 30.550772 reg_l2 21.708017
loss 4.694579
STEP 17 ================================
prereg loss 1.6403888 reg_l1 30.54589 reg_l2 21.705278
loss 4.6949778
STEP 18 ================================
prereg loss 1.6410606 reg_l1 30.541153 reg_l2 21.702755
loss 4.695176
STEP 19 ================================
prereg loss 1.6415642 reg_l1 30.536568 reg_l2 21.700466
loss 4.695221
STEP 20 ================================
prereg loss 1.6419587 reg_l1 30.532148 reg_l2 21.69841
loss 4.6951733
STEP 21 ================================
prereg loss 1.6422932 reg_l1 30.527906 reg_l2 21.696608
loss 4.6950836
STEP 22 ================================
prereg loss 1.6426109 reg_l1 30.52384 reg_l2 21.695045
loss 4.694995
STEP 23 ================================
prereg loss 1.6429498 reg_l1 30.519953 reg_l2 21.69372
loss 4.6949453
STEP 24 ================================
prereg loss 1.6436424 reg_l1 30.516235 reg_l2 21.692636
loss 4.695266
STEP 25 ================================
prereg loss 1.6467011 reg_l1 30.51284 reg_l2 21.691917
loss 4.697985
STEP 26 ================================
prereg loss 1.6501434 reg_l1 30.50976 reg_l2 21.691544
loss 4.7011194
STEP 27 ================================
prereg loss 1.6539522 reg_l1 30.506983 reg_l2 21.691519
loss 4.7046504
STEP 28 ================================
prereg loss 1.6581137 reg_l1 30.504496 reg_l2 21.69182
loss 4.7085633
STEP 29 ================================
prereg loss 1.6625998 reg_l1 30.502277 reg_l2 21.692436
loss 4.7128277
STEP 30 ================================
prereg loss 1.6673806 reg_l1 30.500322 reg_l2 21.693352
loss 4.717413
STEP 31 ================================
prereg loss 1.672408 reg_l1 30.498604 reg_l2 21.694555
loss 4.7222686
STEP 32 ================================
prereg loss 1.6776342 reg_l1 30.497095 reg_l2 21.696028
loss 4.7273436
STEP 33 ================================
prereg loss 1.6830072 reg_l1 30.49579 reg_l2 21.697756
loss 4.7325864
STEP 34 ================================
prereg loss 1.6884546 reg_l1 30.494665 reg_l2 21.69973
loss 4.737921
STEP 35 ================================
prereg loss 1.6939245 reg_l1 30.493715 reg_l2 21.701921
loss 4.743296
STEP 36 ================================
prereg loss 1.6993624 reg_l1 30.492905 reg_l2 21.704336
loss 4.748653
STEP 37 ================================
prereg loss 1.7047306 reg_l1 30.492247 reg_l2 21.70696
loss 4.7539554
STEP 38 ================================
prereg loss 1.7100054 reg_l1 30.491714 reg_l2 21.709766
loss 4.7591767
STEP 39 ================================
prereg loss 1.7162204 reg_l1 30.491306 reg_l2 21.712774
loss 4.7653513
STEP 40 ================================
prereg loss 1.7215348 reg_l1 30.49078 reg_l2 21.715765
loss 4.7706127
STEP 41 ================================
prereg loss 1.7250217 reg_l1 30.49016 reg_l2 21.718746
loss 4.774038
cutoff 0.32417518 network size 48
STEP 42 ================================
prereg loss 1.7268338 reg_l1 30.165287 reg_l2 21.616652
loss 4.7433624
STEP 43 ================================
prereg loss 1.7289622 reg_l1 30.164652 reg_l2 21.619732
loss 4.745427
STEP 44 ================================
prereg loss 1.7322667 reg_l1 30.16422 reg_l2 21.623055
loss 4.7486887
STEP 45 ================================
prereg loss 1.736202 reg_l1 30.16399 reg_l2 21.626604
loss 4.752601
STEP 46 ================================
prereg loss 1.7407681 reg_l1 30.163969 reg_l2 21.63036
loss 4.757165
STEP 47 ================================
prereg loss 1.7459495 reg_l1 30.164124 reg_l2 21.634321
loss 4.762362
STEP 48 ================================
prereg loss 1.7517177 reg_l1 30.164448 reg_l2 21.63847
loss 4.7681627
STEP 49 ================================
prereg loss 1.758023 reg_l1 30.164928 reg_l2 21.642797
loss 4.774516
STEP 50 ================================
prereg loss 1.7648126 reg_l1 30.165552 reg_l2 21.647305
loss 4.781368
STEP 51 ================================
prereg loss 1.772011 reg_l1 30.16631 reg_l2 21.651966
loss 4.788642
STEP 52 ================================
prereg loss 1.7803552 reg_l1 30.167173 reg_l2 21.656775
loss 4.7970724
STEP 53 ================================
prereg loss 1.7870617 reg_l1 30.167831 reg_l2 21.661484
loss 4.803845
STEP 54 ================================
prereg loss 1.7915654 reg_l1 30.168314 reg_l2 21.66611
loss 4.808397
STEP 55 ================================
prereg loss 1.7974067 reg_l1 30.168938 reg_l2 21.670916
loss 4.8143005
STEP 56 ================================
prereg loss 1.8035657 reg_l1 30.169704 reg_l2 21.6759
loss 4.820536
STEP 57 ================================
prereg loss 1.8100194 reg_l1 30.170595 reg_l2 21.681047
loss 4.827079
STEP 58 ================================
prereg loss 1.8167409 reg_l1 30.171614 reg_l2 21.686363
loss 4.8339024
STEP 59 ================================
prereg loss 1.8237098 reg_l1 30.172745 reg_l2 21.691841
loss 4.8409843
STEP 60 ================================
prereg loss 1.8326323 reg_l1 30.173988 reg_l2 21.697474
loss 4.850031
STEP 61 ================================
prereg loss 1.8376701 reg_l1 30.175005 reg_l2 21.702984
loss 4.8551707
STEP 62 ================================
prereg loss 1.8413181 reg_l1 30.17582 reg_l2 21.708406
loss 4.8589
STEP 63 ================================
prereg loss 1.8465142 reg_l1 30.176815 reg_l2 21.714033
loss 4.864196
STEP 64 ================================
prereg loss 1.8523171 reg_l1 30.177973 reg_l2 21.71986
loss 4.8701143
STEP 65 ================================
prereg loss 1.8587295 reg_l1 30.179308 reg_l2 21.725868
loss 4.8766603
STEP 66 ================================
prereg loss 1.864513 reg_l1 30.180792 reg_l2 21.732065
loss 4.882592
STEP 67 ================================
prereg loss 1.8705202 reg_l1 30.181787 reg_l2 21.737585
loss 4.888699
STEP 68 ================================
prereg loss 1.8768475 reg_l1 30.18236 reg_l2 21.742508
loss 4.8950834
STEP 69 ================================
prereg loss 1.8834416 reg_l1 30.182564 reg_l2 21.746931
loss 4.901698
STEP 70 ================================
prereg loss 1.8902122 reg_l1 30.18245 reg_l2 21.750929
loss 4.9084573
STEP 71 ================================
prereg loss 1.8998634 reg_l1 30.182072 reg_l2 21.754595
loss 4.918071
STEP 72 ================================
prereg loss 1.9041258 reg_l1 30.181105 reg_l2 21.757685
loss 4.9222364
STEP 73 ================================
prereg loss 1.9049076 reg_l1 30.17963 reg_l2 21.760315
loss 4.9228706
STEP 74 ================================
prereg loss 1.9081833 reg_l1 30.178122 reg_l2 21.762867
loss 4.925996
STEP 75 ================================
prereg loss 1.9117581 reg_l1 30.176601 reg_l2 21.765398
loss 4.929418
STEP 76 ================================
prereg loss 1.90821 reg_l1 30.175133 reg_l2 21.76795
loss 4.925723
STEP 77 ================================
prereg loss 1.9023569 reg_l1 30.172466 reg_l2 21.768911
loss 4.9196033
STEP 78 ================================
prereg loss 1.8977485 reg_l1 30.168882 reg_l2 21.768667
loss 4.9146366
STEP 79 ================================
prereg loss 1.8947934 reg_l1 30.164654 reg_l2 21.767588
loss 4.9112587
STEP 80 ================================
prereg loss 1.893613 reg_l1 30.160019 reg_l2 21.766012
loss 4.909615
STEP 81 ================================
prereg loss 1.8953944 reg_l1 30.155212 reg_l2 21.764269
loss 4.910916
cutoff 0.32341442 network size 47
STEP 82 ================================
prereg loss 1.8935053 reg_l1 29.826603 reg_l2 21.657732
loss 4.8761654
STEP 83 ================================
prereg loss 1.8899059 reg_l1 29.821354 reg_l2 21.655935
loss 4.8720417
STEP 84 ================================
prereg loss 1.8871119 reg_l1 29.816534 reg_l2 21.654758
loss 4.8687654
STEP 85 ================================
prereg loss 1.8843212 reg_l1 29.812256 reg_l2 21.654333
loss 4.865547
STEP 86 ================================
prereg loss 1.881305 reg_l1 29.808605 reg_l2 21.65478
loss 4.8621655
STEP 87 ================================
prereg loss 1.8778956 reg_l1 29.805666 reg_l2 21.656193
loss 4.8584623
STEP 88 ================================
prereg loss 1.8740065 reg_l1 29.80347 reg_l2 21.658623
loss 4.854354
STEP 89 ================================
prereg loss 1.8696263 reg_l1 29.802061 reg_l2 21.662088
loss 4.8498325
STEP 90 ================================
prereg loss 1.8648034 reg_l1 29.801437 reg_l2 21.66661
loss 4.8449473
STEP 91 ================================
prereg loss 1.8596442 reg_l1 29.801603 reg_l2 21.672169
loss 4.8398046
STEP 92 ================================
prereg loss 1.8542851 reg_l1 29.802528 reg_l2 21.678713
loss 4.834538
STEP 93 ================================
prereg loss 1.8488982 reg_l1 29.804182 reg_l2 21.686203
loss 4.829316
STEP 94 ================================
prereg loss 1.8436552 reg_l1 29.80651 reg_l2 21.694576
loss 4.8243065
STEP 95 ================================
prereg loss 1.8398985 reg_l1 29.80948 reg_l2 21.70376
loss 4.8208466
STEP 96 ================================
prereg loss 1.8308345 reg_l1 29.81253 reg_l2 21.713318
loss 4.8120875
STEP 97 ================================
prereg loss 1.8240746 reg_l1 29.81613 reg_l2 21.72355
loss 4.8056874
STEP 98 ================================
prereg loss 1.8186492 reg_l1 29.820202 reg_l2 21.734375
loss 4.800669
STEP 99 ================================
prereg loss 1.8147007 reg_l1 29.824678 reg_l2 21.745687
loss 4.7971687
STEP 100 ================================
prereg loss 1.8139518 reg_l1 29.82949 reg_l2 21.757418
loss 4.7969007
STEP 101 ================================
prereg loss 1.8077824 reg_l1 29.834082 reg_l2 21.769125
loss 4.7911906
STEP 102 ================================
prereg loss 1.8052527 reg_l1 29.838913 reg_l2 21.781122
loss 4.789144
STEP 103 ================================
prereg loss 1.8047553 reg_l1 29.843937 reg_l2 21.793348
loss 4.7891493
STEP 104 ================================
prereg loss 1.8062336 reg_l1 29.849081 reg_l2 21.805737
loss 4.7911415
STEP 105 ================================
prereg loss 1.809573 reg_l1 29.854315 reg_l2 21.818245
loss 4.7950044
STEP 106 ================================
prereg loss 1.8146106 reg_l1 29.859581 reg_l2 21.830835
loss 4.8005686
STEP 107 ================================
prereg loss 1.8226936 reg_l1 29.864859 reg_l2 21.843493
loss 4.8091793
STEP 108 ================================
prereg loss 1.8242476 reg_l1 29.869638 reg_l2 21.855837
loss 4.8112116
STEP 109 ================================
prereg loss 1.8287954 reg_l1 29.874434 reg_l2 21.868261
loss 4.816239
STEP 110 ================================
prereg loss 1.8346751 reg_l1 29.879229 reg_l2 21.880741
loss 4.822598
STEP 111 ================================
prereg loss 1.8417882 reg_l1 29.884047 reg_l2 21.893288
loss 4.830193
STEP 112 ================================
prereg loss 1.850182 reg_l1 29.888859 reg_l2 21.905895
loss 4.839068
STEP 113 ================================
prereg loss 1.8541423 reg_l1 29.893192 reg_l2 21.918242
loss 4.8434615
STEP 114 ================================
prereg loss 1.8598425 reg_l1 29.89761 reg_l2 21.930723
loss 4.8496037
STEP 115 ================================
prereg loss 1.8671961 reg_l1 29.902113 reg_l2 21.943357
loss 4.8574076
STEP 116 ================================
prereg loss 1.8762059 reg_l1 29.90673 reg_l2 21.956148
loss 4.866879
STEP 117 ================================
prereg loss 1.886829 reg_l1 29.911467 reg_l2 21.969149
loss 4.877976
STEP 118 ================================
prereg loss 1.8989942 reg_l1 29.916348 reg_l2 21.982363
loss 4.890629
STEP 119 ================================
prereg loss 1.9140768 reg_l1 29.921402 reg_l2 21.995834
loss 4.906217
STEP 120 ================================
prereg loss 1.921326 reg_l1 29.926115 reg_l2 22.009228
loss 4.9139376
STEP 121 ================================
prereg loss 1.9319516 reg_l1 29.931084 reg_l2 22.022968
loss 4.9250603
cutoff 0.32193094 network size 46
STEP 122 ================================
prereg loss 1.9350516 reg_l1 29.614384 reg_l2 21.93343
loss 4.89649
STEP 123 ================================
prereg loss 1.9518523 reg_l1 29.620195 reg_l2 21.948086
loss 4.913872
STEP 124 ================================
prereg loss 1.9754108 reg_l1 29.62653 reg_l2 21.963263
loss 4.9380636
STEP 125 ================================
prereg loss 1.9908704 reg_l1 29.632744 reg_l2 21.978575
loss 4.954145
STEP 126 ================================
prereg loss 2.0127215 reg_l1 29.639465 reg_l2 21.99445
loss 4.9766684
STEP 127 ================================
prereg loss 2.0378616 reg_l1 29.64664 reg_l2 22.010887
loss 5.002526
STEP 128 ================================
prereg loss 2.0656252 reg_l1 29.654234 reg_l2 22.027882
loss 5.031049
STEP 129 ================================
prereg loss 2.0951056 reg_l1 29.662205 reg_l2 22.045422
loss 5.061326
STEP 130 ================================
prereg loss 2.1292303 reg_l1 29.670532 reg_l2 22.063517
loss 5.0962834
STEP 131 ================================
prereg loss 2.145718 reg_l1 29.678585 reg_l2 22.081768
loss 5.113577
STEP 132 ================================
prereg loss 2.1656127 reg_l1 29.686989 reg_l2 22.10057
loss 5.1343117
STEP 133 ================================
prereg loss 2.1846664 reg_l1 29.695702 reg_l2 22.119907
loss 5.154237
STEP 134 ================================
prereg loss 2.2028084 reg_l1 29.704697 reg_l2 22.139742
loss 5.173278
STEP 135 ================================
prereg loss 2.220146 reg_l1 29.71395 reg_l2 22.160059
loss 5.1915407
STEP 136 ================================
prereg loss 2.236931 reg_l1 29.723454 reg_l2 22.180815
loss 5.209276
STEP 137 ================================
prereg loss 2.2581327 reg_l1 29.733198 reg_l2 22.202017
loss 5.2314525
STEP 138 ================================
prereg loss 2.2596865 reg_l1 29.742588 reg_l2 22.22325
loss 5.2339454
STEP 139 ================================
prereg loss 2.2681653 reg_l1 29.752245 reg_l2 22.244904
loss 5.24339
STEP 140 ================================
prereg loss 2.2795238 reg_l1 29.762177 reg_l2 22.266954
loss 5.2557416
STEP 141 ================================
prereg loss 2.2941372 reg_l1 29.772377 reg_l2 22.289368
loss 5.2713747
STEP 142 ================================
prereg loss 2.31221 reg_l1 29.782814 reg_l2 22.312136
loss 5.2904916
STEP 143 ================================
prereg loss 2.335462 reg_l1 29.793509 reg_l2 22.33525
loss 5.3148127
STEP 144 ================================
prereg loss 2.347168 reg_l1 29.803808 reg_l2 22.358313
loss 5.327549
STEP 145 ================================
prereg loss 2.3653204 reg_l1 29.8144 reg_l2 22.381739
loss 5.3467607
STEP 146 ================================
prereg loss 2.3881779 reg_l1 29.82527 reg_l2 22.40552
loss 5.3707047
STEP 147 ================================
prereg loss 2.4155138 reg_l1 29.836401 reg_l2 22.429657
loss 5.3991537
STEP 148 ================================
prereg loss 2.4470172 reg_l1 29.847782 reg_l2 22.454144
loss 5.431795
STEP 149 ================================
prereg loss 2.4825988 reg_l1 29.859407 reg_l2 22.478998
loss 5.468539
STEP 150 ================================
prereg loss 2.5065093 reg_l1 29.870619 reg_l2 22.503849
loss 5.4935713
STEP 151 ================================
prereg loss 2.534686 reg_l1 29.882114 reg_l2 22.52912
loss 5.5228977
STEP 152 ================================
prereg loss 2.5664685 reg_l1 29.893875 reg_l2 22.554804
loss 5.5558558
STEP 153 ================================
prereg loss 2.6014826 reg_l1 29.905888 reg_l2 22.580906
loss 5.5920715
STEP 154 ================================
prereg loss 2.6392488 reg_l1 29.918142 reg_l2 22.607431
loss 5.6310635
STEP 155 ================================
prereg loss 2.679216 reg_l1 29.930634 reg_l2 22.634405
loss 5.6722794
STEP 156 ================================
prereg loss 2.7296865 reg_l1 29.943361 reg_l2 22.661829
loss 5.724023
STEP 157 ================================
prereg loss 2.746426 reg_l1 29.955666 reg_l2 22.68934
loss 5.741993
STEP 158 ================================
prereg loss 2.7556345 reg_l1 29.967651 reg_l2 22.71699
loss 5.7523994
STEP 159 ================================
prereg loss 2.771148 reg_l1 29.979965 reg_l2 22.745132
loss 5.7691445
STEP 160 ================================
prereg loss 2.7927268 reg_l1 29.992596 reg_l2 22.773727
loss 5.7919865
STEP 161 ================================
prereg loss 2.8204944 reg_l1 30.005527 reg_l2 22.802765
loss 5.8210473
cutoff 0.32762167 network size 45
STEP 162 ================================
prereg loss 2.993454 reg_l1 29.691126 reg_l2 22.724884
loss 5.9625664
STEP 163 ================================
prereg loss 3.0371418 reg_l1 29.70599 reg_l2 22.75563
loss 6.007741
STEP 164 ================================
prereg loss 3.0908017 reg_l1 29.720905 reg_l2 22.786661
loss 6.062892
STEP 165 ================================
prereg loss 3.1533175 reg_l1 29.735891 reg_l2 22.818027
loss 6.1269064
STEP 166 ================================
prereg loss 3.2232692 reg_l1 29.750965 reg_l2 22.84977
loss 6.1983657
STEP 167 ================================
prereg loss 3.2988896 reg_l1 29.766138 reg_l2 22.88198
loss 6.275503
STEP 168 ================================
prereg loss 3.3782182 reg_l1 29.781456 reg_l2 22.914707
loss 6.356364
STEP 169 ================================
prereg loss 3.4591541 reg_l1 29.796915 reg_l2 22.948019
loss 6.4388456
STEP 170 ================================
prereg loss 3.539743 reg_l1 29.812546 reg_l2 22.981968
loss 6.5209975
STEP 171 ================================
prereg loss 3.6182969 reg_l1 29.828358 reg_l2 23.016596
loss 6.6011324
STEP 172 ================================
prereg loss 3.693593 reg_l1 29.844393 reg_l2 23.051952
loss 6.6780324
STEP 173 ================================
prereg loss 3.764967 reg_l1 29.860651 reg_l2 23.088072
loss 6.751032
STEP 174 ================================
prereg loss 3.832366 reg_l1 29.877165 reg_l2 23.124971
loss 6.8200827
STEP 175 ================================
prereg loss 3.8962314 reg_l1 29.893967 reg_l2 23.16265
loss 6.885628
STEP 176 ================================
prereg loss 3.9574647 reg_l1 29.91105 reg_l2 23.201143
loss 6.94857
STEP 177 ================================
prereg loss 4.0172496 reg_l1 29.928457 reg_l2 23.24043
loss 7.0100956
STEP 178 ================================
prereg loss 4.0768976 reg_l1 29.946186 reg_l2 23.280514
loss 7.071516
STEP 179 ================================
prereg loss 4.1377697 reg_l1 29.964243 reg_l2 23.321383
loss 7.1341944
STEP 180 ================================
prereg loss 4.201111 reg_l1 29.982656 reg_l2 23.363035
loss 7.1993766
STEP 181 ================================
prereg loss 4.268029 reg_l1 30.001411 reg_l2 23.405445
loss 7.2681704
STEP 182 ================================
prereg loss 4.3394012 reg_l1 30.020527 reg_l2 23.448599
loss 7.341454
STEP 183 ================================
prereg loss 4.415919 reg_l1 30.039984 reg_l2 23.492489
loss 7.419917
STEP 184 ================================
prereg loss 4.49802 reg_l1 30.05978 reg_l2 23.537117
loss 7.5039983
STEP 185 ================================
prereg loss 4.5859046 reg_l1 30.079918 reg_l2 23.582468
loss 7.5938964
STEP 186 ================================
prereg loss 4.6796074 reg_l1 30.100382 reg_l2 23.628538
loss 7.689646
STEP 187 ================================
prereg loss 4.778921 reg_l1 30.121157 reg_l2 23.67532
loss 7.7910366
STEP 188 ================================
prereg loss 4.8835034 reg_l1 30.142227 reg_l2 23.722816
loss 7.897726
STEP 189 ================================
prereg loss 4.9928427 reg_l1 30.163578 reg_l2 23.771019
loss 8.009201
STEP 190 ================================
prereg loss 5.1063533 reg_l1 30.185205 reg_l2 23.819939
loss 8.124874
STEP 191 ================================
prereg loss 5.2233534 reg_l1 30.207085 reg_l2 23.869574
loss 8.244062
STEP 192 ================================
prereg loss 5.3431997 reg_l1 30.229197 reg_l2 23.91993
loss 8.366119
STEP 193 ================================
prereg loss 5.465256 reg_l1 30.251549 reg_l2 23.970991
loss 8.490411
STEP 194 ================================
prereg loss 5.5889974 reg_l1 30.274103 reg_l2 24.02278
loss 8.616407
STEP 195 ================================
prereg loss 5.714039 reg_l1 30.296875 reg_l2 24.075283
loss 8.743727
STEP 196 ================================
prereg loss 5.8401546 reg_l1 30.319841 reg_l2 24.128496
loss 8.872139
STEP 197 ================================
prereg loss 5.9672747 reg_l1 30.342997 reg_l2 24.182426
loss 9.0015745
STEP 198 ================================
prereg loss 6.0955367 reg_l1 30.366327 reg_l2 24.237066
loss 9.13217
STEP 199 ================================
prereg loss 6.2252 reg_l1 30.389845 reg_l2 24.292406
loss 9.264185
STEP 200 ================================
prereg loss 6.3566785 reg_l1 30.413538 reg_l2 24.348442
loss 9.398032
2022-07-20T17:57:54.302

julia> serialize("cf-45-parameters-matrix.ser", trainable["network_matrix"])

julia> serialize("cf-45-parameters-opt.ser", opt)

julia> interleaving_steps!(200, 40)
2022-07-20T18:24:04.802
STEP 1 ================================
prereg loss 6.4904623 reg_l1 30.4374 reg_l2 24.405182
loss 9.534203
cutoff 0.26820114 network size 44
STEP 2 ================================
prereg loss 481.41476 reg_l1 30.193228 reg_l2 24.390673
loss 484.43408
STEP 3 ================================
prereg loss 479.80563 reg_l1 30.216282 reg_l2 24.444738
loss 482.82727
STEP 4 ================================
prereg loss 477.53067 reg_l1 30.235262 reg_l2 24.493687
loss 480.5542
STEP 5 ================================
prereg loss 474.83963 reg_l1 30.251108 reg_l2 24.538576
loss 477.86475
STEP 6 ================================
prereg loss 471.8839 reg_l1 30.264587 reg_l2 24.580362
loss 474.91037
STEP 7 ================================
prereg loss 468.7641 reg_l1 30.276382 reg_l2 24.61988
loss 471.79175
STEP 8 ================================
prereg loss 465.5509 reg_l1 30.287064 reg_l2 24.657824
loss 468.57962
STEP 9 ================================
prereg loss 462.2948 reg_l1 30.29714 reg_l2 24.69483
loss 465.32452
STEP 10 ================================
prereg loss 459.0309 reg_l1 30.307066 reg_l2 24.731405
loss 462.0616
STEP 11 ================================
prereg loss 455.78387 reg_l1 30.317232 reg_l2 24.768024
loss 458.8156
STEP 12 ================================
prereg loss 452.56802 reg_l1 30.327988 reg_l2 24.80507
loss 455.60083
STEP 13 ================================
prereg loss 449.39154 reg_l1 30.339634 reg_l2 24.842888
loss 452.4255
STEP 14 ================================
prereg loss 446.25757 reg_l1 30.352432 reg_l2 24.881763
loss 449.29282
STEP 15 ================================
prereg loss 443.1644 reg_l1 30.366621 reg_l2 24.921951
loss 446.20105
STEP 16 ================================
prereg loss 440.10724 reg_l1 30.382391 reg_l2 24.96366
loss 443.14548
STEP 17 ================================
prereg loss 437.0797 reg_l1 30.399885 reg_l2 25.007057
loss 440.1197
STEP 18 ================================
prereg loss 434.07236 reg_l1 30.419249 reg_l2 25.052292
loss 437.1143
STEP 19 ================================
prereg loss 431.07486 reg_l1 30.440563 reg_l2 25.099474
loss 434.11893
STEP 20 ================================
prereg loss 428.07648 reg_l1 30.463903 reg_l2 25.148691
loss 431.12286
STEP 21 ================================
prereg loss 425.06583 reg_l1 30.489292 reg_l2 25.200016
loss 428.11475
STEP 22 ================================
prereg loss 422.0313 reg_l1 30.51674 reg_l2 25.253477
loss 425.08298
STEP 23 ================================
prereg loss 418.96198 reg_l1 30.546236 reg_l2 25.309113
loss 422.0166
STEP 24 ================================
prereg loss 415.84613 reg_l1 30.57775 reg_l2 25.36692
loss 418.9039
STEP 25 ================================
prereg loss 412.67465 reg_l1 30.61122 reg_l2 25.426899
loss 415.73578
STEP 26 ================================
prereg loss 409.43677 reg_l1 30.646576 reg_l2 25.489029
loss 412.50143
STEP 27 ================================
prereg loss 406.125 reg_l1 30.683748 reg_l2 25.55329
loss 409.1934
STEP 28 ================================
prereg loss 402.72986 reg_l1 30.722654 reg_l2 25.619625
loss 405.80212
STEP 29 ================================
prereg loss 399.24496 reg_l1 30.763184 reg_l2 25.68801
loss 402.3213
STEP 30 ================================
prereg loss 395.66364 reg_l1 30.805262 reg_l2 25.758383
loss 398.74417
STEP 31 ================================
prereg loss 391.98044 reg_l1 30.84877 reg_l2 25.8307
loss 395.0653
STEP 32 ================================
prereg loss 388.1909 reg_l1 30.893637 reg_l2 25.904905
loss 391.28024
STEP 33 ================================
prereg loss 384.29074 reg_l1 30.939735 reg_l2 25.980946
loss 387.3847
STEP 34 ================================
prereg loss 380.27728 reg_l1 30.987011 reg_l2 26.058762
loss 383.37598
STEP 35 ================================
prereg loss 376.14853 reg_l1 31.035345 reg_l2 26.138298
loss 379.25208
STEP 36 ================================
prereg loss 371.90247 reg_l1 31.084688 reg_l2 26.219505
loss 375.01093
STEP 37 ================================
prereg loss 367.5381 reg_l1 31.134924 reg_l2 26.302307
loss 370.65158
STEP 38 ================================
prereg loss 363.0565 reg_l1 31.185999 reg_l2 26.386656
loss 366.17508
STEP 39 ================================
prereg loss 358.45746 reg_l1 31.237827 reg_l2 26.472502
loss 361.58124
STEP 40 ================================
prereg loss 353.74286 reg_l1 31.29035 reg_l2 26.559772
loss 356.8719
STEP 41 ================================
prereg loss 348.9151 reg_l1 31.343498 reg_l2 26.648394
loss 352.04944
cutoff 0.1968618 network size 43
STEP 42 ================================
prereg loss 314.10037 reg_l1 31.200342 reg_l2 26.699572
loss 317.2204
STEP 43 ================================
prereg loss 309.57672 reg_l1 31.260006 reg_l2 26.792986
loss 312.70273
STEP 44 ================================
prereg loss 304.97552 reg_l1 31.320284 reg_l2 26.887653
loss 308.10754
STEP 45 ================================
prereg loss 300.30777 reg_l1 31.381027 reg_l2 26.983438
loss 303.44586
STEP 46 ================================
prereg loss 295.5846 reg_l1 31.442127 reg_l2 27.08019
loss 298.72882
STEP 47 ================================
prereg loss 290.81683 reg_l1 31.50346 reg_l2 27.177805
loss 293.9672
STEP 48 ================================
prereg loss 286.01508 reg_l1 31.56494 reg_l2 27.276148
loss 289.17157
STEP 49 ================================
prereg loss 281.18854 reg_l1 31.626467 reg_l2 27.375101
loss 284.3512
STEP 50 ================================
prereg loss 276.34723 reg_l1 31.687958 reg_l2 27.47455
loss 279.51602
STEP 51 ================================
prereg loss 271.50156 reg_l1 31.749327 reg_l2 27.574375
loss 274.67648
STEP 52 ================================
prereg loss 266.6611 reg_l1 31.810495 reg_l2 27.674444
loss 269.84216
STEP 53 ================================
prereg loss 261.8368 reg_l1 31.87137 reg_l2 27.774628
loss 265.02393
STEP 54 ================================
prereg loss 257.03845 reg_l1 31.931877 reg_l2 27.874792
loss 260.23163
STEP 55 ================================
prereg loss 252.27751 reg_l1 31.991928 reg_l2 27.974802
loss 255.4767
STEP 56 ================================
prereg loss 247.564 reg_l1 32.051445 reg_l2 28.074507
loss 250.76913
STEP 57 ================================
prereg loss 242.90964 reg_l1 32.110336 reg_l2 28.173756
loss 246.12067
STEP 58 ================================
prereg loss 238.32568 reg_l1 32.16851 reg_l2 28.2724
loss 241.54254
STEP 59 ================================
prereg loss 233.82326 reg_l1 32.225883 reg_l2 28.37026
loss 237.04585
STEP 60 ================================
prereg loss 229.4141 reg_l1 32.282356 reg_l2 28.467186
loss 232.64233
STEP 61 ================================
prereg loss 225.11005 reg_l1 32.337833 reg_l2 28.56298
loss 228.34383
STEP 62 ================================
prereg loss 220.92209 reg_l1 32.392223 reg_l2 28.657488
loss 224.16132
STEP 63 ================================
prereg loss 216.86209 reg_l1 32.44543 reg_l2 28.750505
loss 220.10663
STEP 64 ================================
prereg loss 212.94102 reg_l1 32.49735 reg_l2 28.841864
loss 216.19077
STEP 65 ================================
prereg loss 209.16989 reg_l1 32.547874 reg_l2 28.93135
loss 212.42468
STEP 66 ================================
prereg loss 205.55936 reg_l1 32.596924 reg_l2 29.018784
loss 208.81905
STEP 67 ================================
prereg loss 202.11899 reg_l1 32.644375 reg_l2 29.103956
loss 205.38342
STEP 68 ================================
prereg loss 198.85794 reg_l1 32.690136 reg_l2 29.186672
loss 202.12695
STEP 69 ================================
prereg loss 195.78432 reg_l1 32.734104 reg_l2 29.266718
loss 199.05772
STEP 70 ================================
prereg loss 192.93729 reg_l1 32.776176 reg_l2 29.343891
loss 196.2149
STEP 71 ================================
prereg loss 190.3215 reg_l1 32.81519 reg_l2 29.415949
loss 193.60303
STEP 72 ================================
prereg loss 187.90048 reg_l1 32.85124 reg_l2 29.483013
loss 191.18561
STEP 73 ================================
prereg loss 185.67522 reg_l1 32.884396 reg_l2 29.5452
loss 188.96365
STEP 74 ================================
prereg loss 183.6461 reg_l1 32.914734 reg_l2 29.60263
loss 186.93758
STEP 75 ================================
prereg loss 181.81105 reg_l1 32.942326 reg_l2 29.655441
loss 185.10529
STEP 76 ================================
prereg loss 180.16768 reg_l1 32.967247 reg_l2 29.703728
loss 183.4644
STEP 77 ================================
prereg loss 178.7111 reg_l1 32.989567 reg_l2 29.747637
loss 182.01006
STEP 78 ================================
prereg loss 177.4352 reg_l1 33.009377 reg_l2 29.787256
loss 180.73613
STEP 79 ================================
prereg loss 176.33183 reg_l1 33.026726 reg_l2 29.822714
loss 179.6345
STEP 80 ================================
prereg loss 175.39168 reg_l1 33.041706 reg_l2 29.854122
loss 178.69585
STEP 81 ================================
prereg loss 174.6043 reg_l1 33.05438 reg_l2 29.881592
loss 177.90973
cutoff 0.13356802 network size 42
STEP 82 ================================
prereg loss 173.73376 reg_l1 32.93125 reg_l2 29.8874
loss 177.02689
STEP 83 ================================
prereg loss 173.25914 reg_l1 32.940006 reg_l2 29.907373
loss 176.55315
STEP 84 ================================
prereg loss 172.89603 reg_l1 32.94661 reg_l2 29.923666
loss 176.19069
STEP 85 ================================
prereg loss 172.62953 reg_l1 32.951134 reg_l2 29.9364
loss 175.92465
STEP 86 ================================
prereg loss 172.44461 reg_l1 32.95368 reg_l2 29.945728
loss 175.73997
STEP 87 ================================
prereg loss 172.32698 reg_l1 32.95433 reg_l2 29.951796
loss 175.6224
STEP 88 ================================
prereg loss 172.26271 reg_l1 32.953194 reg_l2 29.954754
loss 175.55803
STEP 89 ================================
prereg loss 172.23918 reg_l1 32.95037 reg_l2 29.954794
loss 175.53423
STEP 90 ================================
prereg loss 172.24498 reg_l1 32.94597 reg_l2 29.952082
loss 175.53958
STEP 91 ================================
prereg loss 172.27008 reg_l1 32.940094 reg_l2 29.946808
loss 175.56409
STEP 92 ================================
prereg loss 172.3054 reg_l1 32.93288 reg_l2 29.939173
loss 175.5987
STEP 93 ================================
prereg loss 172.34433 reg_l1 32.924435 reg_l2 29.929388
loss 175.63678
STEP 94 ================================
prereg loss 172.38068 reg_l1 32.91489 reg_l2 29.917654
loss 175.67216
STEP 95 ================================
prereg loss 172.40994 reg_l1 32.90437 reg_l2 29.904198
loss 175.70038
STEP 96 ================================
prereg loss 172.4304 reg_l1 32.89299 reg_l2 29.889225
loss 175.7197
STEP 97 ================================
prereg loss 172.43903 reg_l1 32.880875 reg_l2 29.87296
loss 175.72711
STEP 98 ================================
prereg loss 172.4357 reg_l1 32.868156 reg_l2 29.855618
loss 175.72252
STEP 99 ================================
prereg loss 172.42107 reg_l1 32.85496 reg_l2 29.837399
loss 175.70656
STEP 100 ================================
prereg loss 172.39493 reg_l1 32.84139 reg_l2 29.818512
loss 175.67906
STEP 101 ================================
prereg loss 172.35957 reg_l1 32.827553 reg_l2 29.799152
loss 175.64233
STEP 102 ================================
prereg loss 172.3155 reg_l1 32.81357 reg_l2 29.779503
loss 175.59686
STEP 103 ================================
prereg loss 172.2652 reg_l1 32.79952 reg_l2 29.759733
loss 175.54515
STEP 104 ================================
prereg loss 172.20999 reg_l1 32.785507 reg_l2 29.74002
loss 175.48854
STEP 105 ================================
prereg loss 172.15225 reg_l1 32.771606 reg_l2 29.720495
loss 175.42941
STEP 106 ================================
prereg loss 172.09244 reg_l1 32.75791 reg_l2 29.701303
loss 175.36823
STEP 107 ================================
prereg loss 172.03316 reg_l1 32.744473 reg_l2 29.682556
loss 175.3076
STEP 108 ================================
prereg loss 171.97469 reg_l1 32.73137 reg_l2 29.664377
loss 175.24782
STEP 109 ================================
prereg loss 171.91905 reg_l1 32.71864 reg_l2 29.646849
loss 175.19092
STEP 110 ================================
prereg loss 171.86623 reg_l1 32.706337 reg_l2 29.630053
loss 175.13686
STEP 111 ================================
prereg loss 171.8165 reg_l1 32.694496 reg_l2 29.61406
loss 175.08595
STEP 112 ================================
prereg loss 171.77129 reg_l1 32.683155 reg_l2 29.59892
loss 175.0396
STEP 113 ================================
prereg loss 171.72986 reg_l1 32.672333 reg_l2 29.584673
loss 174.99709
STEP 114 ================================
prereg loss 171.6924 reg_l1 32.66205 reg_l2 29.571362
loss 174.9586
STEP 115 ================================
prereg loss 171.65901 reg_l1 32.65231 reg_l2 29.559
loss 174.92424
STEP 116 ================================
prereg loss 171.62946 reg_l1 32.643135 reg_l2 29.547596
loss 174.89377
STEP 117 ================================
prereg loss 171.60333 reg_l1 32.634518 reg_l2 29.537153
loss 174.86679
STEP 118 ================================
prereg loss 171.58049 reg_l1 32.626453 reg_l2 29.527664
loss 174.84314
STEP 119 ================================
prereg loss 171.56003 reg_l1 32.61893 reg_l2 29.519117
loss 174.82191
STEP 120 ================================
prereg loss 171.54199 reg_l1 32.611946 reg_l2 29.511484
loss 174.80319
STEP 121 ================================
prereg loss 171.5264 reg_l1 32.60548 reg_l2 29.504744
loss 174.78694
cutoff 0.23282906 network size 41
STEP 122 ================================
prereg loss 171.512 reg_l1 32.36668 reg_l2 29.44466
loss 174.74866
STEP 123 ================================
prereg loss 171.49915 reg_l1 32.361206 reg_l2 29.439608
loss 174.73526
STEP 124 ================================
prereg loss 171.48717 reg_l1 32.356194 reg_l2 29.435335
loss 174.7228
STEP 125 ================================
prereg loss 171.47617 reg_l1 32.35161 reg_l2 29.431808
loss 174.71133
STEP 126 ================================
prereg loss 171.46558 reg_l1 32.34742 reg_l2 29.428959
loss 174.70032
STEP 127 ================================
prereg loss 171.45518 reg_l1 32.343616 reg_l2 29.426762
loss 174.68954
STEP 128 ================================
prereg loss 171.44516 reg_l1 32.340168 reg_l2 29.425163
loss 174.67918
STEP 129 ================================
prereg loss 171.43527 reg_l1 32.337025 reg_l2 29.42411
loss 174.66898
STEP 130 ================================
prereg loss 171.42519 reg_l1 32.334183 reg_l2 29.423552
loss 174.6586
STEP 131 ================================
prereg loss 171.41466 reg_l1 32.331596 reg_l2 29.423439
loss 174.64781
STEP 132 ================================
prereg loss 171.40428 reg_l1 32.329254 reg_l2 29.423725
loss 174.6372
STEP 133 ================================
prereg loss 171.39388 reg_l1 32.327114 reg_l2 29.424356
loss 174.62659
STEP 134 ================================
prereg loss 171.38309 reg_l1 32.325153 reg_l2 29.425282
loss 174.6156
STEP 135 ================================
prereg loss 171.37227 reg_l1 32.32334 reg_l2 29.426466
loss 174.6046
STEP 136 ================================
prereg loss 171.36143 reg_l1 32.32166 reg_l2 29.42786
loss 174.5936
STEP 137 ================================
prereg loss 171.35057 reg_l1 32.320072 reg_l2 29.429422
loss 174.58258
STEP 138 ================================
prereg loss 171.33945 reg_l1 32.31857 reg_l2 29.43111
loss 174.5713
STEP 139 ================================
prereg loss 171.3283 reg_l1 32.317123 reg_l2 29.432888
loss 174.56
STEP 140 ================================
prereg loss 171.31735 reg_l1 32.315716 reg_l2 29.434713
loss 174.54892
STEP 141 ================================
prereg loss 171.30627 reg_l1 32.314323 reg_l2 29.436565
loss 174.5377
STEP 142 ================================
prereg loss 171.2952 reg_l1 32.312927 reg_l2 29.4384
loss 174.52649
STEP 143 ================================
prereg loss 171.2846 reg_l1 32.311516 reg_l2 29.4402
loss 174.51576
STEP 144 ================================
prereg loss 171.27382 reg_l1 32.310074 reg_l2 29.44193
loss 174.50482
STEP 145 ================================
prereg loss 171.26318 reg_l1 32.30859 reg_l2 29.443577
loss 174.49405
STEP 146 ================================
prereg loss 171.25278 reg_l1 32.307045 reg_l2 29.445108
loss 174.48347
STEP 147 ================================
prereg loss 171.24255 reg_l1 32.305424 reg_l2 29.446514
loss 174.4731
STEP 148 ================================
prereg loss 171.23247 reg_l1 32.303734 reg_l2 29.44778
loss 174.46284
STEP 149 ================================
prereg loss 171.22273 reg_l1 32.301956 reg_l2 29.448881
loss 174.45293
STEP 150 ================================
prereg loss 171.21281 reg_l1 32.30009 reg_l2 29.449825
loss 174.44283
STEP 151 ================================
prereg loss 171.20349 reg_l1 32.298126 reg_l2 29.450594
loss 174.4333
STEP 152 ================================
prereg loss 171.1942 reg_l1 32.29606 reg_l2 29.45117
loss 174.4238
STEP 153 ================================
prereg loss 171.185 reg_l1 32.29389 reg_l2 29.451574
loss 174.41438
STEP 154 ================================
prereg loss 171.17578 reg_l1 32.291626 reg_l2 29.451773
loss 174.40494
STEP 155 ================================
prereg loss 171.16693 reg_l1 32.28925 reg_l2 29.451786
loss 174.39586
STEP 156 ================================
prereg loss 171.15794 reg_l1 32.28677 reg_l2 29.451605
loss 174.38663
STEP 157 ================================
prereg loss 171.14941 reg_l1 32.284184 reg_l2 29.451246
loss 174.37784
STEP 158 ================================
prereg loss 171.1408 reg_l1 32.281498 reg_l2 29.450693
loss 174.36894
STEP 159 ================================
prereg loss 171.13246 reg_l1 32.278713 reg_l2 29.44997
loss 174.36034
STEP 160 ================================
prereg loss 171.12424 reg_l1 32.27583 reg_l2 29.449068
loss 174.35182
STEP 161 ================================
prereg loss 171.11572 reg_l1 32.272854 reg_l2 29.44799
loss 174.343
cutoff 0.26036114 network size 40
STEP 162 ================================
prereg loss 171.10774 reg_l1 32.00943 reg_l2 29.37897
loss 174.30869
STEP 163 ================================
prereg loss 171.09975 reg_l1 32.006294 reg_l2 29.377592
loss 174.30037
STEP 164 ================================
prereg loss 171.09149 reg_l1 32.00308 reg_l2 29.376064
loss 174.2918
STEP 165 ================================
prereg loss 171.08348 reg_l1 31.999783 reg_l2 29.374401
loss 174.28346
STEP 166 ================================
prereg loss 171.07584 reg_l1 31.99643 reg_l2 29.372627
loss 174.27548
STEP 167 ================================
prereg loss 171.06795 reg_l1 31.99301 reg_l2 29.370716
loss 174.26724
STEP 168 ================================
prereg loss 171.0602 reg_l1 31.989527 reg_l2 29.368706
loss 174.25916
STEP 169 ================================
prereg loss 171.05252 reg_l1 31.985989 reg_l2 29.366604
loss 174.25111
STEP 170 ================================
prereg loss 171.04453 reg_l1 31.98241 reg_l2 29.364408
loss 174.24277
STEP 171 ================================
prereg loss 171.03693 reg_l1 31.978788 reg_l2 29.362133
loss 174.2348
STEP 172 ================================
prereg loss 171.02966 reg_l1 31.975128 reg_l2 29.359793
loss 174.22717
STEP 173 ================================
prereg loss 171.02196 reg_l1 31.971434 reg_l2 29.357395
loss 174.2191
STEP 174 ================================
prereg loss 171.0144 reg_l1 31.967724 reg_l2 29.354933
loss 174.21118
STEP 175 ================================
prereg loss 171.0072 reg_l1 31.963976 reg_l2 29.352436
loss 174.2036
STEP 176 ================================
prereg loss 170.99982 reg_l1 31.96022 reg_l2 29.349903
loss 174.19583
STEP 177 ================================
prereg loss 170.99283 reg_l1 31.95645 reg_l2 29.34734
loss 174.18848
STEP 178 ================================
prereg loss 170.98549 reg_l1 31.952665 reg_l2 29.344759
loss 174.18076
STEP 179 ================================
prereg loss 170.97827 reg_l1 31.948877 reg_l2 29.342157
loss 174.17316
STEP 180 ================================
prereg loss 170.97142 reg_l1 31.945087 reg_l2 29.339556
loss 174.16592
STEP 181 ================================
prereg loss 170.96428 reg_l1 31.941303 reg_l2 29.33694
loss 174.15842
STEP 182 ================================
prereg loss 170.95721 reg_l1 31.937513 reg_l2 29.33433
loss 174.15097
STEP 183 ================================
prereg loss 170.95055 reg_l1 31.933731 reg_l2 29.331726
loss 174.14392
STEP 184 ================================
prereg loss 170.94347 reg_l1 31.92996 reg_l2 29.32913
loss 174.13646
STEP 185 ================================
prereg loss 170.93646 reg_l1 31.926203 reg_l2 29.326548
loss 174.12909
STEP 186 ================================
prereg loss 170.93005 reg_l1 31.922464 reg_l2 29.323982
loss 174.1223
STEP 187 ================================
prereg loss 170.92334 reg_l1 31.918724 reg_l2 29.321442
loss 174.11522
STEP 188 ================================
prereg loss 170.91656 reg_l1 31.915005 reg_l2 29.318914
loss 174.10806
STEP 189 ================================
prereg loss 170.91008 reg_l1 31.911306 reg_l2 29.316418
loss 174.10121
STEP 190 ================================
prereg loss 170.90363 reg_l1 31.90762 reg_l2 29.313948
loss 174.09439
STEP 191 ================================
prereg loss 170.89728 reg_l1 31.903954 reg_l2 29.311502
loss 174.08768
STEP 192 ================================
prereg loss 170.89061 reg_l1 31.900305 reg_l2 29.309082
loss 174.08064
STEP 193 ================================
prereg loss 170.8844 reg_l1 31.89668 reg_l2 29.306683
loss 174.07407
STEP 194 ================================
prereg loss 170.87845 reg_l1 31.89307 reg_l2 29.304323
loss 174.06775
STEP 195 ================================
prereg loss 170.87186 reg_l1 31.889477 reg_l2 29.301983
loss 174.0608
STEP 196 ================================
prereg loss 170.86583 reg_l1 31.885908 reg_l2 29.299677
loss 174.05441
STEP 197 ================================
prereg loss 170.85991 reg_l1 31.882355 reg_l2 29.29739
loss 174.04814
STEP 198 ================================
prereg loss 170.85359 reg_l1 31.87882 reg_l2 29.295132
loss 174.04147
STEP 199 ================================
prereg loss 170.8479 reg_l1 31.8753 reg_l2 29.292902
loss 174.03543
STEP 200 ================================
prereg loss 170.84164 reg_l1 31.8718 reg_l2 29.290693
loss 174.02882
2022-07-20T18:34:42.974
```
