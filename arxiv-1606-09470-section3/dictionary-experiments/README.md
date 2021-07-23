Since the ability to transform dictionary in the differentiable fashion is central to my approach,
and since it is going less smoothly than I was hoping for, let's dedicate a separate subdirectory
to experiments which involve taking gradients through dictionaries.

Note that examples in https://fluxml.ai/Zygote.jl/latest/ strongly suggest that it all should be easy.

---

Our first experiment, `gradient-of-relu-on-dict.ipynb` was actually performed on June 22. You can see that taking a gradient of a computation involving 
ReLU applied to an array is easy in Julia Flux.

However, taking a gradient of a computation involving ReLU applied to a dictionary turned out to be non-trivial, because there are some problems with gradients of iterators
in the current version of Flux/Zygote.

I successfully found a workaround here, and the gradient is computed, but I don't like this particular solution for a number of reasons:
it involves an entirely unnecessary mutation and it is not compositional.

We are going to try again.

---

The straight functional refactor does not work. We document it as `unsuccessful-refactor-of-gradient-of-relu-on-dict.ipynb`.

This is also an extremely small Jupyter notebook, but it just took me 3 reloads to render it on GitHub (use `nbviewer` if necessary).

The first problem here is that `map` is not provided for the dictionary, and the explanations for that are quite unsatisfying.

So, here is a horrible mutable workaround which works:

```julia
function my_map(my_f, my_dict)
    new_dict = deepcopy(my_dict)
    map!(my_f, values(new_dict))
    new_dict
end
```

But then we bumping into this diagnostics: https://github.com/FluxML/Zygote.jl/issues/408

So here we at least have a compositional solution, although still with encapsulated mutations (which are unfortunate),
but it does not work with the current Zygote. 

OK, the question of how to make this without mutations is a deeper Julia-related question (we have to implement `map`,
if Julia would not want to provide one of the box, and in some sense, all those immutable operations do mutate
newly allocated memory, so in some sense one can argue that `my_map` is already that), but we need to make another
at making gradients for a compositional solution work.

---

So, I tried a few things in `second-unsuccessful-refactor-of-gradient-of-relu-on-dict.ipynb`.

I tried to add type annotation to `my_map`, and I tried to replace it with

```julia
function my_map(my_f, my_dict::Dict{Any, Float32})
    new_dict = Dict{Any, Float32}()
    for k in keys(my_dict)
        new_dict[k] = f(my_dict[k])
    end
    new_dict
end
```

and I tried to incorporate adjoints from this unintegrated pull request

https://github.com/FluxML/Zygote.jl/pull/412

(not that I expected it to work, since I don't have explicit get or get!, but
I don't really know the internals of the system, so I thought to give it a chance),

but I was still getting

`MethodError: no method matching getindex(::Dict{Any, Any})`

diagnostics.

---

Julia Flux `relu` is more tastefully written, compared to my `f(x)=max(0,x)`, although this difference has not played any role in the above notebooks so far:

```julia
relu(x) = max(zero(x), x)
```

---
---
---

The right thing to do is to stop using mutable dictionaries and start using `PersistentHashMap` from https://github.com/JuliaCollections/FunctionalCollections.jl

After all, mutability of the underlying data structure here is just an annoyance,
I really want immutable dictionaries with shared common parts here. And it turns out
that `FunctionalCollections.jl` implements that, so hopefully this will just work
for us.

At least, `map` works for these structures. There are some caveats about how they can and can't be created:

```julia
julia> using FunctionalCollections

julia> test_dict = Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Dict{Any, Float32} with 3 entries:
  "y" => 4.0
  8   => -3.0
  :x  => 0.0

julia> test_pers = @Persistent test_dict
ERROR: LoadError: Unsupported @Persistent syntax
Stacktrace:
 [1] error(s::String)
   @ Base .\error.jl:33
 [2] var"@Persistent"(__source__::LineNumberNode, __module__::Module, ex::Any)
   @ FunctionalCollections C:\Users\anhin\.julia\packages\FunctionalCollections\1e7f3\src\FunctionalCollections.jl:68
in expression starting at REPL[9]:1

julia> test_pers = @Persistent Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => -3.0]

julia> mapvalues(f, m::PersistentHashMap) = map(kv -> (kv[1], f(kv[2])), m)
mapvalues (generic function with 1 method)

julia> relu(x) = max(zero(x), x)
relu (generic function with 1 method)

julia> mapvalues(relu, test_pers)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => 0.0]
```

But, frustratingly enough, this still does not work with Flux/Zygote:

```julia
julia> using Flux

julia> # it does not matter which of our ReLU will get used, my definition is the same as what's in Flux

julia> p = params(test_pers)
Params([])

julia> sum(values(mapvalues(relu, test_pers)))
4.0f0

julia> grads = gradient(()->sum(values(mapvalues(relu, test_pers))), p)
ERROR: map is not defined on dictionaries
Stacktrace:
  [1] error(s::String)
    @ Base .\error.jl:33
  [2] map(f::Zygote.StaticGetter{1}, #unused#::PersistentHashMap{Tuple{Any, Float32}, typeof(∂(#3))})
    @ Base .\abstractarray.jl:2325
  [3] macro expansion
    @ C:\Users\anhin\.julia\packages\Zygote\0da6K\src\lib\array.jl:197 [inlined]
  [4] _unzip(tuples::PersistentHashMap{Tuple{Any, Float32}, typeof(∂(#3))}, #unused#::Val{2})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\lib\array.jl:197
  [5] unzip(tuples::PersistentHashMap{Tuple{Any, Float32}, typeof(∂(#3))})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\lib\array.jl:202
  [6] ∇map(cx::Zygote.Context, f::Function, args::PersistentHashMap{Any, Float32})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\lib\array.jl:222
  [7] _pullback
    @ C:\Users\anhin\.julia\packages\Zygote\0da6K\src\lib\array.jl:256 [inlined]
  [8] _pullback
    @ C:\Users\anhin\.julia\packages\FunctionalCollections\1e7f3\src\PersistentMap.jl:185 [inlined]
  [9] _pullback(::Zygote.Context, ::typeof(map), ::var"#3#4"{typeof(relu)}, ::PersistentHashMap{Any, Float32})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\compiler\interface2.jl:0
 [10] _pullback
    @ .\REPL[11]:1 [inlined]
 [11] _pullback(::Zygote.Context, ::typeof(mapvalues), ::typeof(relu), ::PersistentHashMap{Any, Float32})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\compiler\interface2.jl:0
 [12] _pullback
    @ .\REPL[18]:1 [inlined]
 [13] _pullback(::Zygote.Context, ::var"#5#6")
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\compiler\interface2.jl:0
 [14] pullback(f::Function, ps::Zygote.Params)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\compiler\interface.jl:250
 [15] gradient(f::Function, args::Zygote.Params)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\0da6K\src\compiler\interface.jl:58
 [16] top-level scope
    @ REPL[18]:1
 [17] top-level scope
    @ C:\Users\anhin\.julia\packages\CUDA\Ozu5O\src\initialization.jl:52
```

I do not have any dictionaries here, I have PersistentHashMap, but Zygote manages to
actually create a dictionary in the process, to apply `map` to it, and then to report
that, of course, map does not work on the built-in dictionaries!

I really hope `Diffractor.jl` will be a solution to this mess.

Well, my Zygote is v0.6.14 and my Flux is 0.12.4, and we now have v0.6.17 and v0.12.5
respectively, and also, who knows, perhaps it is better to say `using Flux`
at the very beginning. Let's upgrade and try again (this is Julia 1.6.1).

The diagnostics is now different, but it still does not work (now Zygote has managed to
create a mutating array in the process, although my program is completely
immutable):

```julia
julia> using Flux
┌ Warning: The NVIDIA driver on this system only supports up to CUDA 10.2.0.
│ For performance reasons, it is recommended to upgrade to a driver that supports CUDA 11.2 or higher.
└ @ CUDA C:\Users\anhin\.julia\packages\CUDA\lwSps\src\initialization.jl:42

julia> using FunctionalCollections

julia> test_pers = @Persistent Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => -3.0]

julia> mapvalues(f, m::PersistentHashMap) = map(kv -> (kv[1], f(kv[2])), m)
mapvalues (generic function with 1 method)

julia> mapvalues(relu, test_pers)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => 0.0]

julia> p = params(test_pers)
Params([])

julia> sum(values(mapvalues(relu, test_pers)))
4.0f0

julia> grads = gradient(()->sum(values(mapvalues(relu, test_pers))), p)
ERROR: Mutating arrays is not supported -- called pop!(::Vector{Int64}, _...)
Stacktrace:
  [1] error(s::String)
    @ Base .\error.jl:33
  [2] (::Zygote.var"#450#451"{Vector{Int64}})(#unused#::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\lib\array.jl:83
  [3] (::Zygote.var"#2371#back#452"{Zygote.var"#450#451"{Vector{Int64}}})(Δ::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\ZygoteRules\OjfTt\src\adjoint.jl:59
  [4] Pullback
    @ C:\Users\anhin\.julia\packages\FunctionalCollections\1e7f3\src\BitmappedVectorTrie.jl:264 [inlined]
  [5] (::typeof(∂(iterate)))(Δ::Tuple{NamedTuple{(:kvs,), Tuple{Vector{Union{Nothing, NamedTuple{(:first, :second), Tuple{Int64, Float32}}}}}}, Nothing})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [6] Pullback
    @ C:\Users\anhin\.julia\packages\FunctionalCollections\1e7f3\src\PersistentMap.jl:176 [inlined]
  [7] (::typeof(∂(iterate)))(Δ::Tuple{NamedTuple{(:first, :second), Tuple{Int64, Float32}}, Nothing})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [8] #209
    @ C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\lib\lib.jl:203 [inlined]
  [9] (::Zygote.var"#1746#back#211"{Zygote.var"#209#210"{Tuple{Tuple{Nothing}, Tuple{Nothing}}, typeof(∂(iterate))}})(Δ::Tuple{NamedTuple{(:first, :second), Tuple{Int64, Float32}}, Nothing})
    @ Zygote C:\Users\anhin\.julia\packages\ZygoteRules\OjfTt\src\adjoint.jl:59
 [10] Pullback
    @ .\abstractdict.jl:64 [inlined]
 [11] (::typeof(∂(iterate)))(Δ::Tuple{Float32, Nothing})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [12] Pullback
    @ .\reduce.jl:60 [inlined]
 [13] (::typeof(∂(_foldl_impl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [14] Pullback
    @ .\reduce.jl:48 [inlined]
 [15] (::typeof(∂(foldl_impl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [16] Pullback
    @ .\reduce.jl:44 [inlined]
 [17] (::typeof(∂(mapfoldl_impl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [18] Pullback (repeats 2 times)
    @ .\reduce.jl:160 [inlined]
 [19] (::typeof(∂(mapfoldl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [20] Pullback
    @ .\reduce.jl:287 [inlined]
 [21] (::typeof(∂(#mapreduce#218)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [22] Pullback
    @ .\reduce.jl:287 [inlined]
 [23] (::typeof(∂(mapreduce)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [24] Pullback
    @ .\reduce.jl:501 [inlined]
 [25] (::typeof(∂(#sum#221)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [26] Pullback
    @ .\reduce.jl:501 [inlined]
 [27] (::typeof(∂(sum)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [28] Pullback
    @ .\reduce.jl:528 [inlined]
 [29] (::typeof(∂(#sum#222)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [30] Pullback
    @ .\reduce.jl:528 [inlined]
 [31] (::typeof(∂(sum)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [32] Pullback
    @ .\REPL[8]:1 [inlined]
 [33] (::typeof(∂(#3)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [34] (::Zygote.var"#90#91"{Zygote.Params, typeof(∂(#3)), Zygote.Context})(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface.jl:348
 [35] gradient(f::Function, args::Zygote.Params)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface.jl:76
 [36] top-level scope
    @ REPL[8]:1
 [37] top-level scope
    @ C:\Users\anhin\.julia\packages\CUDA\lwSps\src\initialization.jl:52
```

OK, let's make all keys to be strings. Still does not work, but the diagnostics is different again:

```julia
julia> t2 = @Persistent Dict(":x"=>0f0, "y"=>4f0, "8"=>-3f0)
Persistent{String, Float32}[y => 4.0, 8 => -3.0, :x => 0.0]

julia> p2 = params(t2)
Params([])

julia> sum(values(mapvalues(relu, t2)))
4.0f0

julia> grads = gradient(()->sum(values(mapvalues(relu, t2))), p2)
ERROR: MethodError: no method matching zero(::String)
Closest candidates are:
  zero(::Union{Type{P}, P}) where P<:Dates.Period at C:\buildbot\worker\package_win64\build\usr\share\julia\stdlib\v1.6\Dates\src\periods.jl:53
  zero(::SparseArrays.AbstractSparseArray) at C:\buildbot\worker\package_win64\build\usr\share\julia\stdlib\v1.6\SparseArrays\src\SparseArrays.jl:55
  zero(::LinearAlgebra.UniformScaling{T}) where T at C:\buildbot\worker\package_win64\build\usr\share\julia\stdlib\v1.6\LinearAlgebra\src\uniformscaling.jl:136
  ...
Stacktrace:
  [1] pair_getfield
    @ C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\lib\base.jl:134 [inlined]
  [2] #2100#back
    @ C:\Users\anhin\.julia\packages\ZygoteRules\OjfTt\src\adjoint.jl:59 [inlined]
  [3] Pullback
    @ .\pair.jl:59 [inlined]
  [4] (::typeof(∂(getindex)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [5] Pullback
    @ .\abstractdict.jl:66 [inlined]
  [6] (::typeof(∂(iterate)))(Δ::Tuple{Float32, Nothing})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [7] Pullback
    @ .\reduce.jl:60 [inlined]
  [8] (::typeof(∂(_foldl_impl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [9] Pullback
    @ .\reduce.jl:48 [inlined]
 [10] (::typeof(∂(foldl_impl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [11] Pullback
    @ .\reduce.jl:44 [inlined]
 [12] (::typeof(∂(mapfoldl_impl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [13] Pullback (repeats 2 times)
    @ .\reduce.jl:160 [inlined]
 [14] (::typeof(∂(mapfoldl)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [15] Pullback
    @ .\reduce.jl:287 [inlined]
 [16] (::typeof(∂(#mapreduce#218)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [17] Pullback
    @ .\reduce.jl:287 [inlined]
 [18] (::typeof(∂(mapreduce)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [19] Pullback
    @ .\reduce.jl:501 [inlined]
 [20] (::typeof(∂(#sum#221)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [21] Pullback
    @ .\reduce.jl:501 [inlined]
 [22] (::typeof(∂(sum)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [23] Pullback
    @ .\reduce.jl:528 [inlined]
 [24] (::typeof(∂(#sum#222)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [25] Pullback
    @ .\reduce.jl:528 [inlined]
 [26] (::typeof(∂(sum)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [27] Pullback
    @ .\REPL[12]:1 [inlined]
 [28] (::typeof(∂(#5)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [29] (::Zygote.var"#90#91"{Zygote.Params, typeof(∂(#5)), Zygote.Context})(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface.jl:348
 [30] gradient(f::Function, args::Zygote.Params)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface.jl:76
 [31] top-level scope
    @ REPL[12]:1
 [32] top-level scope
    @ C:\Users\anhin\.julia\packages\CUDA\lwSps\src\initialization.jl:52
```

A similar diagnostics if we make all keys to be symbols:

```julia
julia> t3 = @Persistent Dict(:x=>0f0, :y=>4f0, :s8=>-3f0)
Persistent{Symbol, Float32}[x => 0.0, y => 4.0, s8 => -3.0]

julia> p3 = params(t3)
Params([])

julia> sum(values(mapvalues(relu, t3)))
4.0f0

julia> grads = gradient(()->sum(values(mapvalues(relu, t3))), p3)
ERROR: MethodError: no method matching zero(::Symbol)
[and so on]
```

Same behavior for Julia 1.6.2 (as expected).

I tried to use

```julia
Zygote.@nograd String

Zygote.@nograd Symbol
```

as per https://discourse.julialang.org/t/zygote-differentiation-issues/54130

but this did not change anything (it is trying to include keys in the derivative
computations, and this causes failure; I hoped that `Zygote.@nograd String` and
`Zygote.@nograd Symbol` would prevent this, at least in cases where I used all strings or
all symbols for keys, but that did not work.
