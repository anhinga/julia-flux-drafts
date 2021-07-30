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

But even if I start my REPL session with

```julia
zero(x::String)=""
```

I am still getting the

```julia
julia> grads = gradient(()->sum(values(mapvalues(relu, t2))), p2)
ERROR: MethodError: no method matching zero(::String)
Closest candidates are:
  zero(::Union{Type{P}, P}) where P<:Dates.Period at C:\buildbot\worker\package_win64\build\usr\share\julia\stdlib\v1.6\Dates\src\periods.jl:53
  zero(::SA) where SA<:StaticArrays.StaticArray at C:\Users\anhin\.julia\packages\StaticArrays\AHT47\src\linalg.jl:88
  zero(::ForwardDiff.Partials) at C:\Users\anhin\.julia\packages\ForwardDiff\QOqCN\src\partials.jl:39
  ...
```

diagnostics, so I obviously need help here.

OK, the right way to do this is

```julia
julia> import Base.zero

julia> zero(x::String)=""
zero (generic function with 23 methods)
```

but then we back to the original error diagnostics for the case of
mixed type keys:

```julia
julia> grads = gradient(()->sum(values(mapvalues(relu, t2))), p2)
ERROR: Mutating arrays is not supported -- called pop!(::Vector{Int64}, _...)
Stacktrace:
```

And the `pop!` it is complaining about is probably this `pop!` at line 264 of
https://github.com/JuliaCollections/FunctionalCollections.jl/blob/master/src/BitmappedVectorTrie.jl

```julia
function Base.iterate(t::SparseBitmappedTrie, state = initial_state(t))
    if isempty(state)
        return nothing
    else
        item = directindex(t, state)
        while true
            index = pop!(state)
            node = directindex(t, state)
            if length(node) > index
                push!(state, index + 1)
                return item, vcat(state, ones(Int, 1 + round(Int, t.shift / shiftby) -
                                                   length(state)))
            elseif node === arrayof(t)
                return item, Int[]
            end
        end
    end
end
```

So, I suppose, I can try to fork or copy this functionality from
`FunctionalCollections` and get inside it and tell Zygote not
to mess with it.

---

So, we indeed can get past this by replacing the above function with

```julia
import Zygote

function Base.iterate(t::SparseBitmappedTrie, state = initial_state(t))
    if isempty(state)
        return nothing
    else
        item = directindex(t, state)
        while true
            index = Zygote.@ignore pop!(state)
            node = directindex(t, state)
            if length(node) > index
                Zygote.@ignore push!(state, index + 1)
                return item, vcat(state, ones(Int, 1 + round(Int, t.shift / shiftby) -
                                                   length(state)))
            elseif node === arrayof(t)
                return item, Int[]
            end
        end
    end
end
```

and by similarly applying `Zygote.@ignore` to a couple of instances of `push!` in
https://github.com/JuliaCollections/FunctionalCollections.jl/blob/master/src/PersistentMap.jl

Then we are getting a new bug, a complaint about foreign function call in line 591 of
https://github.com/JuliaLang/julia/blob/v1.6.2/base/essentials.jl

which is the `ccall` in the function `getindex` which is called by function `iterate` in line 599.

```julia
function getindex(v::SimpleVector, i::Int)
    @boundscheck if !(1 <= i <= length(v))
        throw(BoundsError(v,i))
    end
    return ccall(:jl_svec_ref, Any, (Any, Int), v, i - 1)
end

iterate(v::SimpleVector, i=1) = (length(v) < i ? nothing : (v[i], i + 1))
```

And yes, `getindex` is an operation which Zygote hates in many situations.

I am going to record what I've done (FunctionalCollectionsMod is my local copy
of FunctionalCollections.jl which I am modifying).

```julia
julia> cd("Desktop/Julia/FunctionalCollectionsMod.jl/src")

julia> using Flux

julia> import Base: zero

julia> zero(x::String) = ""
zero (generic function with 48 methods)

julia> zero(x::Symbol) = Symbol("")
zero (generic function with 49 methods)

julia> push!(LOAD_PATH, pwd())
4-element Vector{String}:
 "@"
 "@v#.#"
 "@stdlib"
 "C:\\Users\\anhin\\Desktop\\Julia\\FunctionalCollectionsMod.jl\\src"

julia> using FunctionalCollectionsMod

julia> test_pers = @Persistent Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => -3.0]

julia> mapvalues(f, m::PersistentHashMap) = map(kv -> (kv[1], f(kv[2])), m)
mapvalues (generic function with 1 method)

julia> mapvalues(relu, test_pers)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => 0.0]

julia> sum(values(mapvalues(relu, test_pers)))
4.0f0

julia> p = params(test_pers)
Params([])

julia> grads = gradient(()->sum(values(mapvalues(relu, test_pers))), p)
ERROR: Can't differentiate foreigncall expression
Stacktrace:
  [1] error(s::String)
    @ Base .\error.jl:33
  [2] Pullback
    @ .\essentials.jl:591 [inlined]
  [3] (::typeof(∂(getindex)))(Δ::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [4] Pullback
    @ .\essentials.jl:599 [inlined]
  [5] (::typeof(∂(iterate)))(Δ::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [6] Pullback
    @ .\tuple.jl:94 [inlined]
  [7] (::typeof(∂(indexed_iterate)))(Δ::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
  [8] Pullback
    @ C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\tools\builtins.jl:17 [inlined]
  [9] (::typeof(∂(literal_indexed_iterate)))(Δ::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [10] Pullback
    @ C:\Users\anhin\Desktop\Julia\FunctionalCollectionsMod.jl\src\PersistentMap.jl:86 [inlined]
 [11] (::typeof(∂(PersistentHashMap)))(Δ::NamedTuple{(:trie, :length), Tuple{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), Tuple{Vector{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), T} where T<:Tuple}, Nothing, Nothing, Nothing, Nothing}}, Nothing}})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [12] Pullback
    @ C:\Users\anhin\Desktop\Julia\FunctionalCollectionsMod.jl\src\PersistentMap.jl:99 [inlined]
 [13] (::typeof(∂(PersistentHashMap)))(Δ::NamedTuple{(:trie, :length), Tuple{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), Tuple{Vector{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), T} where T<:Tuple}, Nothing, Nothing, Nothing, Nothing}}, Nothing}})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [14] #209
    @ C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\lib\lib.jl:203 [inlined]
 [15] #1746#back
    @ C:\Users\anhin\.julia\packages\ZygoteRules\OjfTt\src\adjoint.jl:59 [inlined]
 [16] Pullback
    @ C:\Users\anhin\Desktop\Julia\FunctionalCollectionsMod.jl\src\PersistentMap.jl:185 [inlined]
 [17] (::typeof(∂(map)))(Δ::NamedTuple{(:trie, :length), Tuple{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), Tuple{Vector{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), T} where T<:Tuple}, Nothing, Nothing, Nothing, Nothing}}, Nothing}})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [18] Pullback
    @ .\REPL[10]:1 [inlined]
 [19] (::typeof(∂(mapvalues)))(Δ::NamedTuple{(:trie, :length), Tuple{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), Tuple{Vector{NamedTuple{(:arr, :shift, :length, :maxlength, :bitmap), T} where T<:Tuple}, Nothing, Nothing, Nothing, Nothing}}, Nothing}})
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [20] Pullback
    @ .\REPL[14]:1 [inlined]
 [21] (::typeof(∂(#3)))(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface2.jl:0
 [22] (::Zygote.var"#90#91"{Zygote.Params, typeof(∂(#3)), Zygote.Context})(Δ::Float32)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface.jl:348
 [23] gradient(f::Function, args::Zygote.Params)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\compiler\interface.jl:76
 [24] top-level scope
    @ REPL[14]:1
 [25] top-level scope
    @ C:\Users\anhin\.julia\packages\CUDA\lwSps\src\initialization.jl:52
```

So, let's see what's going on here. This line 599 of `essentials.jl`

```julia
iterate(v::SimpleVector, i=1) = (length(v) < i ? nothing : (v[i], i + 1))
```

is called from line 94 of https://github.com/JuliaLang/julia/blob/v1.6.2/base/tuple.jl

```julia
function indexed_iterate(I, i, state)
    x = iterate(I, state)
    x === nothing && throw(BoundsError(I, i))
    x
end
```

and that presumably comes from https://github.com/JuliaCollections/FunctionalCollections.jl/blob/master/src/PersistentMap.jl

```
function PersistentHashMap(itr)
    if length(itr) == 0
        return PersistentHashMap()
    end
    K, V = typejoin(map(typeof, itr)...).types
    m = PersistentHashMap{K, V}()
    for (k, v) in itr
        m = assoc(m, k, v)
    end
    m
end
```

more specifically, from its line 86

```julia
    K, V = typejoin(map(typeof, itr)...).types
```

Now we need to figure out why it references `indexed_iterate` with the
first parameter being `SimpleVector` (which refers to a piece of C code)
instead of something more Julian.

Hmmm... I was about to say that we did not need to figure this out, we just needed
to block this line with `Zygote.@ignore` and I had done that in this fashion

```julia
    K, V = Zygote.@ignore typejoin(map(typeof, itr)...).types
```

but strangely enough, the diagnostics did not change. Now, if the problem is in
variable destructuring, `K, V =`, then we have a problem because blocking
assignments with `Zygote.@ignore` causes all kinds of troubles in my recent experience.

Yes, actually, if we rewrite the above as

```julia
    K_and_V = Zygote.@ignore typejoin(map(typeof, itr)...).types
    K, V = K_and_V
```

then the trace complains about the second of these lines. So we probably don't
even need `Zygote.@ignore` in the first one, but we need to figure out
what to do with this destructuring.

There is an open issue associated with destructuring:

https://github.com/FluxML/Zygote.jl/issues/303

So, let's rewrite like this:

```julia
    K_and_V = typejoin(map(typeof, itr)...).types
	K = K_and_V[1]
	V = K_and_V[2]
```

Remarkably enough, it breaks at the `getindex` triggered by the last of these 3 lines:

```julia
	V = K_and_V[2]
```

And it breaks because K_and_V is an `svec` (a SimpleVector), and indexing those
involve a `ccall`, as we have seen above.

But this is fixable in a more standard way:

```julia
	K = Zygote.@ignore K_and_V[1]
	V = Zygote.@ignore K_and_V[2]
```

And it looks like it might be the last error; at least, we have finally got

```julia
julia> grads = gradient(()->sum(values(mapvalues(relu, test_pers))), p)
Grads(...)
```

But not so simple, alas (it seems that no one has tried anything like that,
and this is why everything is so flaky when I am trying to use Zygote in
this apparently rather non-standard way):

```julia
julia> grads[test_pers]
ERROR: KeyError: key PersistentHashMap{Any, Float32}("y" => 4.0, :x => 0.0, 8 => -3.0) not found

julia> length(grads)
0
```

So this does not break outright anymore, but `params(test_pers)` creates an empty collection of parameters.

---

Note that in our very first example we had a similar pathological situation, but strangely enough things worked:

```julia
pars = Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Dict{Any, Float32} with 3 entries:
  "y" => 4.0
  8   => -3.0
  :x  => 0.0

julia> pd = params(pars)
Params([])

julia> length(pd)
0

julia> function sum_relu_dict(d)
           s = 0f0
           for k in keys(d)
               s += relu(d[k])
           end
           s
       end
sum_relu_dict (generic function with 1 method)

julia> sum_relu_dict(pars)
4.0f0

julia> gr = gradient(()->sum_relu_dict(pars), pd)
Grads(...)

julia> length(gr)
0

julia> gr[pars]
Dict{Any, Any} with 3 entries:
  "y" => 1.0
  8   => 0.0
  :x  => 0.0

julia> keys(gr)
Params([])
```

It's weird in the case of a dictionary; `gr` and `pd` seem empty on inspection, but somehow
`gr` still can access the correct result.

Unfortunately, this miracle did not manifest for PersistentHashMap. Let's investigate
what's going on.

Upgraded to newly release Flux 0.12.6. This has not changed anything.

Let's take a note of the Flux's `params` which is defined here: https://github.com/FluxML/Flux.jl/blob/v0.12.6/src/functor.jl

```julia
params!(p::Params, x::AbstractArray{<:Number}, seen = IdSet()) = push!(p, x)

function params!(p::Params, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    params!(p, child, seen)
  end
end

function params(m...)
  ps = Params()
  params!(ps, m)
  return ps
end
```

Here `Params` is from `using Zygote: Params` statement, and we indeed see
that the treatment is separate for `x::AbstractArray{<:Number}` and for
all other types of parameters.

`Params` and `Grads` live here: https://github.com/FluxML/Zygote.jl/blob/master/src/compiler/interface.jl

Looking at our two examples, we see that Params and Grads.params are non-informative
in both cases, but Grads.grads has information for our Dict example and does not have
information for our PersistentHashMap example:

```julia
julia> gr.params
Params([])

julia> gr.grads
IdDict{Any, Any} with 2 entries:
  :(Main.pars)                                   => Dict{Any, Any}("y"=>1.0, 8=>0.0, :x=>0.0)
  Dict{Any, Float32}("y"=>4.0, 8=>-3.0, :x=>0.0) => Dict{Any, Any}("y"=>1.0, 8=>0.0, :x=>0.0)

julia> grads.params
Params([])

julia> grads.grads
IdDict{Any, Any} with 1 entry:
  :(Main.test_pers) => [nothing, nothing, nothing]

julia> pd
Params([])

julia> pd.order
Zygote.Buffer{Any, Vector{Any}}(Any[], false)

julia> pd.params
Zygote.IdSet{Any}(IdDict{Any, Nothing}())

julia> p.order
Zygote.Buffer{Any, Vector{Any}}(Any[], false)
```

In the above, `gr.grads` is the only field with the information.

I have just double-checked that `pd = params(pars)` does not have anything to do with that, 
and one can simply use `gradient(()->sum_relu_dict(pars), params())`.

I have also double-checked that out of

```julia
julia> gr.grads
IdDict{Any, Any} with 2 entries:
  :(Main.pars)                                   => Dict{Any, Any}("y"=>1.0, 8=>0.0, :x=>0.0)
  Dict{Any, Float32}("y"=>4.0, 8=>-3.0, :x=>0.0) => Dict{Any, Any}("y"=>1.0, 8=>0.0, :x=>0.0)
```

it is the second of these key-value pairs which works during the normal use `gr[pars]`.

---

**I want to record a strange idea**.

So far things were breaking when one created new Dict via map-like operations, and things were
breaking when one used PersistentHashMap as Params.

However, creating new PersistentHashMap via map-like operations seems to work, and
taking Dict as Params seems to work (in some strange way which feels like a hack).

So, in principle, there is an option of starting with Dict as Params, and
generating new PersistentHashMap from that via map-like operations, and
things might just work.

I would still rather do everything with PersistentHashMap, but this is
a decent backup idea for the time being.

---

When I try to implement this idea I see that a custom adjoint is actually needed here:

```julia
julia> cd("Desktop/Julia/FunctionalCollectionsMod.jl/src")

julia> using Flux

julia> import Base: zero

julia> zero(x::String) = ""
zero (generic function with 48 methods)

julia> zero(x::Symbol) = Symbol("")
zero (generic function with 49 methods)

julia> push!(LOAD_PATH, pwd())
4-element Vector{String}:
 "@"
 "@v#.#"
 "@stdlib"
 "C:\\Users\\anhin\\Desktop\\Julia\\FunctionalCollectionsMod.jl\\src"

julia> using FunctionalCollectionsMod

julia> empty_pers = phmap{Any, Float32}()
Persistent{Any, Float32}[]

julia> pars = Dict(:x=>0f0, "y"=>4f0, 8=>-3f0)
Dict{Any, Float32} with 3 entries:
  "y" => 4.0
  8   => -3.0
  :x  => 0.0
  
julia> function phmap_from_dict(d)
           pers = phmap{Any, Float32}()
           for k in keys(d)
               pers = assoc(pers, k, d[k])
           end
           pers
       end
phmap_from_dict (generic function with 1 method)

julia> my_pers = phmap_from_dict(pars)
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => -3.0]

julia> mapvalues(f, m::PersistentHashMap) = map(kv -> (kv[1], f(kv[2])), m)
mapvalues (generic function with 1 method)

julia> mapvalues(relu, phmap_from_dict(pars))
Persistent{Any, Float32}[y => 4.0, x => 0.0, 8 => 0.0]

julia> sum(values(mapvalues(relu, phmap_from_dict(pars))))
4.0f0

julia> grads = gradient(()->sum(values(mapvalues(relu, phmap_from_dict(pars)))), params())
ERROR: Need an adjoint for constructor PersistentHashMap{Any, Float32}. Gradient is of type Vector{Nothing}
```

We can use an ugly hack to avoid using a constructor:

```julia
julia> @Persistent Dict(:dummy=>0f0, "dummy"=>0f0)
Persistent{Any, Float32}[dummy => 0.0, dummy => 0.0]

julia> keys(ans)
KeySet for a PersistentHashMap{Any, Float32} with 2 entries. Keys:
  "dummy"
  :dummy

julia> function phmap_from_dict_hack(d)
           pers = @Persistent Dict(:dummy=>0f0, "dummy"=>0f0)
           for k in keys(d)
               pers = assoc(pers, k, d[k])
           end
           pers
       end
phmap_from_dict_hack (generic function with 1 method)

julia> my_pers = phmap_from_dict_hack(pars)
Persistent{Any, Float32}[y => 4.0, dummy => 0.0, x => 0.0, 8 => -3.0, dummy => 0.0]

julia> mapvalues(relu, phmap_from_dict_hack(pars))
Persistent{Any, Float32}[y => 4.0, dummy => 0.0, x => 0.0, 8 => 0.0, dummy => 0.0]

julia> sum(values(mapvalues(relu, phmap_from_dict_hack(pars))))
4.0f0

julia> grads = gradient(()->sum(values(mapvalues(relu, phmap_from_dict_hack(pars)))), params())
ERROR: Mutating arrays is not supported -- called setindex!(::Vector{FunctionalCollectionsMod.SparseBitmappedTrie{PersistentArrayMap{Any, Float32}}}, _...)
Stacktrace:
  [1] error(s::String)
    @ Base .\error.jl:33
  [2] (::Zygote.var"#438#439"{Vector{FunctionalCollectionsMod.SparseBitmappedTrie{PersistentArrayMap{Any, Float32}}}})(#unused#::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\Zygote\TaBlo\src\lib\array.jl:76
  [3] (::Zygote.var"#2341#back#440"{Zygote.var"#438#439"{Vector{FunctionalCollectionsMod.SparseBitmappedTrie{PersistentArrayMap{Any, Float32}}}}})(Δ::Nothing)
    @ Zygote C:\Users\anhin\.julia\packages\ZygoteRules\OjfTt\src\adjoint.jl:59
  [4] Pullback
    @ C:\Users\anhin\Desktop\Julia\FunctionalCollectionsMod.jl\src\BitmappedVectorTrie.jl:236 [inlined]
```

There is a substantial use of array mutation here. Now it's really time to think what to do next.

Now it's time to read the code of Zygote and to read what people are saying about such issues, 
and perhaps to wait and see how Diffractor performs in this sense (it should be
available no later than 3 days from now, I think).

---

Insights from JuliaCon 2021.

1) The Julia AD ecosystem is way more complex and rapidly evolving than I realized.
   
Things are almost certainly doable, but the "right way" to do things is often
non-obvious (between Zygote, Diffractor, ChainRules, and Enzyme as the main players).

2) I received this advice on the "hackaton" channel of the conference Discord:

Look at implementing the following open issue, "Differential for AbstractDict",
as a hackaton task:

https://github.com/JuliaDiff/ChainRulesCore.jl/issues/186

And this might make sense as both `PersistentHashMap` and `Dict` are subtypes of
`AbstractDict`. Although if that's enough it's not clear why my attempts with Dict
would not work, given that "Dictionary differentials",

https://github.com/JuliaDiff/ChainRulesCore.jl/pull/183

was merged a year ago.

3) There is a new brilliant Dictionaries.jl (and it is **not** a subtype of AbstractDict).

So if I want to play with

https://github.com/andyferris/Dictionaries.jl

then `issue 186` does not have anything to do with that, since this is not an
`AbstractDict`, but an `AbstractDictionary` with a very different interface.

The only warning is that the author have not tried to autodiff through
these new dictionaries at all.

---

_Time for a pause_
