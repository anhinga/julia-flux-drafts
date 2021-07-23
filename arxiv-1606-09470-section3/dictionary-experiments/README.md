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
