Experiments around a DMM prototyped in Section 3 of "Programming Patterns in Dataflow Matrix Machines and Generalized Recurrent Neural Nets", https://arxiv.org/abs/1606.09470

(A version of duplicate characters or duplicate words detector. We think this is a nice toy problem to work as a playground for initial "program synthesis via DMMs" experiments.)

---

The information about experiments which involve taking gradients through dictionaries is now located in the
`dictionary-experiments` subdirectory.

I concluded that this is not doable without adding new custom rules (either to Zygote itself or to ChainRules/ChainRulesCore).
The `dictionary-experiments` subdirectory documents various unsuccessful attempts to differentiate through dictionaries
without custom rules. (Of course, I have not tried to merge the pull request allowing mutable arrays; but even if this can be
made to work in this fashion, it would be a slow and inefficient way to do things; adding new custom rules is the
correct way to do these things.)
