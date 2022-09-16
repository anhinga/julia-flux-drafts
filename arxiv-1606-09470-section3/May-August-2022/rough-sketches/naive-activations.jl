# Julia 1606-09470

# preliminary sketches which might be rather incorrect ("an underdrawing")

# eventually, we'd like to recreate flexibility of our Clojure implementation +
# various flexibility enhancements described in https://www.cs.brandeis.edu/~bukatin/dmm-notes-2018.pdf
# and performance enhancements associated with full-strength Pytrees or their equivalent
# and also described in https://github.com/anhinga/2019-design-notes and also
# in https://github.com/anhinga/2019-design-notes and its predecessors.

# But here we might start with a simplified setup, trying to establish
# applicability of differentiable programming methods first. And then this simplified
# setup can be used as a regression test, when we eventually produce more
# sophisticated systematic implementations.

# We might try various setups, as each of those setups would form a separate
# benchmark for differentiable programming systems, and hence will be useful
# for our project of assembling a corpus of differentiable programming benchmarks.

# Let's start with "naive activation functions". We'll modify them in various ways.

# (Plenty of bugs in this auxiliary sketch; this file is not a part of any running code.)

using Flux

zero_value = 0.0f0
one_value = 1.0f0
end_of_text = "."

# I am sure this one can be written better
# d = d1 + d2 for dictionaries
#
function sum_dicts(d1, d2)
    d = deepcopy(d1)
	for k in keys(d2)
		d[k] = get(d, k, zero_value) + d2[k]
	end
	d
end

# sum_dicts can be used as the activation function of an accumulator
		    
# one possible way to write a linear combination
# (not necessarily the way we want)
function lin_comb_dicts(coefs, dicts)
    d = Dict()
	n = min(length(coefs), length(dicts))
	for i in 1:n
	    for k in keys(dicts[i])
		    d[k] = get(d, k, zero_value) + coefs[i]*dicts[i][k]
		end
    end
	d
end

# max of (dictionary values and zero)
function max_norm_zero(d)
    s = zero_value
	for k in keys(d)
	    s = max(s, d[k])
    end
	s
end

# dot product of dictionaries
function dot_product_dicts(d1, d2)
    s = zero_value
	for k in keys(d1)
	    s += d1[k] * get(d2, k, zero_value)
	end
	s
end

# const number 1
function const_1()
    one_value
end

# const end_of_string or end_of_fragment
function const_end()
    d = Dict()
	d[end_of_text] = one_value
	d
end


# function output will exist in two incarnations
#   * output the result
#   * compute the loss
