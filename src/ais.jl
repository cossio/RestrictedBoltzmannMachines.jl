# Normal AIS tends to ovestimate likelihood.
# Use http://proceedings.mlr.press/v38/burda15.html to lower-bound the log-likelihood
# together we have an interval of where log-likelihood lies.

#= Make inputs_v_to_h, inputs_h_to_v, etc... accept batches in multiple dimensions.
Useful here sice we put temperatures in one dimension and batches in another. =#
