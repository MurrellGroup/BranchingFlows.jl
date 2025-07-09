poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask)
#This version has a min of zero, but identical gradients to the above. Makes the loss trajectory clearer but doesn't change the optimization.
shifted_poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask) - sum(mask .* (count .- xlogy.(count, count))) / sum(mask)
