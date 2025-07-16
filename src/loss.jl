poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask)
#This version has a min of zero, but identical gradients to the above. Makes the loss trajectory clearer but doesn't change the optimization.
shifted_poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask) - sum(mask .* (count .- xlogy.(count, count))) / sum(mask)

#Bregman Poisson loss
bpl(mu, count) = mu .- xlogy.(count, mu)
sbpl(mu, count) = mu .- xlogy.(count, mu) .- (count .- xlogy.(count, count))
Flowfusion.floss(P::CoalescentFlow, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(sbpl(P.split_transform(X̂₁), X₁), c, mask)
#Design choice: should we enforce the splits to have a singleton dimension up front? Maybe yes, because then we can do splits, deaths, etc all in one tensor?