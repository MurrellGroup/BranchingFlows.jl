poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask)
#This version has a min of zero, but identical gradients to the above. Makes the loss trajectory clearer but doesn't change the optimization.
shifted_poisson_loss(mu, count, mask) = sum(mask .* (mu .- xlogy.(count, mu))) / sum(mask) - sum(mask .* (count .- xlogy.(count, count))) / sum(mask)

#Bregman Poisson loss
bpl(mu, count) = mu .- xlogy.(count, mu)
sbpl(mu, count) = mu .- xlogy.(count, mu) .- (count .- xlogy.(count, count))
Flowfusion.floss(P::CoalescentFlow, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(sbpl(P.split_transform(X̂₁), X₁), c, mask)

#Logit BCE for deletions:
_relu(x) = ifelse(x<0, zero(x), x)
_softplus(x) = log1p(exp(-abs(x))) + _relu(x)
_logσ(x) = -_softplus(-x)
lbce(X̂₁, X₁) = @.((1 - X₁) * X̂₁ - _logσ(X̂₁))
Flowfusion.floss(P::UniformDeletion, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(lbce(X̂₁, X₁), c, mask)

#Design choice: should we enforce the splits to have a singleton dimension up front? Maybe yes, because then we can do splits, deaths, etc all in one tensor?