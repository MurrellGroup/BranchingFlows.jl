#Bregman Poisson loss
bpl(mu, count) = mu .- xlogy.(count, mu)
sbpl(mu, count) = mu .- xlogy.(count, mu) .- (count .- xlogy.(count, count))
Flowfusion.floss(P::CoalescentFlow, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(sbpl(P.split_transform(X̂₁), X₁), c, mask)

#Logit BCE for deletions:
_relu(x) = ifelse(x<0, zero(x), x)
_softplus(x) = log1p(exp(-abs(x))) + _relu(x)
_logσ(x) = -_softplus(-x)
lbce(X̂₁, X₁) = @.((1 - X₁) * X̂₁ - _logσ(X̂₁))
Flowfusion.floss(P::Deletion, X̂₁, X₁, mask, c) = Flowfusion.scaledmaskedmean(lbce(X̂₁, X₁), c, mask)
