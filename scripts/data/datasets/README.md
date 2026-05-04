Here we maintain a couple of specially formatted `.csv` files with *factual* outcomes.

Columns have prefixes `t:`, `c:`, and `o:` to indicate treatments, covariates, and outcomes, respectively.

The idea is to enable generation of SCMs from these files, in which the possible values for covariates and treatments are fixed, but we can generate (possibly) noisy data for the outcome variables.

Maybe we want to only allow to configure covariate combinations that occur in the original data, but maybe we want to only fix the domains of the covariates by those values and allow arbitrary combinations.