abstract type StandardErrorEstimator end

mutable struct Bootstrap <: StandardErrorEstimator
    bootstrap_samples::Int
    subsample_size::Int
end

struct Spherical <: StandardErrorEstimator end



# Estimator Code
function ivreg(X, y, Z, se_estimator::Spherical)
    β_hat = inv(Z'X)*Z'y
    y_hat = X * β_hat 
    e = y - y_hat

    n = length(y)
    k = size(X, 2)
    s2 = e'e / (n-k)

    se = sqrt.(s2 .* diag(inv(Z'X)))

    return DataFrame(β̂ = β_hat,
                    se = se)
end

ivreg(X, y, Z) = ivreg(X, y, Z, Spherical())

function ivreg(X, y, Z, se_estimator::Bootstrap)
    n = length(y)
    n_samples = se_estimator.bootstrap_samples
    subsample_size = se_estimator.subsample_size
    β̂s = zeros(n_samples, size(X, 2))

    for k in 1:n_samples
        inds = rand(1:n, subsample_size)
        x_sub = X[inds, :]
        y_sub = y[inds, :]
        z_sub = Z[inds, :]

        β̂s[k,:] = inv(z_sub'x_sub)*z_sub'y_sub
    end

    β̂_iv = vec(mean( β̂s,  dims = 1))
    se_hat = vec(std(β̂s,  dims = 1))

    return DataFrame(β̂ = β̂_iv,
                    se = se_hat)
end

export ivreg