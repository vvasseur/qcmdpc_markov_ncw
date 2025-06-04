//! This module implements the basis model for tracking syndrome weight (`s`),
//! error weight (`t`), and the number of shared bits with near codewords (`u`).
//!
//! This forms the basis for more complex transition models that can be derived
//! from this starting point.
//! This module considers key for which we know the degrees in the subgraph of
//! the Tanner graph induced by near codewords.
use std::collections::HashMap;

use dashmap::DashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    code::{BlockDegrees, MDPCCode},
    distribution::{
        Convolve, ConvolveAll, ConvolveAllPowers, ConvolveLeft, ConvolveOne, Normalize, OffsetVec,
        Simplify, TrimZeros,
    },
    errors::Result,
    f64log::F64Log,
    models::{
        basic::{CountersSTU, STUBasic},
        traits::{CounterModel, InitialStateModel},
    },
};

use crate::ALPHA;

/// Basic counter model for STU states with specific degree distributions.
///
/// This model computes counter distributions for positions in the codeword
/// based on the current decoder state and specific degree distributions
/// for near codewords in the Tanner graph.
pub struct STUBasicSpecificCounterModel {
    code: MDPCCode,
    degrees: BlockDegrees,
    hypergeom_cache: DashMap<(usize, usize, usize), Vec<F64Log>>,
    probabilities_cache: DashMap<(usize, usize), Vec<F64Log>>,
}

impl STUBasicSpecificCounterModel {
    /// Creates a new STU basic specific counter model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `degrees` - Block degrees containing degree distributions for the
    ///   specific block
    ///
    /// # Returns
    /// A new `STUBasicSpecificCounterModel` instance
    pub fn new(code: &MDPCCode, degrees: BlockDegrees) -> Self {
        STUBasicSpecificCounterModel {
            code: code.clone(),
            degrees,
            hypergeom_cache: DashMap::new(),
            probabilities_cache: DashMap::new(),
        }
    }

    fn hypergeom(&self, n: usize, w: usize, t: usize) -> Vec<F64Log> {
        let cache = &self.hypergeom_cache;
        cache
            .entry((n, w, t))
            .or_insert_with(|| F64Log::hypergeom_distribution(n, w, t))
            .clone()
    }

    /// Computes the probabilities for the STU model
    ///
    /// # Arguments
    ///
    /// * `t` - Error weight
    /// * `u` - Number of common bits with the nearest near codeword
    ///
    /// # Returns
    ///
    /// A vector of normalized probabilities
    ///
    /// # Formulas
    ///
    /// The probabilities computed correspond to the following rho and pi values
    /// (see Lemmas 6.2-6.7 of \[ALMPRTV25] for details):
    ///
    /// Rho values:
    /// * `ρ_b1(Δ, ℓ)`: Probability that a parity-check labeled `2a` with degree
    ///   `Δ` in `G` is adjacent to `ℓ` bits in error
    /// * `ρ_b2(Δ, ℓ)`: Probability that a parity-check labeled `a+b` with
    ///   degree `Δ` in `G` is adjacent to `ℓ` bits in error, where `b` is
    ///   another bad bit
    /// * `ρ_b3(Δ, ℓ)`: Probability that a parity-check labeled `a+b` with
    ///   degree `Δ` in `G` is adjacent to `ℓ` bits in error, where `b` is a
    ///   suspicious bit
    /// * `ρ_s1(Δ, ℓ)`: Probability that a parity-check labeled `2a` with degree
    ///   `Δ` in `G` is adjacent to `ℓ` bits in error
    /// * `ρ_s2(Δ, ℓ)`: Probability that a parity-check labeled `a+b` with
    ///   degree `Δ` in `G` is adjacent to `ℓ` bits in error, where `b` is
    ///   another suspicious bit
    /// * `ρ_s3(Δ, ℓ)`: Probability that a parity-check labeled `a+b` with
    ///   degree `Δ` in `G` is adjacent to `ℓ` bits in error, where `b` is a bad
    ///   bit
    /// * `ρ_e(Δ, ℓ)`: Probability that a parity-check node of `G` of degree `Δ`
    ///   adjacent to a normal bit in error contains exactly `ℓ` erroneous bits
    /// * `ρ_g(Δ, ℓ)`: Probability that a parity-check node of `G` of degree `Δ`
    ///   adjacent to a good bit contains exactly `ℓ` erroneous bits
    ///
    /// π values:
    /// * `π_b1 = (1/d) * Σ[Δ odd] n_Δ * Σ[ℓ] ρ_b1(Δ, 2ℓ+1)`
    /// * `π_b2 = (1/(d(d-1))) * Σ[Δ ≥ 1] (Δ - Δ mod 2)n_Δ * Σ[ℓ] ρ_b2(Δ, 2ℓ+1)`
    /// * `π_b3 = (1/(d(d-1))) * Σ[Δ ≥ 1] (Δ - Δ mod 2)n_Δ * Σ[ℓ] ρ_b3(Δ, 2ℓ+1)`
    /// * `π_s1 = (1/d) * Σ[Δ odd] n_Δ * Σ[ℓ] ρ_s1(Δ, 2ℓ+1)`
    /// * `π_s2 = (1/(d(d-1))) * Σ[Δ ≥ 1] (Δ - Δ mod 2)n_Δ * Σ[ℓ] ρ_s2(Δ, 2ℓ+1)`
    /// * `π_s3 = (1/(d(d-1))) * Σ[Δ ≥ 1] (Δ - Δ mod 2)n_Δ * Σ[ℓ] ρ_s3(Δ, 2ℓ+1)`
    /// * `π_e = (1/(d*(n-d))) * Σ[Δ] (w-Δ)n_Δ * Σ[ℓ] ρ_e(Δ, 2ℓ+1)`
    /// * `π_g = (1/(d*(n-d))) * Σ[Δ] (w-Δ)n_Δ * Σ[ℓ] ρ_g(Δ, 2ℓ+1)`
    ///
    /// Where `n_Δ` is the number of parity-check equations of degree `Δ` in
    /// `G`.
    ///
    /// All π values are then normalized by dividing by `E(s|t,u)`, where:
    /// `E(s|t,u) = (1/w)(u*π_b1 + u(u-1)π_b2 + u(d-u)π_b3 + (d-u)π_s1 +
    /// (d-u)(d-u-1)π_s2 + (d-u)u*π_s3 + (t-u)d*π_e + (n+u-t-d)d*π_g)`
    pub fn compute_probabilities(&self, t: usize, u: usize) -> Vec<F64Log> {
        if let Some(cached) = self.probabilities_cache.get(&(t, u)) {
            return cached.clone();
        }

        let d = self.code.d;
        let w = self.code.w;
        let n = self.code.n;
        let d_minus_1 = d.saturating_sub(1);
        let d_minus_u_minus_1 = d.saturating_sub(u + 1);
        let u_minus_1 = u.saturating_sub(1);
        let d_minus_u = d.saturating_sub(u);
        let t_minus_u = t.saturating_sub(u);
        let n_minus_d = n.saturating_sub(d);
        let n_plus_u_minus_t_minus_d = (n + u).saturating_sub(t + d);

        let compute_rho = |deg_ncw, deg_err_ncw, deg_other, deg_err_other, delta, ell| {
            if deg_ncw <= d
                && deg_ncw <= delta
                && deg_err_ncw <= u
                && deg_err_other + u <= t
                && deg_err_ncw <= ell
                && u + deg_ncw <= d + deg_err_ncw
            {
                let coeffs1 = self.hypergeom(d - deg_ncw, delta - deg_ncw, u - deg_err_ncw);
                let coeffs2 = self.hypergeom(
                    n - d - deg_other,
                    w - delta - deg_other,
                    t - u - deg_err_other,
                );
                (deg_err_other..=ell - deg_err_ncw)
                    .map(|j| {
                        *coeffs1.get(j - deg_err_other).unwrap_or(&F64Log::new(0.))
                            * *coeffs2
                                .get(ell - deg_err_ncw - j)
                                .unwrap_or(&F64Log::new(0.))
                    })
                    .sum()
            } else {
                F64Log::new(0.)
            }
        };

        let compute_sum = |degrees: &HashMap<usize, usize>,
                           deg_ncw,
                           deg_err_ncw,
                           deg_other,
                           deg_err_other| {
            degrees
                .iter()
                .map(|(delta, &coeff)| {
                    let odd_ell_sum: F64Log = (1..=w)
                        .step_by(2)
                        .map(|ell| {
                            compute_rho(deg_ncw, deg_err_ncw, deg_other, deg_err_other, *delta, ell)
                        })
                        .sum();
                    odd_ell_sum * coeff as f64
                })
                .sum::<F64Log>()
        };

        // The following values correspond to factors in the computation of π:
        //
        // 1. double_a: corresponds to `(Δ mod 2) n_Δ`
        //
        // 2. a_plus_b: corresponds to `(Δ - Δ mod 2) n_Δ`
        //
        // 3. other: corresponds to `(w - Δ) n_Δ`
        let double_a: HashMap<usize, usize> = self
            .degrees
            .iter()
            .filter(|&(&degree, _)| degree % 2 != 0)
            .map(|(&degree, &count)| (degree, count))
            .collect();

        let a_plus_b: HashMap<usize, usize> = self
            .degrees
            .iter()
            .filter(|&(&degree, _)| degree > 1)
            .map(|(&degree, &count)| (degree, count * (degree - (degree % 2))))
            .collect();

        let other: HashMap<usize, usize> = self
            .degrees
            .iter()
            .map(|(&degree, &count)| (degree, count * (w - degree)))
            .collect();

        let sum_b1: F64Log = compute_sum(&double_a, 1, 1, 0, 0) / d as f64;
        let sum_b2: F64Log = compute_sum(&a_plus_b, 2, 2, 0, 0) / (d * d_minus_1) as f64;
        let sum_b3: F64Log = compute_sum(&a_plus_b, 2, 1, 0, 0) / (d * d_minus_1) as f64;
        let sum_s1: F64Log = compute_sum(&double_a, 1, 0, 0, 0) / d as f64;
        let sum_s2: F64Log = compute_sum(&a_plus_b, 2, 0, 0, 0) / (d * d_minus_1) as f64;
        let sum_s3: F64Log = compute_sum(&a_plus_b, 2, 1, 0, 0) / (d * d_minus_1) as f64;
        let sum_e: F64Log = compute_sum(&other, 0, 0, 1, 1) / (n_minus_d * d) as f64;
        let sum_g: F64Log = compute_sum(&other, 0, 0, 1, 0) / (n_minus_d * d) as f64;

        let weights = [
            u as f64,
            (u * u_minus_1) as f64,
            (u * d_minus_u) as f64,
            d_minus_u as f64,
            (d_minus_u * d_minus_u_minus_1) as f64,
            (d_minus_u * u) as f64,
            (t_minus_u * d) as f64,
            (n_plus_u_minus_t_minus_d * d) as f64,
        ];

        let probabilities = [sum_b1, sum_b2, sum_b3, sum_s1, sum_s2, sum_s3, sum_e, sum_g];

        let s_knowing_t_u = weights
            .iter()
            .zip(probabilities.iter())
            .map(|(&wt, &pr)| pr * wt)
            .sum::<F64Log>()
            / w as f64;

        let normalized_probabilities: Vec<F64Log> =
            probabilities.iter().map(|&p| p / s_knowing_t_u).collect();

        self.probabilities_cache
            .insert((t, u), normalized_probabilities.clone());
        normalized_probabilities
    }
}

impl CounterModel for STUBasicSpecificCounterModel {
    type State = STUBasic;
    type BasicState = STUBasic;
    type Counter = CountersSTU;

    /// Generates a counter for the given state
    ///
    /// # Arguments
    ///
    /// * `state` - The current state
    ///
    /// # Returns
    ///
    /// A CountersSTU instance
    ///
    /// # Models
    ///
    /// This function implements Model 3 of \[ALMPRTV25] for the different types
    /// of positions:
    ///
    /// 1. Bad bit counter: Modeled by the sum of:
    ///    - A Bernoulli random variable with parameter `s * π_b1 / E(s|t,u)`
    ///    - (u-1) Bernoulli random variables with parameter `s * π_b2 /
    ///      E(s|t,u)`
    ///    - (d-u) Bernoulli random variables with parameter `s * π_b3 /
    ///      E(s|t,u)`
    ///
    /// 2. Suspicious bit counter: Modeled by the sum of:
    ///    - A Bernoulli random variable with parameter `s * π_s1 / E(s|t,u)`
    ///    - (d-u-1) Bernoulli random variables with parameter `s * π_s2 /
    ///      E(s|t,u)`
    ///    - u Bernoulli random variables with parameter `s * π_s3 / E(s|t,u)`
    ///
    /// 3. Normal bit in error counter: Modeled by the sum of d Bernoulli random
    ///    variables with parameter `s * π_e / E(s|t,u)`
    ///
    /// 4. Good bit counter: Modeled by the sum of d Bernoulli random variables
    ///    with parameter `s * π_g / E(s|t,u)`
    fn get_counters_distribution(&self, state: &Self::State) -> Result<Self::Counter> {
        let STUBasic { s, t, u } = state;
        let normalized_probabilities = self.compute_probabilities(*t, *u);

        let bad = if *u > 0 {
            let x1 = F64Log::binomial_distribution(normalized_probabilities[0] * *s as f64, 1);
            let x2 = F64Log::binomial_distribution(normalized_probabilities[1] * *s as f64, *u - 1);
            let x3 = F64Log::binomial_distribution(
                normalized_probabilities[2] * *s as f64,
                self.code.d - *u,
            );
            x1.convolve(&x2, *u).convolve(&x3, self.code.d)
        } else {
            vec![F64Log::new(0.0); self.code.d + 1]
        };

        let sus = if *u < self.code.d {
            let x1 = F64Log::binomial_distribution(normalized_probabilities[3] * *s as f64, 1);
            let x2 = F64Log::binomial_distribution(
                normalized_probabilities[4] * *s as f64,
                self.code.d - *u - 1,
            );
            let x3 = F64Log::binomial_distribution(normalized_probabilities[5] * *s as f64, *u);
            x1.convolve(&x2, self.code.d - *u)
                .convolve(&x3, self.code.d)
        } else {
            vec![F64Log::new(0.0); self.code.d + 1]
        };

        let err = if *u < *t {
            F64Log::binomial_distribution(normalized_probabilities[6] * *s as f64, self.code.d)
        } else {
            vec![F64Log::new(0.0); self.code.d + 1]
        };

        let good = if self.code.n + *u >= self.code.d + *t {
            F64Log::binomial_distribution(normalized_probabilities[7] * *s as f64, self.code.d)
        } else {
            vec![F64Log::new(0.0); self.code.d + 1]
        };

        Ok(CountersSTU {
            good,
            sus,
            err,
            bad,
        })
    }
}

/// Basic initial state model for STU states with specific degree distributions.
///
/// This model computes the initial probability distribution of states
/// for a given error weight using specific degree distributions and
/// hypergeometric distributions.
pub struct STUBasicSpecificInitialStateModel {
    code: MDPCCode,
    degrees: BlockDegrees,
    hypergeom_cache: DashMap<(usize, usize, usize), Vec<F64Log>>,
}

impl STUBasicSpecificInitialStateModel {
    /// Creates a new STU basic specific initial state model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `degrees` - Block degrees containing degree distributions for the
    ///   specific block
    ///
    /// # Returns
    /// A new `STUBasicSpecificInitialStateModel` instance
    pub fn new(code: &MDPCCode, degrees: BlockDegrees) -> Self {
        STUBasicSpecificInitialStateModel {
            code: code.clone(),
            degrees,
            hypergeom_cache: DashMap::new(),
        }
    }

    fn hypergeom(&self, n: usize, w: usize, t: usize) -> Vec<F64Log> {
        let cache = &self.hypergeom_cache;
        cache
            .entry((n, w, t))
            .or_insert_with(|| F64Log::hypergeom_distribution(n, w, t))
            .clone()
    }
}

impl InitialStateModel for STUBasicSpecificInitialStateModel {
    type State = STUBasic;
    /// Generates the initial state distribution for a given error weight.
    ///
    /// Details are given in Appendix A of \[ALMPRTV25].
    ///
    /// # Arguments
    ///
    /// * `t` - error weight for which to compute the initial distribution
    ///
    /// # Returns
    ///
    /// A HashMap mapping each possible initial state to its probability
    fn get_initial_distribution(&self, t: usize) -> HashMap<Self::State, F64Log> {
        let r = self.code.r;
        let n = self.code.n;
        let d = self.code.d;
        let w = self.code.w;
        let alpha = F64Log::from_ln(-10f64.powf(-ALPHA as f64).ln_1p(), 1);

        (0..=d)
            .flat_map(|u| {
                // Helper function to compute rho values
                let compute_rho = |delta, ell| {
                    let coeffs1 = self.hypergeom(d, delta, u);
                    let coeffs2 = self.hypergeom(n - d, w - delta, t - u);
                    (0..=ell)
                        .map(|j| {
                            coeffs1.get(j).copied().unwrap_or(F64Log::new(0.))
                                * coeffs2.get(ell - j).copied().unwrap_or(F64Log::new(0.))
                        })
                        .sum()
                };

                let degrees = &self.degrees;

                // Compute rho values for each degree
                let rhos: HashMap<usize, OffsetVec<F64Log>> = degrees
                    .keys()
                    .map(|&delta| {
                        let distribution = (0..=w)
                            .map(|ell| compute_rho(delta, ell))
                            .collect::<Vec<F64Log>>();
                        (
                            delta,
                            OffsetVec {
                                offset: 0,
                                stride: 1,
                                vec: distribution,
                            },
                        )
                    })
                    .collect();

                // Split rhos into odd and even parts
                let (rhos_odd, rhos_even): (
                    HashMap<usize, OffsetVec<F64Log>>,
                    HashMap<usize, OffsetVec<F64Log>>,
                ) = rhos
                    .into_iter()
                    .map(|(delta, rho)| {
                        let (odd, even) = rho.split_parity();
                        ((delta, odd.trim_zeros()), (delta, even.trim_zeros()))
                    })
                    .unzip();

                // Helper function to compute convolution powers for all possible s <= weight
                let compute_convolution_powers = |vec: OffsetVec<F64Log>, weight: usize| {
                    // Compute all convolution powers at once
                    let convolution_powers = vec.convolve_all_powers(weight, d * t);

                    // Divide each power by its factorial
                    let convolution_powers: HashMap<_, _> = convolution_powers
                        .into_par_iter()
                        .map(|(s, convolved_vec)| {
                            // Divide each component by s! (fact_s)
                            // This is done to ensure that when we convolve this vector with others
                            // later, we're effectively computing the
                            // exponential generating function of the distribution
                            let divided_vec = convolved_vec
                                .vec
                                .iter()
                                .map(|component| *component / F64Log::factorial(s))
                                .collect::<Vec<_>>();

                            (
                                s,
                                OffsetVec {
                                    offset: convolved_vec.offset,
                                    stride: convolved_vec.stride,
                                    vec: divided_vec,
                                },
                            )
                        })
                        .collect();
                    convolution_powers
                };

                // Compute convolution powers for odd and even parts of each rho (for each
                // degree delta)
                let s_odd: HashMap<usize, _> = rhos_odd
                    .into_par_iter()
                    .map(|(delta, s)| {
                        (
                            delta,
                            compute_convolution_powers(s, *degrees.get(&delta).unwrap()),
                        )
                    })
                    .collect();

                let s_even: HashMap<usize, _> = rhos_even
                    .into_par_iter()
                    .map(|(delta, s)| {
                        (
                            delta,
                            compute_convolution_powers(s, *degrees.get(&delta).unwrap()),
                        )
                    })
                    .collect();

                // For each degree delta, we convolve s nodes with odd rho and (degrees[delta] -
                // s) nodes with even rho This ensures we have exactly
                // degrees[delta] nodes of degree delta in total
                let s: Vec<_> = s_odd
                    .into_par_iter()
                    .map(|(delta, odd)| {
                        let even = s_even.get(&delta).unwrap();
                        let weight = { *degrees.get(&delta).unwrap() };
                        odd.convolve_left(even, weight, d * t).simplify(&alpha)
                    })
                    .collect();

                // Iteratively convolve the array of sequence distributions together:
                // We only need to compute one specific term in the final convolution.
                // So we split the sequences into the largest one and the rest
                // Then convolve all the rest together first
                // And finally use convolve_one() for the final convolution with the largest
                // sequence This avoids computing unnecessary terms in the final
                // convolution
                let (s_largest, s_rest) = {
                    let mut sorted = s.clone();
                    sorted.sort_by_key(|m| m.len());
                    let largest = sorted.pop().unwrap_or_default();
                    (largest, sorted)
                };

                let s_sum = s_rest
                    .into_iter()
                    .convolve_all(d * t)
                    .unwrap_or_else(HashMap::new)
                    .simplify(&alpha);

                // Compute syndrome distribution
                // This calculation considers only the convolutions of s_sum with s_largest
                // where the total weight is d*t. d*t represents the total
                // number of connected edges in the Tanner graph between parity checks and error
                // bits. The resulting distribution gives probabilities for
                // different syndrome weight values at this fixed total
                // weight d*t.
                let s_dist: Vec<F64Log> = (0..=r)
                    .into_par_iter()
                    .map(|s| {
                        if d * t % 2 == s % 2 && s + u * u.saturating_sub(1) <= d * t {
                            s_sum.convolve_one(&s_largest, &[s, (d * t)])
                        } else {
                            F64Log::new(0.)
                        }
                    })
                    .collect();

                let norm_s_dist = s_dist.normalize().unwrap_or_default();

                // Create STUState for each s and multiply by probability of u
                norm_s_dist
                    .into_iter()
                    .enumerate()
                    .filter_map(|(s, p)| {
                        if !p.is_zero() {
                            Some((STUBasic { s, t, u }, p))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
