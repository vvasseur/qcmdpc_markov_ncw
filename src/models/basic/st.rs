//! This module implements the basis model for tracking syndrome weight (s),
//! error weight (t).
//!
//! This forms the basis for more complex transition models that can be derived
//! from this starting point.
use std::collections::HashMap;

use dashmap::DashMap;

use crate::{
    code::MDPCCode,
    distribution::{ConvolveAllPowers, ConvolveOne, Normalize, OffsetVec, TrimZeros},
    errors::Result,
    f64log::F64Log,
    models::traits::{Counter, CounterModel, InitialStateModel},
};

/// Basic state representation for MDPC decoding with syndrome weight (s) and
/// error weight (t)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct STBasic {
    /// Syndrome weight
    pub s: usize,
    /// Error weight
    pub t: usize,
}

/// Basic counter model for ST (syndrome weight, error weight) states.
///
/// This model computes counter distributions for positions in the codeword
/// based on the current decoder state and xi parameter.
pub struct STBasicCounterModel {
    code: MDPCCode,
    xi: f64,
    hypergeom_cache: DashMap<(usize, usize, usize), Vec<F64Log>>,
}

impl STBasicCounterModel {
    /// Creates a new ST basic counter model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `xi` - Xi parameter controlling the counter distribution
    ///
    /// # Returns
    /// A new `STBasicCounterModel` instance
    pub fn new(code: &MDPCCode, xi: f64) -> Self {
        STBasicCounterModel {
            code: code.clone(),
            xi,
            hypergeom_cache: DashMap::new(),
        }
    }

    fn avg_x(&self, t: usize) -> f64 {
        let rho_table = self.hypergeom(self.code.n, self.code.w, t);

        let mut x = F64Log::new(0.);
        let mut denom = F64Log::new(0.);

        for i in (1..std::cmp::min(10, t)).step_by(2) {
            x += rho_table[i] * (i as f64 - 1.0);
            denom += rho_table[i];
        }

        if denom.is_zero() {
            0.0
        } else {
            (x / denom).as_f64()
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

impl CounterModel for STBasicCounterModel {
    type State = STBasic;
    type BasicState = STBasic;
    type Counter = CountersST;

    fn get_counters_distribution(&self, state: &Self::State) -> Result<Self::Counter> {
        let STBasic { s, t } = state;
        let p0 = F64Log::new(
            (((self.code.w - 1) * s) as f64 - self.avg_x(*t) * *s as f64)
                / ((self.code.n - t) as f64)
                / self.code.d as f64,
        );
        let p1 = F64Log::new(
            (*s as f64 + self.xi * self.avg_x(*t) * *s as f64) / *t as f64 / self.code.d as f64,
        );
        Ok(CountersST {
            good: F64Log::binomial_distribution(p0, self.code.d),
            bad: F64Log::binomial_distribution(p1, self.code.d),
        })
    }
}

/// Basic initial state model for ST (syndrome weight, error weight) states.
///
/// This model computes the initial probability distribution of states
/// for a given error weight using hypergeometric distributions.
pub struct STBasicInitialStateModel {
    code: MDPCCode,
    hypergeom_cache: DashMap<(usize, usize, usize), Vec<F64Log>>,
}

impl STBasicInitialStateModel {
    /// Creates a new ST basic initial state model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    ///
    /// # Returns
    /// A new `STBasicInitialStateModel` instance
    pub fn new(code: &MDPCCode) -> Self {
        STBasicInitialStateModel {
            code: code.clone(),
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

impl InitialStateModel for STBasicInitialStateModel {
    type State = STBasic;

    fn get_initial_distribution(&self, t: usize) -> HashMap<Self::State, F64Log> {
        let r = self.code.r;
        let n = self.code.n;
        let d = self.code.d;
        let w = self.code.w;

        let rho = OffsetVec {
            offset: 0,
            stride: 1,
            vec: self.hypergeom(n, w, t),
        }
        .trim_zeros();
        let (rho_odd, rho_even) = rho.split_parity();

        let compute_convolution_powers = |vec: OffsetVec<F64Log>| {
            // Compute all convolution powers at once
            let convolution_powers = vec.convolve_all_powers(r, d * t);

            // Divide each power by its factorial
            convolution_powers
                .into_iter()
                .map(|(s, convolved_vec)| {
                    // Divide each component by s! (fact_s)
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
                .collect()
        };

        let s_odd: HashMap<_, _> = compute_convolution_powers(rho_odd);
        let s_even: HashMap<_, _> = compute_convolution_powers(rho_even);

        // Compute syndrome distribution
        // Computes probabilities of syndrome weights by convolving the
        // odd and even distributions, considering a fixed total weight
        // d*t representing edges between parity checks and error bits.
        let s_dist: Vec<F64Log> = (0..=r)
            .map(|s| s_odd.convolve_one(&s_even, &[s, (d * t)]))
            .collect();

        let norm_s_dist = s_dist.normalize().unwrap_or_default();

        norm_s_dist
            .into_iter()
            .enumerate()
            .filter_map(|(s, p)| {
                if !p.is_zero() {
                    Some((STBasic { s, t }, p))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Counters for good and bad positions in the MDPC code
#[derive(Debug, Clone)]
pub struct CountersST {
    /// Probability distribution for positions not in the error vector (n - t
    /// positions)
    pub good: Vec<F64Log>,
    /// Probability distribution for positions in the error vector (t positions)
    pub bad: Vec<F64Log>,
}

impl Counter<STBasic> for CountersST {
    /// Computes the "locking" probability.
    ///
    /// # Arguments
    /// * `code` - Reference to the `MDPCCode`
    /// * `state` - The current state as a STBasic struct
    /// * `threshold` - Threshold value used for decoding
    ///
    /// # Returns
    /// The computed locking probability as a `F64Log`
    fn lock(&self, code: &MDPCCode, state: STBasic, threshold: usize) -> (F64Log, F64Log) {
        let STBasic { t, .. } = state;
        let CountersST { good, bad } = self;

        let calculate_sums = |counters: &[F64Log]| {
            let all_zeros = counters.iter().all(|val| val.is_zero());

            if all_zeros {
                F64Log::new(1.0)
            } else {
                counters.iter().take(threshold).copied().sum()
            }
        };

        let p_gless_t = calculate_sums(good);
        let p_bless_t = calculate_sums(bad);

        let result = p_gless_t.pow(code.n.saturating_sub(t) as f64) * p_bless_t.pow(t as f64);
        if result.as_f64() > 1.0 {
            (F64Log::new(1.0), F64Log::new(0.0))
        } else {
            (result, result.complement())
        }
    }

    /// Modifies probability arrays for "good" and "bad" positions based on
    /// uniform sampling.
    ///
    /// # Arguments
    /// * `code` - Reference to the `MDPCCode`
    /// * `state` - The current state as a STBasic struct
    ///
    /// # Returns
    /// A new `CountersST` instance with modified probabilities
    fn uniform(&self, code: &MDPCCode, state: STBasic) -> Result<Self> {
        let STBasic { t, .. } = state;
        let CountersST { good, bad } = self;

        let scalar_mul = |counters: &[F64Log], c: usize| {
            counters
                .iter()
                .map(|&prob| prob * F64Log::new(c as f64 / code.n as f64))
                .collect()
        };

        let new_good = scalar_mul(good, code.n - t);
        let new_bad = scalar_mul(bad, t);

        Ok(CountersST {
            good: new_good,
            bad: new_bad,
        })
    }

    /// Computes transitions for the counters.
    ///
    /// # Arguments
    /// * `threshold` - Threshold value used for decoding
    ///
    /// # Returns
    /// A tuple containing a new `CountersST` instance and flip/no-flip
    /// probabilities
    fn filter(&self, threshold: usize) -> (Self, F64Log) {
        let CountersST { good, bad } = self;
        let p_flip = [good, bad]
            .iter()
            .flat_map(|c| c.iter().skip(threshold))
            .copied()
            .sum();

        let filter = |counters: &[F64Log]| {
            counters
                .iter()
                .enumerate()
                .map(|(idx, &prob)| {
                    if idx >= threshold {
                        prob
                    } else {
                        F64Log::new(0.0)
                    }
                })
                .collect()
        };

        (
            CountersST {
                good: filter(good),
                bad: filter(bad),
            },
            p_flip,
        )
    }
}
