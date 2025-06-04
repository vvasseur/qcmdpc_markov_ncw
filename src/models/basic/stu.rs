//! This module implements the basic types for tracking syndrome weight (`s`),
//! error weight (`t`), and the number of shared bits with near codewords (`u`).
use crate::{code::MDPCCode, errors::Result, f64log::F64Log, models::traits::Counter};

/// Basic state representation for MDPC decoding with syndrome weight (s), error
/// weight (t), and common bits with nearest near codeword (u)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct STUBasic {
    /// Syndrome weight
    pub s: usize,
    /// Error weight
    pub t: usize,
    /// Number of common bits with nearest near codeword
    pub u: usize,
}

/// Counters for good, suspicious, erroneous, and bad positions in the MDPC code
#[derive(Debug, Clone)]
pub struct CountersSTU {
    /// Probability distribution for positions unrelated to the near codeword
    /// and not in the error vector
    pub good: Vec<F64Log>,
    /// Probability distribution for positions in the nearest near codeword but
    /// not in the error vector
    pub sus: Vec<F64Log>,
    /// Probability distribution for positions unrelated to the near codeword
    /// but in the error vector
    pub err: Vec<F64Log>,
    /// Probability distribution for positions in both the nearest near codeword
    /// and the error vector
    pub bad: Vec<F64Log>,
}

impl Counter<STUBasic> for CountersSTU {
    /// Computes the "locking" probability.
    ///
    /// # Arguments
    /// * `code` - Reference to the `MDPCCode`
    /// * `state` - The current state as a STUBasic struct
    /// * `threshold` - Threshold value for decoding
    ///
    /// # Returns
    /// The computed locking probability as a `F64Log`
    fn lock(&self, code: &MDPCCode, state: STUBasic, threshold: usize) -> (F64Log, F64Log) {
        let STUBasic { t, u, .. } = state;
        let CountersSTU {
            good,
            sus,
            err,
            bad,
        } = self;

        let calculate_sums = |counters: &[F64Log]| {
            let all_zeros = counters.iter().all(|val| val.is_zero());

            if all_zeros {
                F64Log::new(1.0)
            } else {
                counters.iter().take(threshold).copied().sum()
            }
        };

        let ng = (code.n + u).saturating_sub(code.d + t);
        let ns = code.d.saturating_sub(u);
        let ne = t.saturating_sub(u);
        let nb = u;

        let p_gless_t = calculate_sums(good);
        let p_sless_t = calculate_sums(sus);
        let p_eless_t = calculate_sums(err);
        let p_bless_t = calculate_sums(bad);

        let p_gless_t_ng = p_gless_t.pow(ng as f64);
        let p_sless_t_ns = p_sless_t.pow(ns as f64);
        let p_eless_t_ne = p_eless_t.pow(ne as f64);
        let p_bless_t_nb = p_bless_t.pow(nb as f64);

        let p_lock = p_gless_t_ng * p_sless_t_ns * p_eless_t_ne * p_bless_t_nb;

        let p_nolock = p_lock.complement();

        let total = p_lock + p_nolock;
        if total.is_zero() {
            (F64Log::new(1.0), F64Log::new(0.0))
        } else {
            (p_lock / total, p_nolock / total)
        }
    }

    /// Modifies probability arrays for "good", "suspicious", "erroneous", and
    /// "bad" positions based on uniform sampling.
    ///
    /// # Arguments
    /// * `code` - Reference to the `MDPCCode`
    /// * `state` - The current state as a STUBasic struct
    ///
    /// # Returns
    /// A Result containing either a new `CountersSTU` instance with modified
    /// probabilities, or an error message
    fn uniform(&self, code: &MDPCCode, state: STUBasic) -> Result<Self> {
        let STUBasic { t, u, .. } = state;
        let CountersSTU {
            good,
            sus,
            err,
            bad,
        } = self;

        let scalar_mul = |counters: &[F64Log], c: usize| {
            counters
                .iter()
                .map(|&prob| prob * (c as f64 / code.n as f64))
                .collect()
        };

        let new_good: Vec<F64Log> = scalar_mul(good, code.n - t - code.d + u);
        let new_sus: Vec<F64Log> = scalar_mul(sus, code.d - u);
        let new_err: Vec<F64Log> = scalar_mul(err, t - u);
        let new_bad: Vec<F64Log> = scalar_mul(bad, u);

        Ok(CountersSTU {
            good: new_good,
            sus: new_sus,
            err: new_err,
            bad: new_bad,
        })
    }

    /// Computes transitions for the counters.
    ///
    /// # Arguments
    /// * `threshold` - Threshold value for transitions
    ///
    /// # Returns
    /// A tuple containing a new `CountersSTU` instance and flip probability
    fn filter(&self, threshold: usize) -> (Self, F64Log) {
        let CountersSTU {
            good,
            sus,
            err,
            bad,
        } = self;
        let p_flip = [good, sus, err, bad]
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
                        F64Log::new(0.)
                    }
                })
                .collect()
        };

        (
            CountersSTU {
                good: filter(good),
                sus: filter(sus),
                err: filter(err),
                bad: filter(bad),
            },
            p_flip,
        )
    }
}
