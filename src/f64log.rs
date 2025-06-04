//! Arithmetic in log domain using the `F64Log` type.
//!
//! We need to compute probabilities with a very wide range of amplitudes. By
//! working in log domain, we can handle extremely small or large probabilities
//! accurately without numerical underflow or overflow issues.
//!
//! The `F64Log` type stores values as their natural logarithm along with a
//! sign. This lets us turn multiplication and division into addition and
//! subtraction, and use log-sum-exp for stable addition of terms with different
//! magnitudes.
use std::{
    cmp::Ordering,
    f64::consts::PI,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::EXACT_FACTORIAL_THRESHOLD;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// A type for performing arithmetic in log domain to handle very large or small
/// numbers without overflow/underflow. Stores values as natural logarithm with
/// a sign.
pub struct F64Log {
    log_value: f64,
    sign: i8,
}

impl Default for F64Log {
    fn default() -> Self {
        F64Log::new(0.)
    }
}

impl PartialEq for F64Log {
    fn eq(&self, other: &Self) -> bool {
        if self.sign != other.sign {
            return false;
        }
        if self.sign == 0 {
            return true;
        }
        self.log_value == other.log_value
    }
}

impl Eq for F64Log {}

impl PartialOrd for F64Log {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F64Log {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.sign.cmp(&other.sign) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => {
                if self.sign == 0 {
                    Ordering::Equal
                } else {
                    match self.log_value.partial_cmp(&other.log_value) {
                        Some(ord) => {
                            if self.sign > 0 {
                                ord
                            } else {
                                ord.reverse()
                            }
                        }
                        None => Ordering::Equal,
                    }
                }
            }
        }
    }
}

impl F64Log {
    /// Creates a new `F64Log` from a regular `f64` by computing its natural
    /// logarithm
    pub fn new(x: f64) -> Self {
        if x == 0.0 {
            F64Log {
                log_value: f64::NEG_INFINITY,
                sign: 0,
            }
        } else if x > 0.0 {
            F64Log {
                log_value: x.ln(),
                sign: 1,
            }
        } else {
            F64Log {
                log_value: (-x).ln(),
                sign: -1,
            }
        }
    }

    /// Creates an `F64Log` directly from a natural log value and sign
    pub fn from_ln(log_x: f64, sign: i8) -> Self {
        if log_x == f64::NEG_INFINITY {
            F64Log {
                log_value: f64::NEG_INFINITY,
                sign: 0,
            }
        } else {
            F64Log {
                log_value: log_x,
                sign,
            }
        }
    }

    /// Converts the `F64Log` back to a regular `f64` value
    pub fn as_f64(self) -> f64 {
        if self.sign == 0 {
            0.0
        } else {
            (self.sign as f64) * self.log_value.exp()
        }
    }

    /// Returns the natural logarithm of the absolute value
    pub fn ln(self) -> f64 {
        self.log_value
    }

    /// Raises the value to an `f64` power
    pub fn pow(&self, exponent: f64) -> Self {
        if self.sign == 0 {
            return F64Log::new(0.0);
        }
        if self.sign < 0 {
            panic!(
                "Cannot raise negative numbers to floating point powers - would result in complex numbers"
            );
        }
        F64Log::from_ln(self.log_value * exponent, 1)
    }

    /// Computes `1 - p`
    pub fn complement(self) -> Self {
        if self.sign <= 0 {
            return F64Log::new(1.0);
        }
        if self.log_value > 0.0 {
            return F64Log::new(0.0);
        }
        F64Log::from_ln((-self.log_value.exp()).ln_1p(), 1)
    }

    /// Compute factorial in log space using Stirling's approximation for large
    /// `n`
    pub fn factorial(n: usize) -> Self {
        if n <= 1 {
            return F64Log::new(1.0);
        }

        if n < EXACT_FACTORIAL_THRESHOLD {
            let log_value = (2..=n).map(|i| (i as f64).ln()).sum();
            F64Log::from_ln(log_value, 1)
        } else {
            let n = n as f64;
            let ln_n = n.ln();

            let log_value = n * ln_n - n + (2.0 * PI * n).ln() * 0.5 + 1.0 / (12.0 * n)
                - 1.0 / (360.0 * n * n * n)
                + 1.0 / (1260.0 * n * n * n * n * n)
                - 1.0 / (1680.0 * n * n * n * n * n * n * n);

            F64Log::from_ln(log_value, 1)
        }
    }

    /// Compute binomial coefficient in log space using factorials
    pub fn choose(n: usize, k: usize) -> Self {
        if k > n {
            return F64Log::new(0.0);
        }
        if k == 0 || k == n {
            return F64Log::new(1.0);
        }

        Self::factorial(n) / (Self::factorial(k) * Self::factorial(n - k))
    }

    /// Compute a single coefficient in the hypergeometric distribution for a
    /// population with a given sample size and number of draws.
    ///
    /// The formula is:
    /// ```text
    /// P(X = k) = binomial(sample_size, k) * binomial(pop_total - sample_size, draws - k) / binomial(pop_total, draws)
    /// ```
    ///
    /// # Arguments
    /// * `pop_total` - The total population size
    /// * `sample_size` - The size of the success sample in the population
    /// * `draws` - The number of draws in the population
    /// * `k` - The number of successes to calculate probability for
    pub fn hypergeom_probability(
        pop_total: usize,
        sample_size: usize,
        draws: usize,
        k: usize,
    ) -> Self {
        if k > sample_size
            || k > draws
            || sample_size > pop_total
            || (draws - k) > (pop_total - sample_size)
        {
            return F64Log::new(0.0);
        }

        Self::choose(sample_size, k) * Self::choose(pop_total - sample_size, draws - k)
            / Self::choose(pop_total, draws)
    }

    /// Calculates hypergeometric distribution coefficients for a given
    /// population, sample size, and number of draws in the population.
    ///
    /// # Arguments
    /// * `pop_total` - The total population size
    /// * `sample_size` - The size of the success sample in the population
    /// * `draws` - The number of draws in the population
    ///
    /// # Returns
    /// A vector containing the hypergeometric distribution coefficients
    pub fn hypergeom_distribution(pop_total: usize, sample_size: usize, draws: usize) -> Vec<Self> {
        (0..=sample_size.min(draws))
            .map(|k| Self::hypergeom_probability(pop_total, sample_size, draws, k))
            .collect()
    }

    /// Calculates binomial distribution probabilities according to the formula:
    /// `P(X = k) = binomial(d, k) * p^k * (1-p)^(d-k)`.
    ///
    /// # Arguments
    /// * `p` - The probability of success on a single trial, within the range
    ///   `[0, 1]`.
    /// * `d` - The total number of trials.
    ///
    /// # Returns
    /// A vector containing probabilities for each possible number of successes.
    pub fn binomial_distribution(p: F64Log, d: usize) -> Vec<Self> {
        if p >= F64Log::new(1.0) {
            let mut parray = vec![F64Log::new(0.0); d + 1];
            parray[d] = F64Log::new(1.0);
            parray
        } else if p <= F64Log::new(0.0) {
            let mut parray = vec![F64Log::new(0.0); d + 1];
            parray[0] = F64Log::new(1.0);
            parray
        } else {
            let ln_p = p.ln();
            let ln_q = p.complement().ln();
            (0..=d)
                .map(|k| {
                    let ln_coef = Self::choose(d, k).ln();
                    Self::from_ln(ln_coef + (k as f64) * ln_p + ((d - k) as f64) * ln_q, 1)
                })
                .collect()
        }
    }

    /// Converts a probability mass function (PMF) to a cumulative distribution
    /// function (CDF).
    ///
    /// # Arguments
    /// * `pmf` - A vector of `F64Log` representing the probability mass
    ///   function of the outcomes.
    ///
    /// # Returns
    /// A vector of `F64Log` values representing the cumulative distribution
    /// function.
    pub fn pmf_to_cdf(pmf: &[F64Log]) -> Vec<F64Log> {
        pmf.iter()
            .scan(F64Log::new(0.0), |state, p| {
                *state += *p;
                Some(*state)
            })
            .collect()
    }

    /// Calculates the cumulative distribution function (CDF) of the largest
    /// order statistic in a sample of size `n` from a set of iid random
    /// variables described by a probability mass function (PMF).
    ///
    /// # Arguments
    /// * `pmf` - Input probability mass function
    /// * `n` - Sample size
    ///
    /// # Returns
    /// CDF of maximum value in sample
    pub fn max_n_cdf(pmf: &[F64Log], n: usize) -> Vec<F64Log> {
        let cdf = Self::pmf_to_cdf(pmf);
        cdf.iter().map(|c| c.pow(n as f64)).collect()
    }

    /// Calculates the probability mass function (PMF) of the largest order
    /// statistic in a sample of size `n` from a set of iid random variables
    /// described by a probability mass function (PMF).
    ///
    /// Rather than computing differences of CDFs, which can lead to numerical
    /// instability, we use the binomial expansion:
    ///  `P(max X_i = k) = sum_{i=1}^n binom(n,i) * P(X<k)^(n-i) * P(X=k)^i`
    ///
    /// # Arguments
    /// * `pmf` - Input probability mass function
    /// * `n` - Sample size
    ///
    /// # Returns
    /// PMF of maximum value in sample
    pub fn max_n_pmf(pmf: &[F64Log], n: usize) -> Vec<F64Log> {
        let cdf = Self::pmf_to_cdf(pmf);
        let choose: Vec<_> = (0..=n).map(|k| F64Log::choose(n, k)).collect();

        (0..pmf.len())
            .into_par_iter()
            .map(|i| {
                if i == 0 {
                    cdf[0].pow(n as f64)
                } else {
                    (1..=n)
                        .map(|k| choose[k] * cdf[i - 1].pow((n - k) as f64) * pmf[i].pow(k as f64))
                        .sum()
                }
            })
            .collect()
    }

    /// Returns whether this value is zero
    pub fn is_zero(self) -> bool {
        self.sign == 0
    }

    /// Helper function to compute log-sum-exp for a collection of same-sign
    /// values
    fn compute_log_sum_exp(terms: &[f64], max_val: f64, sign: i8) -> Self {
        if terms.is_empty() {
            return F64Log::new(0.0);
        }

        if max_val == f64::NEG_INFINITY {
            return F64Log::new(0.0);
        }

        let sum: f64 = terms.iter().map(|&x| (x - max_val).exp()).sum();

        F64Log::from_ln(max_val + sum.ln(), sign)
    }
}

impl Add for F64Log {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.sign == 0 {
            return other;
        }
        if other.sign == 0 {
            return self;
        }

        if self.sign == other.sign {
            let (max, min) = if self.log_value > other.log_value {
                (self.log_value, other.log_value)
            } else {
                (other.log_value, self.log_value)
            };

            if max - min > f64::MAX_EXP as f64 * std::f64::consts::LN_2 {
                return F64Log::from_ln(max, self.sign);
            }

            F64Log::from_ln(max + (min - max).exp().ln_1p(), self.sign)
        } else {
            match self.log_value.partial_cmp(&other.log_value) {
                Some(std::cmp::Ordering::Equal) => F64Log::new(0.0),
                Some(std::cmp::Ordering::Greater) => F64Log::from_ln(
                    self.log_value + (-(-self.log_value + other.log_value).exp()).ln_1p(),
                    self.sign,
                ),
                Some(std::cmp::Ordering::Less) => F64Log::from_ln(
                    other.log_value + (-(-other.log_value + self.log_value).exp()).ln_1p(),
                    other.sign,
                ),
                None => panic!("NaN encountered in addition"),
            }
        }
    }
}

impl Sub for F64Log {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + F64Log {
            log_value: other.log_value,
            sign: -other.sign,
        }
    }
}

impl Mul for F64Log {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        F64Log::from_ln(self.log_value + other.log_value, self.sign * other.sign)
    }
}

impl Div for F64Log {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert!(other.sign != 0, "Division by zero");
        F64Log::from_ln(self.log_value - other.log_value, self.sign * other.sign)
    }
}

impl Add<f64> for F64Log {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        self + F64Log::new(other)
    }
}

impl Sub<f64> for F64Log {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        self - F64Log::new(other)
    }
}

impl Mul<f64> for F64Log {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        self * F64Log::new(other)
    }
}

impl Div<f64> for F64Log {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        self / F64Log::new(other)
    }
}

impl AddAssign for F64Log {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for F64Log {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl MulAssign for F64Log {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl DivAssign for F64Log {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl Sum for F64Log {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut pos_max = f64::NEG_INFINITY;
        let mut neg_max = f64::NEG_INFINITY;
        let mut pos_terms = Vec::new();
        let mut neg_terms = Vec::new();

        for value in iter {
            match value.sign {
                1 => {
                    if value.log_value > pos_max {
                        pos_max = value.log_value;
                    }
                    pos_terms.push(value.log_value);
                }
                -1 => {
                    if value.log_value > neg_max {
                        neg_max = value.log_value;
                    }
                    neg_terms.push(value.log_value);
                }
                // Zero values don't contribute
                _ => {}
            }
        }

        let pos_sum = Self::compute_log_sum_exp(&pos_terms, pos_max, 1);
        let neg_sum = Self::compute_log_sum_exp(&neg_terms, neg_max, -1);

        pos_sum + neg_sum
    }
}

impl std::iter::Product for F64Log {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let values: Vec<_> = iter.collect();
        if values.is_empty() {
            return F64Log::new(1.0);
        }

        let mut sign = 1;
        let mut sum = 0.0;
        for value in values {
            sign *= value.sign;
            sum += value.log_value;
        }

        F64Log::from_ln(sum, sign)
    }
}

impl From<f64> for F64Log {
    fn from(x: f64) -> Self {
        F64Log::new(x)
    }
}
