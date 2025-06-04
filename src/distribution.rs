//! Convolutions and other operations on probability distributions.
//!
//! Core operations for probability distributions including convolution,
//! normalization, and various algebraic operations.
//!
//! We use `OffsetVec` to efficiently represent distributions with 'small'
//! support and handle odd/even value separation.
use std::{collections::HashMap, hash::Hash};

use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::f64log::F64Log;

/// A vector with offset and stride for efficient distribution operations.
#[derive(Clone, Debug)]
pub struct OffsetVec<T> {
    /// Offset of the vector.
    pub offset: usize,
    /// Stride between elements, where `v[i] = v'[offset + stride * i]`
    pub stride: usize,
    /// The underlying vector data.
    pub vec: Vec<T>,
}

impl<T: Clone> OffsetVec<T> {
    /// Splits a vector into odd and even indexed elements.
    ///
    /// # Returns
    /// A tuple containing:
    /// * Odd-indexed elements with adjusted offset and stride
    /// * Even-indexed elements with adjusted offset and stride
    pub fn split_parity(&self) -> (OffsetVec<T>, OffsetVec<T>) {
        let even_vec: Vec<T> = self.vec.iter().cloned().step_by(2).collect();
        let odd_vec: Vec<T> = self.vec.iter().skip(1).cloned().step_by(2).collect();

        let even_offset_vec = OffsetVec {
            offset: self.offset,
            stride: 2 * self.stride,
            vec: even_vec,
        };
        let odd_offset_vec = OffsetVec {
            offset: self.offset + self.stride,
            stride: 2 * self.stride,
            vec: odd_vec,
        };

        (odd_offset_vec, even_offset_vec)
    }
}

/// Convolve two distributions.
pub trait Convolve {
    /// Computes the convolution up to a maximum index.
    ///
    /// # Arguments
    /// * `other` - Distribution to convolve with
    /// * `max_idx` - Upper bound for result indices
    ///
    /// # Returns
    /// The convolved distribution
    fn convolve(&self, other: &Self, max_idx: usize) -> Self;

    /// Optional convolution that returns None for empty results.
    ///
    /// # Arguments
    /// * `other` - Distribution to convolve with
    /// * `max_idx` - Upper bound for result indices
    ///
    /// # Returns
    /// Some(result) or None if inputs are empty
    fn convolve_opt(&self, other: &Self, max_idx: usize) -> Option<Self>
    where
        Self: Sized;
}

/// Computes individual terms in convolution operations.
pub trait ConvolveOne {
    /// Computes one convolution term for given indices.
    ///
    /// # Arguments
    /// * `other` - Distribution to convolve with
    /// * `idx` - Array of indices for nested indexing
    ///
    /// # Returns
    /// The single convolution term
    fn convolve_one(&self, other: &Self, idx: &[usize]) -> F64Log;
}

/// Performs 'indexed' convolution from the left side.
pub trait ConvolveLeft {
    /// Computes `Z[i] = X[i] ⊗ Y[idx - i]` up to `max_idx`.
    ///
    /// # Arguments
    /// * `other` - Distribution Y to convolve with
    /// * `idx` - Index for operation
    /// * `max_idx` - Upper bound for indices
    ///
    /// # Returns
    /// The resulting distribution Z
    fn convolve_left(&self, other: &Self, idx: usize, max_idx: usize) -> Self;
}

/// Specializes convolution with itself.
///
/// This is usually more efficient than regular convolution since we can
/// take advantage of symmetry in the calculation.
pub trait ConvolveSelf {
    /// Computes self-convolution up to `max_idx`.
    fn convolve_self(&self, max_idx: usize) -> Self;

    /// Optional self-convolution returning None for empty results.
    fn convolve_self_opt(&self, max_idx: usize) -> Option<Self>
    where
        Self: Sized;
}

/// Generates all convolution powers of a distribution.
pub trait ConvolveAllPowers {
    /// Computes convolution powers from 0 to `n`.
    ///
    /// # Arguments
    /// * `n` - Maximum power to compute
    /// * `max_idx` - Upper bound for indices
    ///
    /// # Returns
    /// Map from exponents to their convolution powers
    fn convolve_all_powers(&self, n: usize, max_idx: usize) -> HashMap<usize, Self>
    where
        Self: Sized;
}

/// Convolve all distributions in an iterator.
pub trait ConvolveAll<T>: Iterator<Item = T>
where
    T: Convolve + Clone + Sync + Send,
{
    /// Convolves all elements in this iterator together up to `max_idx`.
    ///
    /// # Arguments
    /// * `max_idx` - Upper bound for indices
    ///
    /// # Returns
    /// The resulting convolved distribution, or None if iterator is empty
    fn convolve_all(self, max_idx: usize) -> Option<T>;
}

impl Convolve for F64Log {
    fn convolve(&self, other: &Self, _max_idx: usize) -> Self {
        *self * *other
    }
    fn convolve_opt(&self, other: &Self, max_idx: usize) -> Option<Self> {
        Some(self.convolve(other, max_idx))
    }
}

impl Convolve for Vec<F64Log> {
    fn convolve(&self, other: &Self, max_idx: usize) -> Self {
        let result_len = (self.len() + other.len())
            .saturating_sub(1)
            .min(max_idx + 1);
        let mut result = vec![F64Log::new(0.0); result_len];

        for i in 0..result_len {
            let start = i.saturating_sub(other.len() - 1);
            result[i] = self
                .iter()
                .enumerate()
                .skip(start)
                .take(i - start + 1)
                .map(|(i0, &s)| s * other[i - i0])
                .sum();
        }
        result
    }

    fn convolve_opt(&self, other: &Self, max_idx: usize) -> Option<Self> {
        if self.is_empty() || other.is_empty() {
            return None;
        }
        Some(self.convolve(other, max_idx))
    }
}

impl<T> Convolve for OffsetVec<T>
where
    Vec<T>: Convolve,
{
    fn convolve(&self, other: &Self, max_idx: usize) -> Self {
        if self.stride != other.stride {
            panic!("Cannot convolve two vectors with different multiplicity");
        }
        OffsetVec {
            offset: self.offset + other.offset,
            stride: self.stride,
            vec: if self.offset + other.offset <= max_idx {
                self.vec.convolve(
                    &other.vec,
                    (max_idx - (self.offset + other.offset)).div_ceil(self.stride),
                )
            } else {
                vec![]
            },
        }
    }
    fn convolve_opt(&self, other: &Self, max_idx: usize) -> Option<Self> {
        if self.vec.is_empty() || other.vec.is_empty() {
            return None;
        }
        Some(self.convolve(other, max_idx))
    }
}

impl<T: Addable + Convolve + Sync + Send> Convolve for HashMap<usize, T> {
    fn convolve(&self, other: &Self, max_idx: usize) -> Self {
        let mut result = Self::default();

        let convolution_results: Vec<_> = self
            .par_iter()
            .flat_map(|(&i, v_val)| {
                other
                    .par_iter()
                    .filter_map(move |(&j, w_val)| {
                        v_val
                            .convolve_opt(w_val, max_idx)
                            .map(|convolved_val| (i + j, convolved_val))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, val) in convolution_results {
            result
                .entry(idx)
                .and_modify(|existing_val| {
                    *existing_val = existing_val.add(&val);
                })
                .or_insert_with(|| val);
        }

        result
    }
    fn convolve_opt(&self, other: &Self, max_idx: usize) -> Option<Self> {
        if self.is_empty() || other.is_empty() {
            return None;
        }
        Some(self.convolve(other, max_idx))
    }
}

impl ConvolveOne for F64Log {
    fn convolve_one(&self, other: &Self, _idx: &[usize]) -> F64Log {
        *self * *other
    }
}

impl<T: Clone> ConvolveOne for Vec<T>
where
    T: ConvolveOne,
{
    fn convolve_one(&self, other: &Self, idx: &[usize]) -> F64Log {
        if idx.is_empty() {
            return F64Log::default();
        }
        let i = idx[0];
        (0..=i)
            .filter_map(|i0| {
                let i1 = i - i0;
                let v0 = self.get(i0);
                let v1 = other.get(i1);
                match (v0, v1) {
                    (Some(s), Some(o)) => Some(s.convolve_one(o, &idx[1..])),
                    _ => None,
                }
            })
            .sum()
    }
}

impl<T: Clone> ConvolveOne for OffsetVec<T>
where
    Vec<T>: ConvolveOne,
{
    fn convolve_one(&self, other: &Self, idx: &[usize]) -> F64Log {
        if self.stride != other.stride {
            panic!("Cannot convolve two vectors with different multiplicity");
        }
        let rel_idx = match idx[0].checked_sub(self.offset + other.offset) {
            Some(i) => i,
            None => return F64Log::default(),
        };
        if rel_idx % self.stride != 0 {
            return F64Log::default();
        }
        let vec_idx = rel_idx / self.stride;
        self.vec.convolve_one(
            &other.vec,
            &[vec_idx]
                .iter()
                .chain(&idx[1..])
                .copied()
                .collect::<Vec<_>>(),
        )
    }
}

impl<V: Clone> ConvolveOne for HashMap<usize, V>
where
    V: ConvolveOne,
{
    fn convolve_one(&self, other: &Self, idx: &[usize]) -> F64Log {
        if idx.is_empty() {
            return F64Log::default();
        }
        let i = idx[0];
        self.iter()
            .filter_map(|(&k0, v0)| match i.checked_sub(k0) {
                Some(i1) => other.get(&i1).map(|v1| v0.convolve_one(v1, &idx[1..])),
                None => None,
            })
            .sum()
    }
}

impl ConvolveSelf for Vec<F64Log> {
    fn convolve_self(&self, max_idx: usize) -> Self {
        let result_len = (2 * self.len()).saturating_sub(1).min(max_idx + 1);
        let mut result = vec![F64Log::new(0.0); result_len];

        for i in 0..result_len {
            let start = i.saturating_sub(self.len() - 1);
            result[i] = self
                .iter()
                .enumerate()
                .skip(start)
                .take(i / 2 - start + 1)
                .map(|(i0, &s)| {
                    if 2 * i0 == i {
                        s * s
                    } else {
                        (s * self[i - i0]) * 2.
                    }
                })
                .sum();
        }
        result
    }

    fn convolve_self_opt(&self, max_idx: usize) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        Some(self.convolve_self(max_idx))
    }
}

impl<T> ConvolveSelf for OffsetVec<T>
where
    Vec<T>: ConvolveSelf,
{
    fn convolve_self(&self, max_idx: usize) -> Self {
        OffsetVec {
            offset: self.offset * 2,
            stride: self.stride,
            vec: if self.offset * 2 <= max_idx {
                self.vec
                    .convolve_self((max_idx - (self.offset * 2)).div_ceil(self.stride))
            } else {
                vec![]
            },
        }
    }

    fn convolve_self_opt(&self, max_idx: usize) -> Option<Self> {
        if self.vec.is_empty() {
            return None;
        }
        Some(self.convolve_self(max_idx))
    }
}

impl<T> ConvolveLeft for HashMap<usize, T>
where
    T: Addable + Convolve,
{
    fn convolve_left(&self, other: &Self, idx: usize, max_idx: usize) -> Self {
        let mut result = Self::default();

        for (&i, v_val) in self.iter() {
            if i <= idx {
                if let Some(w_val) = other.get(&(idx - i)) {
                    if let Some(convolved_val) = v_val.convolve_opt(w_val, max_idx) {
                        result
                            .entry(i)
                            .and_modify(|existing_val| {
                                *existing_val = existing_val.add(&convolved_val);
                            })
                            .or_insert_with(|| convolved_val);
                    }
                }
            }
        }

        result
    }
}

/// Helper structure for efficient computation of convolution powers.
///
/// Used to compute all convolution powers in range [1..max] by building an
/// optimized computation tree. Each power n is computed using previous powers
/// via addition or doubling operations. The number of operations needed to
/// compute any power is bounded by the height of the computation tree O(log n).
struct PowerTree {
    tree: HashMap<usize, Option<usize>>,
    level: Vec<usize>,
    max: usize,
}

impl PowerTree {
    /// Creates a new power tree for computing powers up to `max`.
    pub fn new(max: usize) -> Self {
        let mut tree = HashMap::new();
        tree.insert(1, None);
        Self {
            tree,
            level: vec![1],
            max,
        }
    }
}

/// Operations in the  power tree.
enum Operation {
    Double(usize),
    Add(usize, usize),
}

impl Iterator for PowerTree {
    /// Iterates through levels of the power tree, returning the operations
    /// performed at each level. Each iteration yields a Vec of operations
    /// (either doubling or adding) needed to compute new powers in that level
    /// from previously computed powers.
    type Item = Vec<Operation>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut results = Vec::new();
        let mut next_level = Vec::new();

        for &x in &self.level {
            let chain: Vec<usize> = std::iter::successors(Some(x), |&n| {
                (n != 1).then(|| self.tree[&n].expect("tree value must exist"))
            })
            .collect();

            for &y in chain.iter().rev() {
                let sum = y + x;
                if sum <= self.max && !self.tree.contains_key(&sum) {
                    self.tree.insert(sum, Some(x));
                    next_level.push(sum);
                    results.push(match y == x {
                        true => Operation::Double(y),
                        false => Operation::Add(y, x),
                    });
                }
            }
        }

        if next_level.is_empty() {
            None
        } else {
            self.level = next_level;
            Some(results)
        }
    }
}

impl ConvolveAllPowers for OffsetVec<F64Log> {
    fn convolve_all_powers(&self, n: usize, max_idx: usize) -> HashMap<usize, Self> {
        let mut powers = HashMap::new();

        // Base cases
        powers.insert(
            0,
            OffsetVec {
                offset: 0,
                stride: self.stride,
                vec: vec![F64Log::new(1.0)],
            },
        );
        powers.insert(1, self.clone());

        let tree = PowerTree::new(n);
        for level_ops in tree {
            // Process operations in this level in parallel
            let results: Vec<_> = level_ops
                .par_iter()
                .filter_map(|op| match op {
                    Operation::Double(x) => powers.get(x).cloned().and_then(|power| {
                        power
                            .convolve_self_opt(max_idx)
                            .map(|result| (2 * x, result))
                    }),
                    Operation::Add(x, y) => {
                        let power_x = powers.get(x);
                        let power_y = powers.get(y);
                        match (power_x, power_y) {
                            (Some(px), Some(py)) => {
                                px.convolve_opt(py, max_idx).map(|result| (x + y, result))
                            }
                            _ => None,
                        }
                    }
                })
                .collect();

            // Insert results back into powers
            for (k, v) in results {
                powers.insert(k, v);
            }
        }

        powers
    }
}

impl<T, I> ConvolveAll<T> for I
where
    T: Convolve + Clone + Sync + Send,
    I: Iterator<Item = T> + Send,
{
    fn convolve_all(self, max_idx: usize) -> Option<T> {
        let elements: Vec<_> = self.collect();

        if elements.is_empty() {
            return None;
        }

        if elements.len() == 1 {
            return Some(elements.into_iter().next().unwrap());
        }

        convolve_recursive(&elements, max_idx)
    }
}

/// Recursive helper function for divide-and-conquer convolution
fn convolve_recursive<T>(elements: &[T], max_idx: usize) -> Option<T>
where
    T: Convolve + Clone,
{
    match elements.len() {
        0 => None,
        1 => Some(elements[0].clone()),
        n => {
            let mid = n / 2;
            let (left, right) = elements.split_at(mid);

            // Process recursively without intermediate collections
            match (
                convolve_recursive(left, max_idx),
                convolve_recursive(right, max_idx),
            ) {
                (Some(l), Some(r)) => l.convolve_opt(&r, max_idx),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }
        }
    }
}

/// Add vectors element-wise.
pub trait Addable {
    /// Performs element-wise addition of two distributions, padding with zeros
    /// as needed. For offset vectors, properly aligns elements based on
    /// their offsets before adding.
    ///
    /// # Arguments
    /// * `other` - Distribution to add element-wise
    ///
    /// # Returns
    /// The sum of the distributions with appropriate padding and offset
    /// handling
    fn add(&self, other: &Self) -> Self;
}

impl Addable for Vec<F64Log> {
    fn add(&self, other: &Self) -> Self {
        self.iter()
            .zip_longest(other.iter())
            .map(|either| match either {
                itertools::EitherOrBoth::Both(a, b) => *a + *b,
                itertools::EitherOrBoth::Left(a) => *a,
                itertools::EitherOrBoth::Right(b) => *b,
            })
            .collect()
    }
}

impl Addable for OffsetVec<F64Log> {
    fn add(&self, other: &Self) -> Self {
        if self.stride != other.stride {
            panic!("Cannot add two vectors with different multiplicity");
        }
        let (far, near, offset_diff, offset_sum) = if self.offset > other.offset {
            (
                &self.vec,
                &other.vec,
                (self.offset - other.offset).div_ceil(self.stride),
                other.offset,
            )
        } else {
            (
                &other.vec,
                &self.vec,
                (other.offset - self.offset).div_ceil(self.stride),
                self.offset,
            )
        };

        let padded_far = std::iter::repeat_n(F64Log::new(0.0), offset_diff)
            .chain(far.iter().copied())
            .collect::<Vec<_>>();

        OffsetVec {
            offset: offset_sum,
            stride: self.stride,
            vec: near.add(&padded_far),
        }
    }
}

/// Simplify distributions by keeping only most significant values.
pub trait Simplify<A> {
    /// Simplify a distribution by keeping the `alpha` most significant
    /// realizations.
    fn simplify(&self, alpha: &A) -> Self;
}

impl Simplify<F64Log> for Vec<F64Log> {
    fn simplify(&self, alpha: &F64Log) -> Self {
        // Calculate the total sum and the target sum.
        let total_sum: F64Log = self.iter().copied().sum();
        let target_sum = total_sum * *alpha;

        let mut range: Option<(usize, usize)> = None;

        for start in 0..self.len() {
            let mut current_sum = F64Log::new(0.0);
            for (end, v) in self.iter().enumerate().skip(start) {
                current_sum += *v;
                if current_sum >= target_sum {
                    range = Some((start, end));
                    break;
                }
            }
            if range.is_some() {
                break;
            }
        }

        match range {
            Some((start, end)) => self[start..=end].to_vec(),
            _ => self.clone(),
        }
    }
}

impl Simplify<F64Log> for OffsetVec<F64Log> {
    fn simplify(&self, alpha: &F64Log) -> Self {
        // Calculate total sum and target sum
        let total_sum: F64Log = self.vec.iter().copied().sum();
        let target_sum = total_sum * *alpha;

        let mut range: Option<(usize, usize)> = None;

        for start in 0..self.vec.len() {
            let mut current_sum = F64Log::new(0.0);
            for (end, v) in self.vec.iter().enumerate().skip(start) {
                current_sum += *v;
                if current_sum >= target_sum {
                    range = Some((start, end));
                    break;
                }
            }
            if range.is_some() {
                break;
            }
        }

        match range {
            Some((start, end)) => OffsetVec {
                offset: self.offset + start * self.stride,
                stride: self.stride,
                vec: self.vec[start..=end].to_vec(),
            },
            _ => self.clone(),
        }
    }
}

impl Simplify<F64Log> for HashMap<usize, OffsetVec<F64Log>> {
    fn simplify(&self, alpha: &F64Log) -> Self {
        let mut sums: Vec<(usize, F64Log)> = self
            .iter()
            .map(|(&k, v)| (k, v.vec.iter().copied().sum()))
            .collect();
        sums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let total_sum: F64Log = sums.iter().copied().map(|(_, sum)| sum).sum();
        let target_sum = total_sum * *alpha;

        let mut current_sum = F64Log::new(0.0);
        let mut selected_keys = Vec::new();

        for (k, sum) in sums {
            current_sum += sum;
            selected_keys.push(k);
            if current_sum >= target_sum {
                break;
            }
        }

        self.iter()
            .filter(|&(&k, _)| selected_keys.contains(&k))
            .map(|(&k, v)| (k, v.clone()))
            .collect()
    }
}

/// Normalize distributions.
pub trait Normalize {
    /// Normalize a distribution by dividing all values by their sum.
    ///
    /// # Returns
    /// Some(normalized) if sum is non-zero, None if distribution is all zeros
    fn normalize(self) -> Option<Self>
    where
        Self: Sized;
}

impl Normalize for Vec<F64Log> {
    fn normalize(self) -> Option<Vec<F64Log>> {
        let sum: F64Log = self.iter().copied().sum();

        if sum.is_zero() {
            None
        } else {
            Some(self.into_iter().map(|val| val / sum).collect())
        }
    }
}

/// Removes zero values from distributions.
pub trait TrimZeros {
    /// Returns a distribution with leading and trailing zeros removed.
    fn trim_zeros(&self) -> Self;
}

impl TrimZeros for OffsetVec<F64Log> {
    fn trim_zeros(&self) -> Self {
        let first_nonzero = self.vec.iter().position(|x| !x.is_zero());
        let last_nonzero = self.vec.iter().rposition(|x| !x.is_zero());

        match (first_nonzero, last_nonzero) {
            (Some(start), Some(end)) => OffsetVec {
                offset: self.offset + start * self.stride,
                stride: self.stride,
                vec: self.vec[start..=end].to_vec(),
            },
            _ => OffsetVec {
                offset: self.offset,
                stride: self.stride,
                vec: vec![],
            },
        }
    }
}

/// Computes the scalar product between distributions.
pub trait ScalarProduct<S> {
    /// Calculates the scalar product of two distributions.
    ///
    /// # Arguments
    /// * `other` - Distribution to compute product with
    ///
    /// # Returns
    /// Some(product) or None if computation fails
    fn scalar_product(&self, other: &Self) -> Option<f64>;

    /// Calculates the top K products between distributions and returns
    /// associated states.
    ///
    /// # Arguments
    /// * `other` - Distribution to compute top products with
    /// * `K` - Number of top products to compute
    ///
    /// # Returns
    /// Result containing Vec of (product, state) tuples for top K products
    /// or error message if computation fails
    fn max_product<const K: usize>(&self, other: &Self) -> Vec<(f64, S)>;
}

impl<S> ScalarProduct<S> for HashMap<S, f64>
where
    S: Clone + Eq + Hash + Sync + std::fmt::Debug,
{
    fn scalar_product(&self, other: &Self) -> Option<f64> {
        self.par_iter()
            .try_fold(
                || 0.0,
                |acc, (key, &value_a)| {
                    if let Some(&value_b) = other.get(key) {
                        Some(acc + value_a * value_b)
                    } else {
                        println!("{:?}", key);
                        None
                    }
                },
            )
            .try_reduce(|| 0.0, |a, b| Some(a + b))
    }

    fn max_product<const K: usize>(&self, other: &Self) -> Vec<(f64, S)> {
        let mut top_k: Vec<(f64, S)> = Vec::with_capacity(K);

        for (key, &value_a) in self {
            if let Some(&value_b) = other.get(key) {
                let product = value_a * value_b;
                if top_k.len() < K {
                    top_k.push((product, key.clone()));
                    top_k.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                } else if product > top_k[K - 1].0 {
                    top_k[K - 1] = (product, key.clone());
                    top_k.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            } else if value_a > 1e-10 {
                println!("{:?} {}", key, value_a);
            }
        }

        top_k
    }
}

/// Truncate a distribution by setting values >= threshold to zero and
/// renormalizing.
pub trait TruncateDistribution {
    /// Truncate values >= threshold and renormalize.
    ///
    /// # Arguments
    /// * `threshold` - Truncation threshold
    ///
    /// # Returns
    /// Some(truncated) if remaining values sum to non-zero, None otherwise
    fn truncate_distr(&self, threshold: usize) -> Option<Self>
    where
        Self: Sized;
}

impl TruncateDistribution for Vec<F64Log> {
    fn truncate_distr(&self, threshold: usize) -> Option<Self> {
        self.iter()
            .enumerate()
            .map(|(i, &p)| if i <= threshold { p } else { F64Log::new(0.0) })
            .collect::<Vec<_>>()
            .normalize()
    }
}
