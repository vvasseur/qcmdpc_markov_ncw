//! Structures needed for working with QC-MDPC codes
//!
//! This module provides:
//! - Core MDPC code parameters including:
//!   - Block length (`r`)
//!   - Code length (`n`)
//!   - Column weight (`d`)
//!   - Row weight (`w`)
//! - Degree distributions of nodes in the subgraph `G` of the Tanner graph
//!   given by near-codewords of the form `xⁱh0(x)⊕0` and `0⊕xⁱh1(x)` (see §2.3
//!   of \[ALMPRTV25])
//!
//! ## Reference
//!
//! > \[ALMPRTV25]: Sarah Arpin, Jun Bo Lau, Antoine Mesnard, Ray Perlner, Angela Robinson, Jean-Pierre Tillich & Valentin Vasseur: Error floor prediction with Markov models for QC-MDPC codes. <https://eprint.iacr.org/2025/153>
use std::{collections::HashMap, fmt::Debug};

use crate::errors::{Error, Result};

/// Structure for an MDPC code.
#[derive(Clone, Debug)]
pub struct MDPCCode {
    /// Block length
    pub r: usize,
    /// Code length
    pub n: usize,
    /// Column weight of the parity-check matrix
    pub d: usize,
    /// Row weight of the parity-check matrix
    pub w: usize,
    /// Number of circulant blocks
    pub index: usize,
}

/// Represents the degree distribution of parity check nodes in the subgraph `G`
/// of the Tanner graph induced by near-codewords in a QC-MDPC code.
///
/// These near-codewords are of the form `ν=h0(x)⊕0` and `ν=0⊕h1(x)`.
///
/// # Notation
///
/// - `n_Δ`: number of parity-check equations in `G` of degree `Δ`
#[derive(Debug)]
pub struct CodeDegrees {
    /// Degree distribution of parity check nodes
    ///
    /// `degrees[i][Δ] = n_i,Δ`
    ///
    /// where `n_i,Δ` is the number of parity-check equations in `G` of degree
    /// `Δ` for block `i`
    pub degrees: Vec<BlockDegrees>,
}

/// Degree distribution of a single block
///
/// Use a simple map `{ Δ: n_Δ }`
/// where `n_Δ` is the number of parity-check equations in `G` of degree `Δ` for
/// the given block
pub type BlockDegrees = HashMap<usize, usize>;

impl MDPCCode {
    /// Creates a new `MDPCCode` instance with parameter validation
    ///
    /// # Arguments
    /// * `r` - Block length
    /// * `n` - Code length
    /// * `d` - Column weight of the parity-check matrix
    /// * `w` - Row weight of the parity-check matrix
    /// * `index` - Number of circulant blocks
    ///
    /// # Errors
    /// Returns `Error::Config` if parameters are invalid
    pub fn new(r: usize, n: usize, d: usize, w: usize, index: usize) -> Result<Self> {
        if d > r {
            return Err(Error::config(format!(
                "Block weight ({}) cannot exceed block length ({})",
                d, r
            )));
        }
        if w > n {
            return Err(Error::config(format!(
                "Row weight ({}) cannot exceed code length ({})",
                w, n
            )));
        }
        if index == 0 {
            return Err(Error::config("Number of circulant blocks must be positive"));
        }

        Ok(MDPCCode { r, n, d, w, index })
    }
}

impl CodeDegrees {
    /// Creates a new `CodeDegrees` instance with the given degree
    /// distributions.
    ///
    /// This function validates the degree distributions according to the
    /// constraints of the QC-MDPC code structure. It checks that:
    /// 1. the sum of all `count`s for odd degrees is `d`,
    /// 2. the sum of all `count`s is `r`,
    /// 3. the sum of all `degree * count` is `d²`.
    ///
    /// # Arguments
    /// * `code` - A reference to the `MDPCCode`.
    /// * `degrees` - An array of degree distributions of nodes in the subgraph
    ///   `G`.
    ///
    /// # Returns
    /// A `Result` containing either the validated `CodeDegrees` instance or an
    /// error.
    pub fn new(code: &MDPCCode, degrees: Vec<HashMap<usize, usize>>) -> Result<Self> {
        for (block_idx, block_degrees) in degrees.iter().enumerate() {
            let sum_odd: usize = block_degrees
                .iter()
                .filter(|&(&k, _)| k % 2 == 1)
                .map(|(_, &v)| v)
                .sum();
            if sum_odd != code.d {
                return Err(Error::config(format!(
                    "Degrees for block {}: sum of counts for odd degrees ({}) must equal d ({})",
                    block_idx, sum_odd, code.d
                )));
            }

            let sum_all: usize = block_degrees.values().sum();
            if sum_all != code.r {
                return Err(Error::config(format!(
                    "Degrees for block {}: sum of all counts ({}) must equal r ({})",
                    block_idx, sum_all, code.r
                )));
            }

            let sum_degree_count: usize = block_degrees.iter().map(|(&k, &v)| k * v).sum();
            if sum_degree_count != code.d.pow(2) {
                return Err(Error::config(format!(
                    "Degrees for block {}: sum of degree*count ({}) must equal d² ({})",
                    block_idx,
                    sum_degree_count,
                    code.d.pow(2)
                )));
            }
        }

        Ok(CodeDegrees { degrees })
    }

    /// Computes degree distributions from a specific code given by polynomials.
    ///
    /// # Arguments
    /// * `r` - Block length (modulus for polynomial arithmetic)
    /// * `blocks` - A slice of vectors, where each vector contains the
    ///   coordinates corresponding to the support of the polynomials `h_i` for
    ///   each block `i`.
    ///
    /// # Returns
    /// A `CodeDegrees` instance containing the computed degree distributions
    /// for all blocks.
    pub fn compute_from_code(r: usize, blocks: &[Vec<usize>]) -> CodeDegrees {
        let mut degree_distributions = Vec::new();

        for block in blocks {
            // Calculate the square (in Z[x]) of each block
            let square: HashMap<usize, usize> = block
                .iter()
                .flat_map(|&vj| block.iter().map(move |&vk| (vj + vk) % r))
                .fold(HashMap::new(), |mut map, sum| {
                    *map.entry(sum).or_insert(0) += 1;
                    map
                });

            // Each component of the square is a degree in the subgraph induced by the near
            // codeword of this block
            let mut histogram: HashMap<usize, usize> =
                square.values().fold(HashMap::new(), |mut map, &count| {
                    *map.entry(count).or_insert(0) += 1;
                    map
                });

            // Add count for degree 0
            if let Some(missing) = r.checked_sub(square.len()) {
                if missing > 0 {
                    histogram.insert(0, missing);
                }
            }

            degree_distributions.push(histogram);
        }

        CodeDegrees {
            degrees: degree_distributions,
        }
    }
}
