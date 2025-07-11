//! # QC-MDPC Markovian Model
//!
//! This program implements a Markovian model to calculate the Decoding Failure
//! Rate (DFR) of QC-MDPC codes using the step-by-step algorithm.
//! We consider a QC-MDPC code `(h0, h1) ∈ (𝔽₂[x](xʳ − 1))²` where `|h0| = |h1|
//! = d` for which we know that there exists near codewords:
//! * `(x^i * h0, 0)` for `i` in `{0, ..., r}`,
//! * `(0, x^i * h1)` for `i` in `{0, ..., r}`.
//!
//! These are of type `(d, d)`, producing error vectors of weight `d` with
//! syndrome weight `d`. Near codewords of type `(a, b)` with small `a` and `b`
//! are known to hinder decoder performance, causing the error floor phenomenon.
//!
//! This model accounts for these near codewords by using states `(s, t, u)`
//! where:
//! * `s` is the syndrome weight
//! * `t` is the error weight
//! * `u` is the number of common bits in the nearest near codeword
//!
//! By doing this, the model can predict an error floor.
//!
//! ## References
//!
//! > \[ALMPRTV25]: Sarah Arpin, Jun Bo Lau, Antoine Mesnard, Ray Perlner, Angela Robinson, Jean-Pierre Tillich & Valentin Vasseur: Error floor prediction with Markov models for QC-MDPC codes. <https://eprint.iacr.org/2025/153>
//!
//! > \[SV19]: Nicolas Sendrier & Valentin Vasseur: On the Decoding Failure Rate
//! > of QC-MDPC Bit-Flipping Decoders. <https://doi.org/10.1007/978-3-030-25510-7_22>
//!
//! ## Models
//!
//! ### Specific keys (§6 of \[ALMPRTV25])
//!
//! You can specify either a custom degree distribution or a code file using
//! `--degrees` or `--code`. The `--degrees` option expects a JSON string
//! representing the distribution of the degrees of the check nodes in the
//! subgraph generated by the near codeword (for each block).
//!
//! While you can provide either option, internally the program always converts
//! code files to degree distributions (this conversion happens automatically
//! with `--code`). You can also explicitly convert a code file to its degree
//! distribution using the `compute-degrees` command and then use that output
//! with `--degrees`.
//!
//! **Format**
//!
//! The `--degrees` option expect a JSON array containing two arrays (one for
//! each block), where each array contains objects with "degree" and "count"
//! fields representing the degree distribution. For example:
//! ```json
//! [[{"degree":0,"count":1577},{"degree":1,"count":17},{"degree":2,"count":122},{"degree":4,"count":7}],
//!  [{"degree":0,"count":1576},{"degree":1,"count":16},{"degree":2,"count":126},{"degree":4,"count":4},{"degree":5,"count":1}]]
//! ```
//!
//! The `--code` option expects a file containing the code description (one
//! element of the support per line, blocks separated by an empty line).
//!
//! #### Default: Perfect keys (Appendix D of \[ALMPRTV25])
//!
//! By default, this model considers "perfect" keys. These keys have distance
//! spectra for each block `h0`, `h1` with multiplicity at most 1. This implies
//! that the support of the blocks are simple cyclic difference sets.
//!
//! If you need to actually construct such a code, for example if you want to
//! confirm the results of the model by comparing it with simulation data, these
//! keys can be constructed using the Singer construction, yielding keys with `r
//! = q^2 + q + 1` and `d ≤ q + 1` for any prime power `q`.
//!
//! ### ST Model (\[SV19])
//!
//! You can use the simpler, earlier, model (ST model) by specifying the `--st`
//! option along with a xi parameter. This model uses states `(s, t)` instead of
//! `(s, t, u)`.
//!
//! ### Threshold Function
//!
//! The threshold function can be specified using the `--threshold` option as a
//! list of mathematical expressions. These expressions can use variables `s`
//! (syndrome weight) and `d` (column weight) and standard mathematical
//! functions. Non-integer values are automatically rounded by default.
//!
//! **Examples**
//! * Simple majority threshold: `--threshold '["(d+1)/2"]'`
//! * Floor of linear function: `--threshold '["floor(0.006016213884791455 * s +
//!   8.797325112097532)"]'`
//! * Ceiling of linear function: `--threshold '["ceil(0.006016213884791455 * s
//!   + 8.797325112097532)"]'`
//! * Multiple independent thresholds: `--threshold '["0.006016213884791455 * s
//!   + 10.797325112097532", "0.006016213884791455 * s + 8.797325112097532",
//!   "(d+1)/2"]'`
//!
//! ## Subcommands
//!
//! 1. `transitions`: Calculate absorbing probabilities for each state in the
//!    Markov chain.
//!
//!    Computes the probability of eventually reaching each absorbing state
//! (Success, Blocked, NearCodeword)    from every possible decoder state. This
//! is equivalent to computing the limit of A^n as n→∞, where A
//!    is the one-step transition matrix.
//!
//!    **Usage**
//!    ```text
//!    transitions \
//!        --r <block_length> \
//!        --d <block_weight> \
//!        --pass <t_pass> \
//!        --fail <t_fail> \
//!        [--output <output_file>] \
//!        [--degrees <custom_degree_distribution> | --code <code_file>] \
//!        [--threshold <threshold_function>] \
//!        [--st <xi_factor>]
//!    ```
//!    Output: zstd compressed binary file containing absorbing probabilities
//!    for each state.
//!
//! 2. `initial-states`: Calculate initial state distribution for a given error
//!    weight.
//!
//!    **Usage**
//!    ```text
//!    initial-states \
//!        --r <block_length> \
//!        --d <block_weight> \
//!        --t <error_weight> \
//!        [--output <output_file>] \
//!        [--degrees <custom_degree_distribution> | --code <code_file>] \
//!        [--st <xi_factor>]
//!    ```
//!    Output: zstd compressed binary file containing initial state
//! probabilities.
//!
//! 3. `dfr`: Combine initial state distribution with absorbing probabilities to
//!    compute the final DFR.
//!
//!    **Usage**
//!    ```text
//!    dfr \
//!        --transitions <transitions_file> \
//!        --initial-states <initial_states_file> \
//!        [--output <output_file>] \
//!        [--st <xi_factor>]
//!    ```
//! Output: JSON file containing detailed DFR analysis:
//! - For ST model: `{"dfr_blocked": 0.001234}`
//! - For STUB model: `{"dfr_ncw": 0.001, "dfr_blocked": 0.002, "by_u": [...]}`
//!   where `by_u` contains detailed breakdown by u parameter values with
//!   conditional and absolute probabilities for each u value.
//!
//! 4. `compute-degrees`: Compute the degree distribution of near codewords in
//!    the Tanner graph.
//!
//!    Usage:
//!    ```text
//!    compute-degrees \
//!        --r <block_length> \
//!        --code <code_file>
//!    ```
//! Output: JSON string containing the degree distribution
//!
//! ## Example Workflow
//!
//! A typical session using this model involves the following steps:
//! (Using `r = 1723` as an example)
//!
//! 1. Compute absorbing probabilities: ```sh cargo run --release -- transitions
//!    --r 1723 --d 17 --pass 1 --fail 200 --code code.txt --output
//!    transitions.json.zstd ```
//!
//! 2. Compute initial state distribution: ```sh cargo run --release --
//!    initial-states --r 1723 --d 17 --t 55 --code code.txt --output
//!    initial_states.json.zstd ```
//!
//! 3. Compute final DFR: ```sh cargo run --release -- dfr --transitions
//!    transitions.json.zstd --initial-states initial_states.json.zstd --output
//!    final_dfr.json ```

/// Alpha parameter used in distribution trimming
///
/// We keep the `(1 - 10^-ALPHA)` most frequent values in intermediate
/// distributions. Higher values mean more precision but slower computation.
pub const ALPHA: i32 = 4;

/// Threshold for exact/approximate factorial calculation
///
/// Values below this use exact calculation, above use Stirling's approximation.
pub const EXACT_FACTORIAL_THRESHOLD: usize = 20;

/// Default compression level for zstd output files
pub const COMPRESSION_LEVEL: i32 = 3;

pub mod code;
pub mod dfr;
pub mod distribution;
pub mod errors;
pub mod f64log;
pub mod models;
pub mod serialize;
pub mod threshold;
