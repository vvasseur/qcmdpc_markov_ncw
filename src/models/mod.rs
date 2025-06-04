//! Core of the computation in the Markov chain.
//!
//! We follow different approaches:
//!
//! - `STModel`: The simple model using only syndrome weight (`s`) and error
//!   weight (`t`) as state
//! - `STUBSpecificModel`: The refined model for specific keys, using (`s`, `t`,
//!   `u`) where `u` represents the number of common bits between the error
//!   pattern and the near codeword
pub mod basic;
mod st;
mod stub;
pub mod traits;

pub use st::{STInitialStateModel, STModel, STState};
pub use stub::{STUBModel, STUBSpecificCounterModel, STUBSpecificInitialStateModel, STUBState};
