//! This module provides the basis models that other derived transition models
//! can build upon.
//!
//! The available base models track:
//!
//! - Syndrome weight (`s`) and error weight (`t`): A simple model
//! - STU Specific: Building on (`s`, `t`, `u`) for keys with known Tanner graph
//!   near codeword degree distributions.
mod st;
mod stu;
mod stu_specific;

pub use st::{CountersST, STBasic, STBasicCounterModel, STBasicInitialStateModel};
pub use stu::{CountersSTU, STUBasic};
pub use stu_specific::{STUBasicSpecificCounterModel, STUBasicSpecificInitialStateModel};
