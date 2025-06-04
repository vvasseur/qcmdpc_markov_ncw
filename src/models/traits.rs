//! Traits for the Markov chain models.
use std::{collections::HashMap, hash::Hash};

use crate::{code::MDPCCode, errors::Result, f64log::F64Log};

/// Transitions between states in the Markov chain.
pub trait TransitionModel<S: State>: Send + Sync {
    /// Generates an iterator of possible states for t between t_low and t_high
    /// inclusive.
    fn iter_all_states(&self) -> Vec<(usize, Vec<S>)>;

    /// Computes the possible next states and their transition probabilities
    /// from a given state
    ///
    /// # Arguments
    /// * `state` - The current state to transition from
    /// * `threshold` - The threshold value for transitions
    ///
    /// # Returns
    /// A Result containing a vector of tuples with possible next states and
    /// their transition probabilities
    fn transitions_from(&self, state: &S, threshold: Vec<usize>) -> Result<Vec<(S, F64Log)>>;
}

/// Defines an interface for counter-based models in MDPC decoding.
/// Provides methods for calculating counter probability distributions
/// based on decoder states.
pub trait CounterModel {
    /// The state type associated with this model.
    type State: Clone;
    type BasicState: Clone;
    /// The counter type associated with this model.
    type Counter: Counter<Self::BasicState>;

    /// Gets the probability distribution of counters for a state.
    ///
    /// Calculates counter distributions needed for the decoding process
    /// at the current decoder state.
    ///
    /// # Arguments
    ///
    /// * `state` - A reference to the decoder state
    ///
    /// # Returns
    ///
    /// Counter probability distribution for the given state, or an error
    /// if the state cannot have counters.
    fn get_counters_distribution(&self, state: &Self::State) -> Result<Self::Counter>;
}

/// Defines an interface for initial states distribution in MDPC decoding.
/// Provides methods for calculating state probability distributions
/// based on error weights.
pub trait InitialStateModel {
    /// The state type associated with this model.
    type State: Clone;

    /// Gets the initial states probability distribution for a given error
    /// weight.
    ///
    /// # Arguments
    ///
    /// * `t` - Number of errors (error weight)
    ///
    /// # Returns
    ///
    /// Initial states probability distribution for the given error weight
    fn get_initial_distribution(&self, t: usize) -> HashMap<Self::State, F64Log>;
}

/// State in the MDPC decoding process.
///
/// This trait defines the common interface for different types of states used
/// in the MDPC code decoding algorithm. Implementations of this trait should
/// provide methods for calculating counters, determining decoded and blocked
/// states, and generating possible states from syndrome and error weights.
///
/// Every state must be able to convert to a basic representation (via
/// `to_basic()`) . This basic representation provides the essential information
/// needed for computation in the Markov chain.
pub trait State: Clone + Eq + Hash {
    type Basic: Clone;

    /// Returns the decoded state.
    fn decoded_state() -> Self;

    /// Returns a blocked state based on the given state.
    fn to_blocked(&self) -> Self;

    /// Returns the syndrome weight of the current state.
    ///
    /// # Errors
    /// Returns an error if called on a state that does not have a syndrome
    /// weight.
    fn s(&self) -> Result<usize>;

    /// Returns the error weight of the current state.
    ///
    /// # Errors
    /// Returns an error if called on a state that does not have an error
    /// weight.
    fn t(&self) -> Result<usize>;

    /// Returns whether this state is absorbing in the Markov chain.
    fn is_absorbing(&self) -> bool;

    /// Returns whether this state is a success in the Markov chain.
    fn is_success(&self) -> bool;
}

/// Counters associated with states in MDPC code decoding.
///
/// This trait defines methods for manipulating and analyzing counters that are
/// used in the decoding process of MDPC codes. It provides functionality for
/// conditioning counters, computing locking probabilities, applying sampling,
/// and computing transitions.
///
/// # Type Parameters
///
/// * `T`: A tuple type representing the state associated with these counters.
pub trait Counter<T: Clone>: Sync {
    /// Computes the "locking" probability for the given state and threshold.
    ///
    /// # Arguments
    /// * `code` - A reference to the MDPCCode.
    /// * `state` - The current state as a tuple.
    /// * `threshold` - The threshold value for decoding.
    ///
    /// # Returns
    /// The computed locking probability as a F64Log.
    fn lock(&self, code: &MDPCCode, state: T, threshold: usize) -> (F64Log, F64Log);

    /// Modifies the counters based on uniform sampling.
    ///
    /// # Arguments
    /// * `code` - A reference to the MDPCCode.
    /// * `state` - The current state as a tuple.
    ///
    /// # Returns
    /// A Result containing either the modified counters or an error message.
    fn uniform(&self, code: &MDPCCode, state: T) -> Result<Self>
    where
        Self: Sized;

    /// Computes transitions for the counters based on the given threshold.
    ///
    /// # Arguments
    /// * `threshold` - The threshold value for transitions.
    ///
    /// # Returns
    /// A tuple containing the new counters, flip probability, and no-flip
    /// probability.
    fn filter(&self, threshold: usize) -> (Self, F64Log)
    where
        Self: Sized;
}
