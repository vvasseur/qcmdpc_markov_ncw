//! Computation of absorbing probabilities for Decoding Failure Rate (DFR)
//! analysis.
//!
//! This module computes the probability of eventually reaching each absorbing
//! state (Success, Blocked, NearCodeword) from every possible initial decoder
//! state. This is equivalent to computing the limit of `Aⁿ` as `n→∞`, where `A`
//! is the one-step transition matrix of the Markov chain.
//!
//! The computation follows an iterative process based on syndrome weight in
//! decreasing order. For each syndrome weight `s` from highest to lowest:
//! - generate all possible states with that syndrome weight,
//! - for each state, compute absorbing probabilities by combining one-step
//!   transitions with previously computed absorbing probabilities.
//!
//! This approach is efficient because transitions always go to either a state
//! with strictly lower syndrome weight (which was already processed) or
//! directly to an absorbing state. Since we only consider thresholds `≥
//! (d+1)/2`, flipping a position with counter `σ ≥ (d+1)/2` decreases syndrome
//! weight by `2σ - d > 0`.
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
};

use log::warn;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    code::MDPCCode,
    f64log::F64Log,
    models::traits::{State, TransitionModel},
    threshold::Threshold,
};

/// Iterator that computes and returns DFR (Decoding Failure Rate) values
/// incrementally.
///
/// This iterator approach allows streaming the DFR computations as they are
/// performed, rather than computing everything upfront. Only a small buffer of
/// intermediate results needs to be kept in memory.
///
/// Returns DFR values in ascending order of syndrome weight (`s`), starting
/// from `s=0`.
pub struct DfrIterator<S>
where
    S: State + Debug + Eq + Send + Sync,
{
    code: MDPCCode,
    current_s: usize,
    dfr_buffer: VecDeque<HashMap<S, HashMap<Option<S>, F64Log>>>,
    model: Box<dyn TransitionModel<S> + Send + Sync>,
    threshold: Threshold,
    states_iter: Box<dyn Iterator<Item = (usize, Vec<S>)>>,
}

impl<S> Iterator for DfrIterator<S>
where
    S: State + Debug + Eq + Send + Sync + 'static,
{
    type Item = HashMap<S, HashMap<Option<S>, f64>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_s > self.code.r {
            return None;
        }

        let (new_s, states) = self.states_iter.next()?;

        let local_dfrs: HashMap<S, HashMap<Option<S>, F64Log>> = states
            .into_par_iter()
            .map(|state| {
                let thresholds = self
                    .threshold
                    .evaluate(&self.code, &state)
                    .expect("Threshold evaluation failed");
                match self.model.transitions_from(&state, thresholds) {
                    Ok(transitions) => (state, transitions),
                    Err(_) => {
                        warn!("Transition computation failed for state {:?}", state);
                        (state, vec![])
                    }
                }
            })
            .fold(
                || HashMap::new(),
                |mut acc: HashMap<S, HashMap<Option<S>, F64Log>>, (state_orig, transitions)| {
                    for (state_dest, proba) in transitions {
                        let dfr_dest = if state_dest.is_absorbing() {
                            let mut map = HashMap::new();
                            map.insert(Some(state_dest.clone()), F64Log::new(1.0));
                            map
                        } else {
                            match state_dest.s() {
                                Ok(s_dest) => self
                                    .dfr_buffer
                                    .get(self.current_s - s_dest)
                                    .and_then(|map| map.get(&state_dest))
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        warn!(
                                            "Transition to unknown state {:?} -> {:?}",
                                            state_orig, state_dest
                                        );
                                        let mut map = HashMap::new();
                                        map.insert(None, F64Log::new(1.0));
                                        map
                                    }),
                                Err(e) => {
                                    warn!(
                                        "Failed to get syndrome weight for state {:?}: {}",
                                        state_dest, e
                                    );
                                    let mut map = HashMap::new();
                                    map.insert(None, F64Log::new(1.0));
                                    map
                                }
                            }
                        };

                        let entry = acc.entry(state_orig.clone()).or_default();
                        for (k, v) in &dfr_dest {
                            *entry.entry(k.clone()).or_insert(F64Log::new(0.0)) += proba * *v;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        if let Some(existing_v) = a.get_mut(&k) {
                            for (inner_k, inner_v) in v {
                                *existing_v.entry(inner_k).or_insert(F64Log::new(0.0)) += inner_v;
                            }
                        } else {
                            a.insert(k, v);
                        }
                    }
                    a
                },
            );

        // Update buffer with new DFRs
        let result = local_dfrs.clone();
        match (self.current_s == new_s, self.dfr_buffer.front_mut()) {
            (true, Some(front)) => front.extend(local_dfrs),
            _ => {
                if self.dfr_buffer.len() > self.code.d {
                    self.dfr_buffer.pop_back();
                }
                self.dfr_buffer.push_front(local_dfrs);
            }
        }
        self.current_s = new_s;

        Some(
            result
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().map(|(k2, v2)| (k2, v2.as_f64())).collect()))
                .collect(),
        )
    }
}

/// Compute absorbing probabilities for Decoding Failure Rate (DFR) analysis
///
/// This function computes the probability of eventually reaching each absorbing
/// state from every decoder state. It uses the recursive formula:
///
/// `AbsorbProb(State, Target) = Σ P(State -> State') * AbsorbProb(State',
/// Target)`
///
/// where `State` is the current state, `State'` are possible next states,
/// `Target` is an absorbing state (Blocked, Success, NearCodeword), and
/// `P(State -> State')` is the one-step transition probability.
///
/// For absorbing states: `AbsorbProb(Target, Target) = 1` and
/// `AbsorbProb(Target, Other) = 0`.
///
/// This computation is equivalent to finding the limit of A^n as n→∞, where A
/// is the transition matrix. The result gives the long-run probability of
/// absorption into each absorbing state.
///
/// Rather than computing only the probability of reaching `Blocked`, we compute
/// probabilities for all absorbing states, allowing separation of different
/// failure modes (blocked vs. near codeword convergence).
///
/// Note: Computation is performed in decreasing order of syndrome weight `s`.
/// This ensures that when computing absorbing probabilities for a state, all
/// states it can transition to have already been processed.
///
/// # Arguments
///
/// * `code` - The QC-MDPC code parameters
/// * `model` - Transition model
/// * `threshold` - Threshold function
///
/// # Returns
///
/// An iterator that yields absorbing probability maps for each syndrome weight
/// `s`, where each map contains:
/// - Keys: all possible states with the given syndrome weight `s`
/// - Values: hashmaps with:
///   - Keys: absorbing states (Some(state)) or None for unknown states
///   - Values: probability of eventually reaching that absorbing state from the
///     key state
pub fn compute_dfr<S>(
    code: MDPCCode,
    model: impl TransitionModel<S> + 'static,
    threshold: Threshold,
) -> DfrIterator<S>
where
    S: State + Debug + Eq + Send + Sync + 'static,
{
    let all_states = model.iter_all_states();
    DfrIterator {
        code: code.clone(),
        current_s: 0,
        dfr_buffer: VecDeque::with_capacity(code.d + 1),
        model: Box::new(model),
        threshold,
        states_iter: Box::new(all_states.into_iter()),
    }
}
