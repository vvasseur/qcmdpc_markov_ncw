//! This module implements the simple threshold model which tracks only
//! the syndrome weight (`s`) and error weight (`t`).
use std::{cmp::Ordering, collections::HashMap};

use serde::{Deserialize, Serialize};

use crate::{
    code::MDPCCode,
    distribution::Normalize,
    errors::{Error, Result},
    f64log::F64Log,
    models::{
        basic::{CountersST, STBasic, STBasicCounterModel, STBasicInitialStateModel},
        traits::{Counter, CounterModel, InitialStateModel, State, TransitionModel},
    },
};

/// State of an MDPC code decoding process using syndrome weight (`s`) and error
/// weight (`t`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum STState {
    /// Valid state with syndrome weight `s` and error weight `t`.
    ST {
        s: usize,
        t: usize,
    },
    /// Blocked or failed decoding state.
    Blocked,
    Success,
}

impl PartialOrd for STState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for STState {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (STState::ST { s: s1, t: t1 }, STState::ST { s: s2, t: t2 }) => {
                s1.cmp(s2).then_with(|| t1.cmp(t2))
            }
            (STState::Blocked, _) => Ordering::Greater,
            (_, STState::Blocked) => Ordering::Less,
            (STState::Success, STState::Success) => Ordering::Equal,
            (STState::Success, _) => Ordering::Greater,
            (_, STState::Success) => Ordering::Less,
        }
    }
}

impl State for STState {
    type Basic = STBasic;

    fn decoded_state() -> Self {
        STState::ST { s: 0, t: 0 }
    }

    fn to_blocked(&self) -> Self {
        match self {
            STState::ST { .. } => STState::Blocked,
            STState::Blocked => STState::Blocked,
            STState::Success => STState::Blocked,
        }
    }

    fn s(&self) -> Result<usize> {
        match self {
            STState::ST { s, .. } => Ok(*s),
            STState::Blocked | STState::Success => Err(Error::model(
                "Cannot access syndrome weight on absorbing state",
            )),
        }
    }

    fn t(&self) -> Result<usize> {
        match self {
            STState::ST { t, .. } => Ok(*t),
            STState::Blocked | STState::Success => Err(Error::model(
                "Cannot access error weight on absorbing state",
            )),
        }
    }

    fn is_absorbing(&self) -> bool {
        matches!(self, STState::Success | STState::Blocked)
    }

    fn is_success(&self) -> bool {
        matches!(self, STState::Success)
    }
}

/// Simple model using syndrome weight (`s`) and error weight (`t`) states.
///
/// This model tracks the decoder state using only syndrome weight and error
/// weight.
pub struct STModel {
    code: MDPCCode,
    counter_model: STBasicCounterModel,
    t_pass: usize,
    t_fail: usize,
}

impl STModel {
    /// Creates a new ST model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `xi` - Xi parameter for the counter model
    /// * `t_pass` - Lower error weight threshold (decoder always succeeds below
    ///   this)
    /// * `t_fail` - Upper error weight threshold (decoder always fails above
    ///   this)
    ///
    /// # Returns
    /// A new `STModel` instance
    pub fn new(code: &MDPCCode, xi: f64, t_pass: usize, t_fail: usize) -> Self {
        STModel {
            code: code.clone(),
            counter_model: STBasicCounterModel::new(code, xi),
            t_pass,
            t_fail,
        }
    }

    fn get_counters_distribution(&self, state: &STState) -> Result<CountersST> {
        match state {
            STState::ST { s, t } => self
                .counter_model
                .get_counters_distribution(&STBasic { s: *s, t: *t }),
            _ => Err(Error::model(format!(
                "This state cannot have counters: {:?}",
                state
            ))),
        }
    }

    fn special_state(&self, state: &STState) -> Option<STState> {
        match state {
            STState::ST { s, t } => match (*t, *s) {
                (t, _) if t >= self.t_fail => Some(STState::Blocked),
                (t, _) if t < self.t_pass => Some(STState::Success),
                (t, s) if t <= self.code.d && s < t * (self.code.d - t + 1) => {
                    Some(STState::Success)
                }
                (t, 0) if t < 2 * self.code.d => Some(STState::Success),
                _ => None,
            },
            _ => None,
        }
    }

    fn condition_possible_counters(
        &self,
        counter: &CountersST,
        state: &STState,
    ) -> Result<CountersST> {
        match state {
            STState::ST { s, t } => {
                let CountersST { good, bad } = counter;

                let condition = |counters: &[F64Log], t_dest: usize| {
                    counters
                        .iter()
                        .enumerate()
                        .map(|(sigma, &prob)| {
                            let s_dest = (*s + self.code.d) as isize - 2 * sigma as isize;
                            let upper_bound = (self.code.d * t_dest) as isize;
                            let lower_bound = if t_dest == 1 {
                                self.code.d as isize
                            } else {
                                0_isize
                            };

                            if sigma > *s
                                || s_dest < 0
                                || s_dest > upper_bound
                                || s_dest < lower_bound
                            {
                                F64Log::new(0.)
                            } else {
                                prob
                            }
                        })
                        .collect::<Vec<_>>()
                };

                let new_good = condition(good, t + 1).normalize();
                let new_bad = condition(bad, t - 1).normalize();

                Ok(CountersST {
                    good: new_good.unwrap_or(vec![F64Log::new(0.); self.code.d + 1]),
                    bad: new_bad.unwrap_or(vec![F64Log::new(0.); self.code.d + 1]),
                })
            }
            _ => Err(Error::model(format!(
                "This state cannot have counters: {:?}",
                state
            ))),
        }
    }
}

impl TransitionModel<STState> for STModel {
    fn iter_all_states(&self) -> Vec<(usize, Vec<STState>)> {
        (0..=self.code.d * self.t_fail)
            .map(move |s| {
                (
                    s,
                    (self.t_pass..=self.t_fail)
                        .flat_map(move |t| {
                            if self.code.d * t % 2 != s % 2 || s > self.code.d * t {
                                vec![]
                            } else {
                                vec![STState::ST { s, t }]
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    fn transitions_from(
        &self,
        state: &STState,
        thresholds: Vec<usize>,
    ) -> Result<Vec<(STState, F64Log)>> {
        // Handle absorbing states
        if state.is_absorbing() {
            return Ok(vec![(*state, F64Log::new(1.0))]);
        }

        // Handle special states
        if let Some(special_state) = self.special_state(state) {
            return Ok(vec![(special_state, F64Log::new(1.0))]);
        }

        let threshold = thresholds[0];

        match *state {
            STState::ST { s, t } => {
                let counters = self.get_counters_distribution(state)?;
                let counters = self.condition_possible_counters(&counters, state)?;

                let mut transitions = Vec::new();

                let (p_lock, p_nonlock) = counters.lock(&self.code, STBasic { s, t }, threshold);
                let uniform_counters = counters
                    .uniform(&self.code, STBasic { s, t })
                    .map_err(|e| Error::model(format!("Counter uniform sampling failed: {}", e)))?;

                let (filtered_counters, p_flip) = uniform_counters.filter(threshold);
                if !p_flip.is_zero() {
                    let CountersST { good, bad } = filtered_counters;

                    let process_counter = |counters: &[F64Log], next_t: usize| {
                        counters
                            .iter()
                            .enumerate()
                            .filter_map(|(c, &proba_counter)| {
                                if !proba_counter.is_zero() {
                                    let next_s = s + self.code.d - 2 * c;
                                    let potential_state = STState::ST {
                                        s: next_s,
                                        t: next_t,
                                    };
                                    let next_state = self
                                        .special_state(&potential_state)
                                        .unwrap_or(potential_state);
                                    let probability = proba_counter * p_nonlock / p_flip;
                                    Some((next_state, probability))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    };

                    transitions.extend(process_counter(&good, t + 1));
                    if t > 0 {
                        transitions.extend(process_counter(&bad, t - 1));
                    }
                }

                transitions.push((STState::Blocked, p_lock));

                // Merge transitions to the same state
                Ok(transitions
                    .into_iter()
                    .fold(
                        HashMap::<STState, Vec<F64Log>>::new(),
                        |mut acc, (state, prob)| {
                            acc.entry(state).or_default().push(prob);
                            acc
                        },
                    )
                    .into_iter()
                    .map(|(state, probs)| (state, probs.iter().copied().sum()))
                    .collect())
            }
            _ => Ok(vec![(STState::Blocked, F64Log::new(1.0))]),
        }
    }
}

/// Initial state model for ST (syndrome weight, error weight) states.
///
/// This model provides the initial probability distribution of decoder states
/// for a given error weight, using the basic ST model as its foundation.
pub struct STInitialStateModel {
    basic_model: STBasicInitialStateModel,
}

impl STInitialStateModel {
    /// Creates a new ST initial state model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    ///
    /// # Returns
    /// A new `STInitialStateModel` instance
    pub fn new(code: &MDPCCode) -> Self {
        STInitialStateModel {
            basic_model: STBasicInitialStateModel::new(code),
        }
    }
}

impl InitialStateModel for STInitialStateModel {
    type State = STState;

    fn get_initial_distribution(&self, t: usize) -> HashMap<Self::State, F64Log> {
        self.basic_model
            .get_initial_distribution(t)
            .into_iter()
            .map(|(basic, prob)| {
                (
                    STState::ST {
                        s: basic.s,
                        t: basic.t,
                    },
                    prob,
                )
            })
            .collect()
    }
}
