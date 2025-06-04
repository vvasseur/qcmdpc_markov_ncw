//! This module implements a model that tracks syndrome weight (`s`), error
//! weight (`t`), and the number of shared bits with near codewords (`u`).
//!
//! This model require specifying degree distributions for the check nodes in
//! the Tanner graph subgraph induced by near codewords, with separate
//! distributions for each block.
use std::{cmp::Ordering, collections::HashMap};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::{
    code::{CodeDegrees, MDPCCode},
    distribution::Normalize,
    errors::{Error, Result},
    f64log::F64Log,
    models::basic::{CountersSTU, STUBasic},
    models::{
        basic::{STUBasicSpecificCounterModel, STUBasicSpecificInitialStateModel},
        traits::{Counter, CounterModel, InitialStateModel, State, TransitionModel},
    },
};

/// State of an MDPC code decoding process using syndrome weight (`s`), error
/// weight (`t`), and the number of common bits with the nearest near codeword
/// (`u`), along with a block index (`b`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum STUBState {
    /// Valid state with syndrome weight `s`, error weight `t`, and common bits
    /// `u`.
    STUB {
        s: usize,
        t: usize,
        u: usize,
        b: usize,
    },
    /// Blocked or failed decoding state.
    Blocked,
    NearCodeword,
    Success,
}

impl PartialOrd for STUBState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for STUBState {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (
                STUBState::STUB {
                    s: s1,
                    t: t1,
                    u: u1,
                    b: b1,
                },
                STUBState::STUB {
                    s: s2,
                    t: t2,
                    u: u2,
                    b: b2,
                },
            ) => s1
                .cmp(s2)
                .then_with(|| t1.cmp(t2))
                .then_with(|| u1.cmp(u2))
                .then_with(|| b1.cmp(b2)),
            (STUBState::Blocked, STUBState::Blocked) => Ordering::Equal,
            (STUBState::NearCodeword, STUBState::NearCodeword) => Ordering::Equal,
            (STUBState::Success, STUBState::Success) => Ordering::Equal,
            (STUBState::Blocked, _) => Ordering::Greater,
            (_, STUBState::Blocked) => Ordering::Less,
            (STUBState::NearCodeword, _) => Ordering::Greater,
            (_, STUBState::NearCodeword) => Ordering::Less,
            (STUBState::Success, _) => Ordering::Greater,
            (_, STUBState::Success) => Ordering::Less,
        }
    }
}

impl State for STUBState {
    type Basic = STUBasic;

    fn decoded_state() -> Self {
        STUBState::Success
    }

    fn to_blocked(&self) -> Self {
        match self {
            STUBState::STUB { .. } => STUBState::Blocked,
            STUBState::Blocked => STUBState::Blocked,
            STUBState::Success => STUBState::Blocked,
            STUBState::NearCodeword => STUBState::Blocked,
        }
    }

    fn s(&self) -> Result<usize> {
        match self {
            STUBState::STUB { s, .. } => Ok(*s),
            STUBState::Blocked | STUBState::NearCodeword | STUBState::Success => Err(Error::model(
                "Cannot access syndrome weight on absorbing state",
            )),
        }
    }

    fn t(&self) -> Result<usize> {
        match self {
            STUBState::STUB { t, .. } => Ok(*t),
            STUBState::Blocked | STUBState::NearCodeword | STUBState::Success => Err(Error::model(
                "Cannot access error weight on absorbing state",
            )),
        }
    }

    fn is_absorbing(&self) -> bool {
        matches!(
            self,
            STUBState::Success | STUBState::Blocked | STUBState::NearCodeword
        )
    }

    fn is_success(&self) -> bool {
        matches!(self, STUBState::Success)
    }
}

/// Counter model for STUB states
pub struct STUBSpecificCounterModel {
    counter_models: Vec<STUBasicSpecificCounterModel>,
}

impl STUBSpecificCounterModel {
    /// Creates a new STUB specific counter model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `degrees` - Code degrees containing degree distributions for each
    ///   block
    ///
    /// # Returns
    /// A new `STUBSpecificCounterModel` instance
    pub fn new(code: &MDPCCode, degrees: CodeDegrees) -> Self {
        STUBSpecificCounterModel {
            counter_models: degrees
                .degrees
                .iter()
                .map(|degs| STUBasicSpecificCounterModel::new(code, degs.clone()))
                .collect(),
        }
    }
}

impl CounterModel for STUBSpecificCounterModel {
    type State = STUBState;
    type BasicState = STUBasic;
    type Counter = CountersSTU;

    fn get_counters_distribution(&self, state: &STUBState) -> Result<CountersSTU> {
        match state {
            STUBState::STUB { s, t, u, b } => {
                self.counter_models[*b].get_counters_distribution(&STUBasic {
                    s: *s,
                    t: *t,
                    u: *u,
                })
            }
            _ => Err(Error::model(format!(
                "This state cannot have counters: {:?}",
                state
            ))),
        }
    }
}

/// STUB model using syndrome weight (`s`), error weight (`t`), shared bits
/// (`u`), and block (`b`) states.
///
/// This is the refined model that tracks the decoder state using syndrome
/// weight, error weight, the number of common bits with the nearest near
/// codeword, and a block index. It provides more precise predictions than the
/// ST model by accounting for near codewords that can cause decoding failures
/// (error floor phenomenon).
pub struct STUBModel<C: CounterModel<Counter = CountersSTU, BasicState = STUBasic>> {
    code: MDPCCode,
    counter_model: C,
    t_pass: usize,
    t_fail: usize,
}

impl<C: CounterModel<Counter = CountersSTU, State = STUBState, BasicState = STUBasic>>
    STUBModel<C>
{
    /// Creates a new STUB model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `counter_model` - Counter model implementing the `CounterModel` trait
    /// * `t_pass` - Lower error weight threshold (decoder always succeeds below
    ///   this)
    /// * `t_fail` - Upper error weight threshold (decoder always fails above
    ///   this)
    ///
    /// # Returns
    /// A new `STUBModel` instance
    pub fn new(code: &MDPCCode, counter_model: C, t_pass: usize, t_fail: usize) -> Self {
        STUBModel {
            code: code.clone(),
            counter_model,
            t_pass,
            t_fail,
        }
    }

    fn get_counters_distribution(&self, state: &STUBState) -> Result<CountersSTU> {
        match state {
            STUBState::STUB { .. } => self.counter_model.get_counters_distribution(state),
            _ => Err(Error::model(format!(
                "This state cannot have counters: {:?}",
                state
            ))),
        }
    }

    fn special_state(&self, state: &STUBState) -> Option<STUBState> {
        match state {
            STUBState::STUB { s, t, u, .. } => match (*s, *t, *u) {
                (s, t, u) if s == self.code.d && t == self.code.d && u == self.code.d => {
                    Some(STUBState::NearCodeword)
                }
                (_, t, _) if t >= self.t_fail => Some(STUBState::Blocked),
                (_, t, _) if t < self.t_pass => Some(STUBState::Success),
                (s, t, _) if t <= self.code.d && s < t * (self.code.d - t + 1) => {
                    Some(STUBState::Success)
                }
                (s, t, _) if s == 0 && t < 2 * self.code.d => Some(STUBState::Success),
                _ => None,
            },
            _ => None,
        }
    }

    fn condition_possible_counters(
        &self,
        counter: &CountersSTU,
        state: &STUBState,
    ) -> Result<CountersSTU> {
        match state {
            STUBState::STUB { s, t, u, .. } => {
                let CountersSTU {
                    good,
                    sus,
                    err,
                    bad,
                } = counter;

                let condition = |counters: &[F64Log], t_dest: usize, u_dest: usize| {
                    if t_dest != 0 && u_dest == 0 {
                        return vec![F64Log::new(0.); self.code.d + 1];
                    }

                    counters
                        .iter()
                        .enumerate()
                        .map(|(sigma, &prob)| {
                            let s_dest = (*s + self.code.d) as isize - 2 * sigma as isize;
                            let lower_bound = self.code.d as isize
                                * (2 * u_dest as isize - t_dest as isize)
                                - u_dest as isize * (u_dest as isize - 1);
                            let upper_bound = self.code.d as isize * t_dest as isize
                                - u_dest as isize * (u_dest as isize - 1);

                            if sigma > *s
                                || s_dest < 0
                                || s_dest < lower_bound
                                || s_dest > upper_bound
                            {
                                F64Log::new(0.)
                            } else {
                                prob
                            }
                        })
                        .collect::<Vec<_>>()
                };

                let new_good = condition(good, t + 1, *u).normalize();
                let new_sus = condition(sus, t + 1, u + 1).normalize();
                let new_err = if *t > 0 {
                    condition(err, t - 1, *u).normalize()
                } else {
                    None
                };
                let new_bad = if *t > 0 && *u > 0 {
                    condition(bad, t - 1, u - 1).normalize()
                } else {
                    None
                };

                Ok(CountersSTU {
                    good: new_good.unwrap_or(vec![F64Log::new(0.); self.code.d + 1]),
                    sus: new_sus.unwrap_or(vec![F64Log::new(0.); self.code.d + 1]),
                    err: new_err.unwrap_or(vec![F64Log::new(0.); self.code.d + 1]),
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

impl<C: CounterModel<Counter = CountersSTU, State = STUBState, BasicState = STUBasic> + Sync + Send>
    TransitionModel<STUBState> for STUBModel<C>
{
    fn iter_all_states(&self) -> Vec<(usize, Vec<STUBState>)> {
        (0..=self.code.d * self.t_fail)
            .map(move |s| {
                (
                    s,
                    (self.t_pass..=self.t_fail)
                        .flat_map(move |t| {
                            let mut states = vec![];
                            for u in 0..=std::cmp::min(t, self.code.d) {
                                for b in 0..self.code.index {
                                    if self.code.d * t % 2 != s % 2
                                        || s > self.code.d * t - u * u.saturating_sub(1)
                                    {
                                        continue;
                                    }
                                    if u == self.code.d && t == self.code.d && s != self.code.d {
                                        continue;
                                    }
                                    if t == 1 && s != self.code.d {
                                        continue;
                                    }
                                    states.push(STUBState::STUB { s, t, u, b });
                                }
                            }
                            states
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    fn transitions_from(
        &self,
        state: &STUBState,
        thresholds: Vec<usize>,
    ) -> Result<Vec<(STUBState, F64Log)>> {
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
            STUBState::STUB { s, t, u, b } => {
                let counters = self.get_counters_distribution(state)?;
                let counters = self.condition_possible_counters(&counters, state)?;

                let mut transitions = Vec::new();

                let (p_lock, p_nonlock) =
                    counters.lock(&self.code, STUBasic { s, t, u }, threshold);

                let uniform_counters = counters.uniform(&self.code, STUBasic { s, t, u })?;

                let (filtered_counters, p_flip) = uniform_counters.filter(threshold);
                if !p_flip.is_zero() {
                    let CountersSTU {
                        good,
                        sus,
                        err,
                        bad,
                    } = filtered_counters;

                    let process_counter = |counters: &[F64Log], next_t: usize, next_u: usize| {
                        counters
                            .iter()
                            .enumerate()
                            .filter_map(|(c, &proba_counter)| {
                                if !proba_counter.is_zero() {
                                    let next_s = s + self.code.d - 2 * c;
                                    let potential_state = STUBState::STUB {
                                        s: next_s,
                                        t: next_t,
                                        u: next_u,
                                        b,
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

                    transitions.extend(process_counter(&good, t + 1, u));
                    transitions.extend(process_counter(&sus, t + 1, u + 1));
                    if t > 0 {
                        transitions.extend(process_counter(&err, t - 1, u));
                    }
                    if t > 0 && u > 0 {
                        transitions.extend(process_counter(&bad, t - 1, u - 1));
                    }
                }

                transitions.push((STUBState::Blocked, p_lock));

                // Merge transitions to the same state
                Ok(transitions
                    .into_iter()
                    .fold(
                        HashMap::<STUBState, Vec<F64Log>>::new(),
                        |mut acc, (state, prob)| {
                            acc.entry(state).or_default().push(prob);
                            acc
                        },
                    )
                    .into_iter()
                    .map(|(state, probs)| (state, probs.iter().copied().sum()))
                    .collect())
            }
            _ => Ok(vec![(STUBState::Blocked, F64Log::new(1.0))]),
        }
    }
}

/// Initial states distribution model for STUB states
pub struct STUBSpecificInitialStateModel {
    code: MDPCCode,
    basic_models: Vec<STUBasicSpecificInitialStateModel>,
    hypergeom_cache: DashMap<(usize, usize, usize), Vec<F64Log>>,
}

impl STUBSpecificInitialStateModel {
    /// Creates a new STUB specific initial state model instance.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure
    /// * `degrees` - Code degrees containing degree distributions for each
    ///   block
    ///
    /// # Returns
    /// A new `STUBSpecificInitialStateModel` instance
    pub fn new(code: &MDPCCode, degrees: CodeDegrees) -> Self {
        STUBSpecificInitialStateModel {
            code: code.clone(),
            basic_models: degrees
                .degrees
                .iter()
                .map(|degs| STUBasicSpecificInitialStateModel::new(code, degs.clone()))
                .collect(),
            hypergeom_cache: DashMap::new(),
        }
    }

    /// Computes the rho values for given parameters
    ///
    /// Computes hypergeometric distribution with caching.
    ///
    /// # Arguments
    /// * `n` - Population size (code length)
    /// * `w` - Number of success states in population (row weight of
    ///   parity-check matrix)
    /// * `t` - Number of draws (error weight)
    ///
    /// # Returns
    /// Vector of hypergeometric probabilities
    pub fn hypergeom(&self, n: usize, w: usize, t: usize) -> Vec<F64Log> {
        let cache = &self.hypergeom_cache;
        cache
            .entry((n, w, t))
            .or_insert_with(|| F64Log::hypergeom_distribution(n, w, t))
            .clone()
    }

    /// Splits initial distribution by the value of u parameter
    ///
    /// # Arguments
    /// * `dist` - HashMap of states and their probabilities
    ///
    /// # Returns
    /// HashMap mapping u values to distributions of states with that u value
    pub fn split_initial_distribution(
        dist: HashMap<STUBState, f64>,
    ) -> HashMap<usize, HashMap<STUBState, f64>> {
        dist.iter()
            .fold(HashMap::new(), |mut acc, (&state, &proba)| {
                if let STUBState::STUB {
                    s: _,
                    t: _,
                    u,
                    b: _,
                } = state
                {
                    acc.entry(u).or_default().insert(state, proba);
                };
                acc
            })
    }
}

impl InitialStateModel for STUBSpecificInitialStateModel {
    type State = STUBState;

    fn get_initial_distribution(&self, t: usize) -> HashMap<Self::State, F64Log> {
        // Calculate the distribution of u (number of common bits with the nearest near
        // codeword)
        let u_distr: HashMap<(usize, usize), F64Log> = {
            // Compute probability distribution for t0
            let prob_t0: Vec<F64Log> = self.hypergeom(self.code.n, self.code.r, t);

            // Compute u0 for block 0 and u1 for block 1, then find max(u0,u1) and
            // argmax(u0,u1) We are only interested in the maximum value and the
            // block in which it occurs This gives us the joint distribution of
            // max(u0,u1) and argmax(u0,u1)
            let mut dist = HashMap::new();
            for u in 0..=self.code.d {
                for t0 in 0..=t {
                    let t1 = t - t0;
                    // Compute CDFs for the maximum of u0 and u1
                    let f0: Vec<F64Log> = F64Log::max_n_cdf(
                        &self.hypergeom(self.code.r, self.code.d, t0),
                        self.code.r,
                    );
                    let f1: Vec<F64Log> = F64Log::max_n_cdf(
                        &self.hypergeom(self.code.r, self.code.d, t1),
                        self.code.r,
                    );
                    // Compute CDFs for the maximum of u0 and u1
                    let p0: Vec<F64Log> = F64Log::max_n_pmf(
                        &self.hypergeom(self.code.r, self.code.d, t0),
                        self.code.r,
                    );
                    let p1: Vec<F64Log> = F64Log::max_n_pmf(
                        &self.hypergeom(self.code.r, self.code.d, t1),
                        self.code.r,
                    );

                    // Case 1: u0 = u and u1 < u
                    let case1_prob = if u > 0 {
                        p0.get(u).copied().unwrap_or_default()
                            * f1.get(u - 1).copied().unwrap_or_default()
                            * prob_t0.get(t0).copied().unwrap_or_default()
                    } else {
                        F64Log::new(0.)
                    };

                    // Case 2: u1 = u and u0 < u
                    let case2_prob = if u > 0 {
                        p1.get(u).copied().unwrap_or_default()
                            * f0.get(u - 1).copied().unwrap_or_default()
                            * prob_t0.get(t0).copied().unwrap_or_default()
                    } else {
                        F64Log::new(0.)
                    };

                    // Case 3: u0 = u1 = u
                    let case3_prob: F64Log = p0.get(u).copied().unwrap_or_default()
                        * p1.get(u).copied().unwrap_or_default()
                        * prob_t0.get(t0).copied().unwrap_or_default();

                    // Update distribution for argmax = 0 (u0 wins or tie)
                    *dist.entry((u, 0)).or_insert(F64Log::new(0.)) += case1_prob + case3_prob / 2.;

                    // Update distribution for argmax = 1 (u1 wins or tie)
                    *dist.entry((u, 1)).or_insert(F64Log::new(0.)) += case2_prob + case3_prob / 2.;
                }
            }
            dist
        };

        self.basic_models
            .iter()
            .enumerate()
            .flat_map(|(b, model)| {
                let u_distr = u_distr.clone();
                model.get_initial_distribution(t).into_iter().map(
                    move |(STUBasic { s, u, .. }, p_s)| {
                        let p_u = u_distr.get(&(u, b)).copied().unwrap_or_default();
                        (STUBState::STUB { s, t, u, b }, p_s * p_u)
                    },
                )
            })
            .collect()
    }
}
