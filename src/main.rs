use std::{
    collections::HashMap,
    fs::File,
    io::{self, Write},
};

use clap::{Parser, Subcommand};
use log::info;
use serde::{Deserialize, Serialize};
use zstd::{Decoder, Encoder};

use qcmdpc_markov_ncw::{
    COMPRESSION_LEVEL,
    code::{CodeDegrees, MDPCCode},
    dfr::compute_dfr,
    distribution::ScalarProduct,
    errors::{Error, Result},
    models::{
        STInitialStateModel, STModel, STState, STUBModel, STUBSpecificCounterModel,
        STUBSpecificInitialStateModel, STUBState, traits::InitialStateModel,
    },
    serialize::{stream_read_from_file, stream_write_to_file},
    threshold::Threshold,
};

/// JSON output structure for ST model merge results
#[derive(Serialize, Deserialize, Debug)]
struct STMergeResult {
    /// Overall decoding failure rate
    dfr_blocked: f64,
}

/// JSON output structure for detailed STUB model results by u value
#[derive(Serialize, Deserialize, Debug)]
struct STUBByUResult {
    /// u parameter value
    u: usize,
    /// Total probability mass for this u value
    probability: f64,
    /// Conditional probability of reaching near codeword given this u
    conditional_dfr_ncw: f64,
    /// Conditional probability of reaching blocked state given this u
    conditional_dfr_blocked: f64,
    /// Absolute probability of reaching near codeword for this u
    absolute_dfr_ncw: f64,
    /// Absolute probability of reaching blocked state for this u
    absolute_dfr_blocked: f64,
}

/// JSON output structure for STUB model merge results
#[derive(Serialize, Deserialize, Debug)]
struct STUBMergeResult {
    /// Overall near codeword decoding failure rate
    dfr_ncw: f64,
    /// Overall blocked state decoding failure rate
    dfr_blocked: f64,
    /// Detailed breakdown by u parameter values
    by_u: Vec<STUBByUResult>,
}

/// Command-line interface structure
#[derive(Parser, Debug)]
#[command(about = "Markovian model for QC-MDPC decoding using the step-by-step algorithm")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Subcommands for the CLI
#[derive(Subcommand, Debug)]
enum Commands {
    /// Calculate absorbing probabilities for each state in the Markov chain for
    /// QC-MDPC codes.
    ///
    /// Computes the probability of eventually reaching each absorbing state
    /// (Success, Blocked, NearCodeword) from every possible decoder state.
    /// This is equivalent to computing the limit of A^n as n→∞, where A
    /// is the one-step transition matrix. The computation is efficient because
    /// transitions always go to either a strictly smaller state (by
    /// syndrome weight) or an absorbing state, allowing states to be processed
    /// in order from highest to lowest syndrome weight.
    ///
    /// The probabilities are computed using a Markovian model tracking either
    /// (`s`, `t`) states or (`s`, `t`, `u`) states based on the model
    /// arguments. For (`s`, `t`) states, `s` is syndrome weight and `t` is
    /// error weight. For (`s`, `t`, `u`) states, `u` is additionally added to
    /// track overlap with the nearest near codeword. Outputs a zstd
    /// compressed binary file containing absorbing probabilities.
    ///
    /// Example:
    /// ```
    /// transitions --r 1723 --d 17 --pass 1 --fail 200 --code code.txt --output transitions.json.zstd
    /// ```
    Transitions {
        /// Block length of the QC-MDPC code (`r`)
        #[arg(short, long, help = "Block length r of the QC-MDPC code")]
        r: usize,

        /// Block weight of the QC-MDPC code (`d`)
        #[arg(short, long, help = "Block weight d where |h0| = |h1| = d")]
        d: usize,

        /// Lower error weight threshold below which the decoder is assumed to
        /// always succeed
        #[arg(
            short,
            long = "pass",
            help = "Decoder always succeeds when error weight is below this threshold"
        )]
        pass: usize,

        /// Upper error weight threshold above which the decoder is assumed to
        /// always fail
        #[arg(
            short,
            long = "fail",
            help = "Decoder always fails when error weight is above this threshold"
        )]
        fail: usize,

        /// Optional output file for the probabilities (zstd compressed)
        #[arg(short, long, help = "Output file path (zstd compressed format)")]
        output: Option<String>,

        /// JSON string containing custom degree distribution. See documentation
        /// for format.
        #[arg(
            short = 'g',
            long,
            group = "degree_input",
            conflicts_with = "code",
            help = "JSON string of degree distribution"
        )]
        degrees: Option<String>,

        /// Path to file containing code description (one element per line,
        /// blocks separated by newline)
        #[arg(
            long,
            group = "degree_input",
            conflicts_with = "degrees",
            help = "File containing code description"
        )]
        code: Option<String>,

        /// Configuration of threshold functions. Expressions can use variables
        /// s (syndrome weight) and d (block weight). Multiple
        /// independent threshold functions can be provided.
        ///
        /// Examples:
        /// - Simple majority: `["(d+1)/2"]`
        /// - Linear with floor: `["floor(0.006 * s + 8.797)"]`
        /// - Multiple thresholds: `["0.006 * s + 10.797", "0.006 * s + 8.797",
        ///   "(d+1)/2"]`
        #[arg(short = 'T', long, help = "JSON array of threshold expressions")]
        threshold: Option<String>,

        /// Use ST model instead of STUB model with specified xi factor (simpler
        /// but less precise)
        #[arg(long, help = "Use ST model with given xi factor")]
        st: Option<f64>,
    },
    /// Calculate initial state distribution for a given error weight.
    ///
    /// Computes the probability distribution over all possible initial decoder
    /// states for a specified error weight. This is used in conjunction
    /// with absorbing probabilities to calculate the final decoding failure
    /// rate.
    ///
    /// Example:
    /// ```
    /// initial-states --r 1723 --d 17 --t 55 --code code.txt --output initial_states.json.zstd
    /// ```
    InitialStates {
        /// Block length of the QC-MDPC code (`r`)
        #[arg(short, long, help = "Block length r of the QC-MDPC code")]
        r: usize,

        /// Block weight of the QC-MDPC code (`d`)
        #[arg(short, long, help = "Block weight d where |h0| = |h1| = d")]
        d: usize,

        /// Error weight (`t`)
        #[arg(short, long, help = "Error weight t")]
        t: usize,

        /// Optional output file for the probabilities (zstd compressed)
        #[arg(short, long, help = "Output file path (zstd compressed format)")]
        output: Option<String>,

        /// JSON string containing custom degree distribution. See documentation
        /// for format.
        #[arg(
            short = 'g',
            long,
            group = "degree_input",
            conflicts_with = "code",
            help = "JSON string of degree distribution"
        )]
        degrees: Option<String>,

        /// Path to file containing code description (one element per line,
        /// blocks separated by newline)
        #[arg(
            long,
            group = "degree_input",
            conflicts_with = "degrees",
            help = "File containing code description"
        )]
        code: Option<String>,

        /// Use ST model instead of STUB model (simpler but less precise)
        #[arg(long, help = "Use simpler ST model")]
        st: bool,
    },
    /// Compute final Decoding Failure Rate by combining initial states and
    /// absorbing probabilities.
    ///
    /// Takes previously computed initial state distributions and absorbing
    /// probabilities to calculate final Decoding Failure Rate and near
    /// codeword probability.
    ///
    /// Outputs detailed JSON with overall DFR values and breakdown by u
    /// parameter for STUB models.
    ///
    /// Example:
    /// ```
    /// dfr --transitions transitions.json.zstd --initial-states initial_states.json.zstd --output final_dfr.json
    /// ```
    Dfr {
        /// Input file with the absorbing probabilities (from
        /// compute-transitions)
        #[arg(
            short,
            long = "transitions",
            help = "Input absorbing probabilities file"
        )]
        transitions_file: String,

        /// Input file with the initial state distribution (from
        /// compute-initial-states)
        #[arg(
            short,
            long = "initial-states",
            help = "Input initial state distribution file"
        )]
        initial_states_file: String,

        /// Optional output file for the probabilities (JSON format)
        #[arg(short, long, help = "Output file path (JSON format)")]
        output: Option<String>,

        /// Use ST model instead of STUB model with specified xi factor
        #[arg(long, help = "Use ST model with given xi factor")]
        st: Option<f64>,
    },
    /// Compute the degree distribution of near codewords in the Tanner graph.
    ///
    /// Analyzes code file to determine degree distribution of near codewords.
    /// Outputs distribution in JSON format suitable for --degrees option.
    ///
    /// Example:
    /// ```
    /// compute-degrees --r 1723 --code code.txt
    /// ```
    ComputeDegrees {
        /// Block length of the QC-MDPC code (`r`)
        #[arg(short, long, help = "Block length r of the QC-MDPC code")]
        r: usize,

        /// Path to the file containing the code description (one element of the
        /// support per line, blocks separated by an empty line)
        #[arg(short, long, help = "File containing code description")]
        code: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::init();

    match cli.command {
        Commands::Transitions {
            r,
            d,
            pass,
            fail,
            output,
            degrees,
            code: code_file,
            threshold,
            st,
        } => {
            let n = 2 * r;
            let w = 2 * d;

            info!("Block length (r): {}", r);
            info!("Block weight (d): {}", d);
            info!("Code length (n): {}", n);
            info!("Row weight (w): {}", w);
            info!("Threshold pass (t_pass): {}", pass);
            info!("Threshold fail (t_fail): {}", fail);

            let threshold = match threshold {
                Some(threshold_str) => threshold_str.parse::<Threshold>()?,
                None => Threshold::default(),
            };
            info!("Threshold: {}", threshold);

            let mut writer: Box<dyn Write> = match output {
                Some(file_path) => {
                    info!("Output file: {}", file_path);
                    Box::new(
                        Encoder::new(File::create(&file_path)?, COMPRESSION_LEVEL)?.auto_finish(),
                    )
                }
                None => Box::new(io::stdout()),
            };

            let code = MDPCCode::new(r, n, d, w, 2)?;
            if let Some(xi) = st {
                info!("ST model with xi: {}", xi);
                let model = STModel::new(&code, xi, pass, fail);
                let dfr_iterator = compute_dfr(code, model, threshold);

                stream_write_to_file(
                    dfr_iterator
                        .flat_map(|dfr_map| dfr_map.into_iter())
                        .filter(|(state, _)| *state != STState::Success),
                    &mut writer,
                )?;
            } else {
                let code_degrees = if let Some(degrees_str) = degrees {
                    info!("Using degree distribution directly from argument");
                    degrees_str
                        .parse::<CodeDegrees>()
                        .map_err(|e| io::Error::other(e.to_string()))?
                } else if let Some(code_file) = code_file {
                    info!("Computing degree distribution directly from code");
                    let code_content = std::fs::read_to_string(code_file)?;
                    let mut code_blocks = Vec::new();
                    let mut current_block = Vec::new();

                    for line in code_content.lines() {
                        if line.trim().is_empty() {
                            if !current_block.is_empty() {
                                code_blocks.push(current_block);
                                current_block = Vec::new();
                            }
                        } else if let Ok(num) = line.trim().parse() {
                            current_block.push(num);
                        }
                    }

                    if !current_block.is_empty() {
                        code_blocks.push(current_block);
                    }

                    CodeDegrees::compute_from_code(r, &code_blocks)
                } else {
                    info!("Using degree distribution of a \"perfect\" key");
                    let degree_maps = (0..code.index)
                        .map(|_| {
                            [
                                (0, r - d * (d + 1) / 2),
                                (1, d),
                                (2, d * d.saturating_sub(1) / 2),
                            ]
                            .into_iter()
                            .collect()
                        })
                        .collect();
                    CodeDegrees::new(&code, degree_maps)?
                };
                info!("Using degree distribution: {:?}", code_degrees);

                let counter_model = STUBSpecificCounterModel::new(&code, code_degrees);
                let model = STUBModel::new(&code, counter_model, pass, fail);
                let dfr_iterator = compute_dfr(code, model, threshold);

                stream_write_to_file(
                    dfr_iterator
                        .flat_map(|dfr_map| dfr_map.into_iter())
                        .filter(|(state, _)| *state != STUBState::Success),
                    &mut writer,
                )?;
            }
        }
        Commands::InitialStates {
            r,
            d,
            t,
            output,
            degrees,
            code: code_file,
            st,
        } => {
            let n = 2 * r;
            let w = 2 * d;

            info!("Block length (r): {}", r);
            info!("Block weight (d): {}", d);
            info!("Code length (n): {}", n);
            info!("Row weight (w): {}", w);
            info!("Error weight (t): {}", t);

            let code = MDPCCode::new(r, n, d, w, 2)?;
            let mut writer: Box<dyn Write> = match output {
                Some(file_path) => {
                    info!("Output file: {}", file_path);
                    Box::new(
                        Encoder::new(File::create(&file_path)?, COMPRESSION_LEVEL)?.auto_finish(),
                    )
                }
                None => Box::new(io::stdout()),
            };

            if st {
                info!("ST model");
                let model = STInitialStateModel::new(&code);
                let s_dist = model.get_initial_distribution(t);

                let mut sorted_states: Vec<_> = s_dist
                    .into_iter()
                    .filter_map(|(k, v)| {
                        if !v.is_zero() {
                            Some((k, v.as_f64()))
                        } else {
                            None
                        }
                    })
                    .collect();
                sorted_states.sort_by_key(|(k, _)| *k);

                stream_write_to_file(sorted_states.into_iter(), &mut writer)?;
            } else {
                let code_degrees = if let Some(degrees_str) = degrees {
                    info!("Using degree distribution directly from argument");
                    degrees_str
                        .parse::<CodeDegrees>()
                        .map_err(|e| io::Error::other(e.to_string()))?
                } else if let Some(code_file) = code_file {
                    info!("Computing degree distribution directly from code");
                    let code_content = std::fs::read_to_string(code_file)?;
                    let mut code_blocks = Vec::new();
                    let mut current_block = Vec::new();

                    for line in code_content.lines() {
                        if line.trim().is_empty() {
                            if !current_block.is_empty() {
                                code_blocks.push(current_block);
                                current_block = Vec::new();
                            }
                        } else if let Ok(num) = line.trim().parse() {
                            current_block.push(num);
                        }
                    }

                    if !current_block.is_empty() {
                        code_blocks.push(current_block);
                    }

                    CodeDegrees::compute_from_code(r, &code_blocks)
                } else {
                    info!("Using degree distribution of a \"perfect\" key");
                    let degree_maps = (0..code.index)
                        .map(|_| {
                            [
                                (0, r - d * (d + 1) / 2),
                                (1, d),
                                (2, d * d.saturating_sub(1) / 2),
                            ]
                            .into_iter()
                            .collect()
                        })
                        .collect();
                    CodeDegrees::new(&code, degree_maps)?
                };
                info!("Using degree distribution: {:?}", code_degrees);

                let s_dist = {
                    let model = STUBSpecificInitialStateModel::new(&code, code_degrees);
                    model.get_initial_distribution(t)
                };

                let mut sorted_states: Vec<_> = s_dist
                    .into_iter()
                    .filter_map(|(k, v)| {
                        if !v.is_zero() {
                            Some((k, v.as_f64()))
                        } else {
                            None
                        }
                    })
                    .collect();
                sorted_states.sort_by_key(|(k, _)| *k);

                stream_write_to_file(sorted_states.into_iter(), &mut writer)?;
            }
        }
        Commands::Dfr {
            transitions_file,
            initial_states_file,
            output,
            st,
        } => {
            info!("Absorbing probabilities file: {}", transitions_file);
            info!("Initial states file: {}", initial_states_file);

            if st.is_some() {
                info!("ST model");
                let dfr_reader = Decoder::new(File::open(&transitions_file)?)?;
                let syndrome_reader = Decoder::new(File::open(&initial_states_file)?)?;

                let dfr: HashMap<STState, f64> =
                    stream_read_from_file::<STState, HashMap<Option<STState>, f64>, _>(dfr_reader)
                        .filter_map(|result| {
                            result.ok().map(|(state, inner_map)| {
                                (
                                    state,
                                    inner_map
                                        .get(&Some(STState::Blocked))
                                        .copied()
                                        .unwrap_or(0.0),
                                )
                            })
                        })
                        .collect::<HashMap<_, _>>();
                let syndrome: HashMap<STState, f64> =
                    stream_read_from_file::<STState, f64, _>(syndrome_reader)
                        .map(|r| r.map_err(|e| e.into()))
                        .collect::<Result<HashMap<_, _>>>()?;

                let dfr_blocked = syndrome
                    .scalar_product(&dfr)
                    .ok_or_else(|| Error::model("Scalar product computation failed"))?;

                let result = STMergeResult { dfr_blocked };

                let mut writer: Box<dyn Write> = match output {
                    Some(file_path) => {
                        info!("Output file: {}", file_path);
                        Box::new(File::create(file_path)?)
                    }
                    None => Box::new(io::stdout()),
                };

                let json_output = serde_json::to_string_pretty(&result)
                    .map_err(|e| Error::parse(format!("JSON serialization failed: {}", e)))?;
                writeln!(writer, "{}", json_output)?;
            } else {
                info!("STUB model");
                let syndrome_reader = Decoder::new(File::open(&initial_states_file)?)?;
                let dfr_reader = Decoder::new(File::open(&transitions_file)?)?;

                let syndrome: HashMap<STUBState, f64> =
                    stream_read_from_file::<STUBState, f64, _>(syndrome_reader)
                        .map(|r| r.map_err(|e| e.into()))
                        .collect::<Result<HashMap<_, _>>>()?;

                let (dfr_ncw, dfr_blocked): (HashMap<STUBState, f64>, HashMap<STUBState, f64>) =
                    stream_read_from_file::<STUBState, HashMap<Option<STUBState>, f64>, _>(
                        dfr_reader,
                    )
                    .filter(|result| {
                        result
                            .as_ref()
                            .is_ok_and(|(state, _)| syndrome.contains_key(state))
                    })
                    .filter_map(|result| {
                        result.ok().map(|(state, inner_map)| {
                            (
                                state,
                                (
                                    inner_map
                                        .get(&Some(STUBState::NearCodeword))
                                        .copied()
                                        .unwrap_or(0.0),
                                    inner_map
                                        .get(&Some(STUBState::Blocked))
                                        .copied()
                                        .unwrap_or(0.0),
                                ),
                            )
                        })
                    })
                    .fold(
                        (HashMap::new(), HashMap::new()),
                        |(mut ncw, mut blocked), (state, (ncw_prob, blocked_prob))| {
                            ncw.insert(state, ncw_prob);
                            blocked.insert(state, blocked_prob);
                            (ncw, blocked)
                        },
                    );

                let overall_dfr_ncw = syndrome.scalar_product(&dfr_ncw).ok_or_else(|| {
                    Error::model("Near codeword scalar product computation failed")
                })?;
                let overall_dfr_blocked =
                    syndrome.scalar_product(&dfr_blocked).ok_or_else(|| {
                        Error::model("Blocked state scalar product computation failed")
                    })?;

                // Split syndrome distribution by u values
                let syndrome_by_u =
                    STUBSpecificInitialStateModel::split_initial_distribution(syndrome);

                let mut by_u_results = syndrome_by_u
                    .into_iter()
                    .filter(|&(u, _)| u != 0)
                    .map(|(u, syndrome_u)| {
                        let probability: f64 = syndrome_u.values().sum();
                        let absolute_dfr_ncw = syndrome_u
                            .scalar_product(&dfr_ncw)
                            .ok_or_else(|| Error::model("NCW scalar product failed"))?;
                        let absolute_dfr_blocked = syndrome_u
                            .scalar_product(&dfr_blocked)
                            .ok_or_else(|| Error::model("Blocked scalar product failed"))?;
                        let conditional_dfr_ncw = if probability > 0.0 {
                            absolute_dfr_ncw / probability
                        } else {
                            0.0
                        };
                        let conditional_dfr_blocked = if probability > 0.0 {
                            absolute_dfr_blocked / probability
                        } else {
                            0.0
                        };

                        Ok(STUBByUResult {
                            u,
                            probability,
                            conditional_dfr_ncw,
                            conditional_dfr_blocked,
                            absolute_dfr_ncw,
                            absolute_dfr_blocked,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                by_u_results.sort_by_key(|result| result.u);

                let result = STUBMergeResult {
                    dfr_ncw: overall_dfr_ncw,
                    dfr_blocked: overall_dfr_blocked,
                    by_u: by_u_results,
                };

                let mut writer: Box<dyn Write> = match output {
                    Some(file_path) => {
                        info!("Output file: {}", file_path);
                        Box::new(File::create(file_path)?)
                    }
                    None => Box::new(io::stdout()),
                };

                let json_output = serde_json::to_string_pretty(&result)
                    .map_err(|e| Error::parse(format!("JSON serialization failed: {}", e)))?;
                writeln!(writer, "{}", json_output)?;
            }
        }
        Commands::ComputeDegrees { r, code } => {
            let mut code_blocks = Vec::new();
            let mut current_block = Vec::new();
            let code_content = std::fs::read_to_string(code)?;

            for line in code_content.lines() {
                if line.trim().is_empty() {
                    if !current_block.is_empty() {
                        code_blocks.push(current_block);
                        current_block = Vec::new();
                    }
                } else if let Ok(num) = line.trim().parse() {
                    current_block.push(num);
                }
            }

            if !current_block.is_empty() {
                code_blocks.push(current_block);
            }

            let degrees = CodeDegrees::compute_from_code(r, &code_blocks);

            let json = serde_json::to_string(&degrees)
                .map_err(|e| Error::model(format!("Failed to serialize degrees: {}", e)))?;
            println!("{}", json);
        }
    }
    Ok(())
}
