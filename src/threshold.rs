//! Threshold functions for MDPC decoding using *evalexpr*.
//!
//! Instead of implementing fixed threshold functions, this module uses
//! *evalexpr* to support arbitrary mathematical expressions that can use the
//! variables `s` (syndrome weight) and `d` (column weight) for dynamic
//! threshold calculation.
use evalexpr::{ContextWithMutableVariables, HashMapContext, Node, Value, build_operator_tree};

use crate::{
    code::MDPCCode,
    errors::{Error, Result},
    models::traits::State,
};

/// Mathematical expression that can be evaluated using variables `s` (syndrome
/// weight) and `d` (column weight)
#[derive(Debug)]
pub struct Expression {
    compiled_expr: Node,
    expression_str: String,
}

/// Collection of threshold expressions for MDPC decoding
#[derive(Debug)]
pub struct Threshold(Vec<Expression>);

impl Expression {
    /// Creates a new threshold from the given mathematical expression string
    /// that can use `s` and `d` variables
    ///
    /// # Arguments
    /// * `expression` - Mathematical expression string that can use `s`
    ///   (syndrome weight) and `d` (column weight)
    ///
    /// # Returns
    /// A `Result` containing the `Expression` or an error if parsing fails
    pub fn new(expression: &str) -> Result<Self> {
        build_operator_tree(expression)
            .map(|compiled_expr| Self {
                compiled_expr,
                expression_str: expression.to_string(),
            })
            .map_err(|e| {
                Error::parse(format!(
                    "Failed to parse threshold expression '{}': {}",
                    expression, e
                ))
            })
    }

    /// Evaluates the expression with the given code parameters and state.
    ///
    /// # Arguments
    /// * `code` - Reference to the QC-MDPC code structure containing parameters
    /// * `state` - Current decoder state
    ///
    /// # Returns
    /// The evaluated threshold value as a `usize`, clamped between `(d+1)/2`
    /// and `d`
    ///
    /// # Errors
    /// Returns `Error::Model` if expression evaluation fails
    pub fn evaluate<S: State>(&self, code: &MDPCCode, state: &S) -> Result<usize> {
        let mut context = HashMapContext::new();
        let s = state.s()?;
        context
            .set_value("s".into(), Value::Float(s as f64))
            .map_err(|e| Error::model(format!("Failed to set syndrome weight variable: {}", e)))?;
        context
            .set_value("d".into(), Value::Float(code.d as f64))
            .map_err(|e| Error::model(format!("Failed to set block weight variable: {}", e)))?;

        let result = self
            .compiled_expr
            .eval_with_context(&context)
            .and_then(|v| v.as_number())
            .map(|n| n.round() as usize)
            .map(|t| t.clamp(code.d.div_ceil(2), code.d))
            .map_err(|e| {
                Error::model(format!(
                    "Failed to evaluate threshold expression '{}': {}",
                    self.expression_str, e
                ))
            })?;

        Ok(result)
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expression_str)
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Threshold(vec![
            Expression::new("(d+1)/2").expect("Default threshold expression should be valid"),
        ])
    }
}

impl Threshold {
    /// Creates a new threshold collection from a vector of expressions.
    ///
    /// # Arguments
    /// * `expressions` - Vector of `Expression` objects to evaluate
    ///
    /// # Returns
    /// A new `Threshold` instance
    pub fn new(expressions: Vec<Expression>) -> Self {
        Self(expressions)
    }

    /// Evaluates all threshold expressions with the given code parameters and
    /// state.
    ///
    /// # Arguments
    /// * `code` - Reference to the MDPC code structure containing parameters
    /// * `state` - Current decoder state
    ///
    /// # Returns
    /// Vector of evaluated threshold values
    ///
    /// # Errors
    /// Returns `Error::Model` if any expression evaluation fails
    pub fn evaluate<S: State>(&self, code: &MDPCCode, state: &S) -> Result<Vec<usize>> {
        self.0
            .iter()
            .map(|expr| expr.evaluate(code, state))
            .collect::<Result<Vec<_>>>()
    }
    /// Returns the number of threshold expressions.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if there are no threshold expressions.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|expr| expr.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
