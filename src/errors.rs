//! Simple error handling for the QC-MDPC Markovian model.
use std::fmt;

/// Main error type for all operations.
#[derive(Debug)]
pub enum Error {
    /// Input/Output related errors
    Io(std::io::Error),

    /// Configuration and parameter validation errors
    Config(String),

    /// Model computation errors
    Model(String),

    /// Parsing errors
    Parse(String),
}

/// Convenience type alias for Results in this crate.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Config(msg) => write!(f, "Configuration error: {}", msg),
            Self::Model(msg) => write!(f, "Model error: {}", msg),
            Self::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl Error {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a model error
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }

    /// Create a parsing error
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
}
