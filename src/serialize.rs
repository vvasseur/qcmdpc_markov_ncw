//! Utilities for input/output.
//!
//! Provides JSON streaming I/O and parsing utilities for state-value pairs,
//! threshold expressions, and code degree distributions.
use std::{
    collections::HashMap,
    io::{self, Read, Write},
    str::FromStr,
};

use serde::{Deserialize, Serialize};
use serde_json::Deserializer;

use crate::{
    code::CodeDegrees,
    errors::{Error, Result},
    threshold::{Expression, Threshold},
};

/// Writes state-value pairs to a writer in JSON format, with each pair on a
/// separate line.
///
/// # Arguments
///
/// * `iter` - An iterator yielding tuples of (state, value) pairs to be
///   serialized
/// * `writer` - A mutable reference to a writer where the JSON data will be
///   written
///
/// # Returns
///
/// Returns `Ok(())` on success, or an `io::Error` if writing fails or
/// serialization fails.
///
/// # Type Parameters
///
/// * `I` - Iterator type that yields (S, T) tuples
/// * `S` - State type that implements Serialize
/// * `T` - Value type that implements Serialize
/// * `W` - Writer type that implements Write
pub fn stream_write_to_file<I, S, T, W>(iter: I, writer: &mut W) -> io::Result<()>
where
    I: Iterator<Item = (S, T)>,
    S: Serialize,
    T: Serialize,
    W: Write,
{
    let mut first = true;
    for (state, value) in iter {
        if !first {
            writer.write_all(b"\n")?;
        }
        first = false;

        let json = serde_json::to_string(&(&state, value)).map_err(io::Error::other)?;
        writer.write_all(json.as_bytes())?;
    }
    writer.flush()?;

    Ok(())
}

/// Reads state-value pairs from a reader in JSON format, where each pair is on
/// a separate line.
///
/// # Arguments
///
/// * `reader` - A reader that implements Read, containing JSON data to be
///   deserialized
///
/// # Returns
///
/// Returns an iterator that yields `io::Result<(S, T)>` for each state-value
/// pair. Each item in the iterator is a Result containing either a successfully
/// parsed (state, value) tuple or an io::Error if parsing fails.
///
/// # Type Parameters
///
/// * `S` - State type that implements Deserialize
/// * `T` - Value type that implements Deserialize
/// * `R` - Reader type that implements Read
pub fn stream_read_from_file<S, T, R>(reader: R) -> impl Iterator<Item = io::Result<(S, T)>>
where
    S: for<'de> Deserialize<'de>,
    T: for<'de> Deserialize<'de>,
    R: Read,
{
    Deserializer::from_reader(reader)
        .into_iter()
        .map(|result| result.map_err(io::Error::other))
}

impl FromStr for Expression {
    type Err = Error;

    /// Parses an expression from a string representation.
    ///
    /// # Arguments
    ///
    /// * `s` - A string containing a mathematical expression
    ///
    /// # Returns
    ///
    /// An `Expression` or an `Error` if parsing fails.
    fn from_str(s: &str) -> Result<Self> {
        Expression::new(s)
    }
}

impl FromStr for Threshold {
    type Err = Error;

    /// Parses a threshold from a JSON array of expression strings.
    ///
    /// # Arguments
    ///
    /// * `s` - A string containing a JSON array of expression strings
    ///
    /// # Returns
    ///
    /// A `Threshold` or an `Error` if parsing fails.
    fn from_str(s: &str) -> Result<Self> {
        let exprs: Vec<String> = serde_json::from_str(s)
            .map_err(|e| Error::parse(format!("Failed to parse threshold JSON: {}", e)))?;
        let result: Vec<Expression> = exprs
            .into_iter()
            .map(|expr| Expression::new(&expr))
            .collect::<Result<Vec<_>>>()?;
        Ok(Threshold::new(result))
    }
}

impl FromStr for CodeDegrees {
    type Err = Error;

    /// Parses code degrees from a JSON string representation.
    ///
    /// # Arguments
    ///
    /// * `s` - A string containing JSON data representing code degrees
    ///
    /// # Returns
    ///
    /// `CodeDegrees` or an `Error` if parsing fails.
    fn from_str(s: &str) -> Result<Self> {
        serde_json::from_str(s)
            .map_err(|e| Error::parse(format!("Failed to parse code degrees JSON: {}", e)))
    }
}

impl Serialize for CodeDegrees {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let entries: Vec<_> = self
            .degrees
            .iter()
            .map(|block| {
                block
                    .iter()
                    .map(|(&degree, &count)| {
                        HashMap::from([
                            ("degree".to_string(), degree),
                            ("count".to_string(), count),
                        ])
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        entries.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CodeDegrees {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let degrees = parse_degree_distribution(&value)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        Ok(CodeDegrees { degrees })
    }
}

/// Parses a JSON value into a Vec of HashMaps representing degree
/// distributions.
///
/// # Arguments
///
/// * `value` - A serde_json::Value containing an array of arrays, where each
///   inner array contains objects with "degree" and "count" fields.
///
/// # Returns
///
/// A Vec of HashMaps where keys are degrees and values are counts, or an Error
/// if parsing fails.
///
/// # Errors
///
/// Returns an error if the JSON is invalid or missing required fields.
fn parse_degree_distribution(value: &serde_json::Value) -> Result<Vec<HashMap<usize, usize>>> {
    let mut vec = Vec::new();
    for block in value
        .as_array()
        .ok_or_else(|| Error::parse("Degree distribution must be an array of blocks"))?
    {
        let mut map = HashMap::new();
        for item in block
            .as_array()
            .ok_or_else(|| Error::parse("Each block must be an array of degree-count objects"))?
        {
            let degree = item["degree"].as_u64().ok_or_else(|| {
                Error::parse("Missing or invalid 'degree' field in degree distribution")
            })? as usize;
            let count = item["count"].as_u64().ok_or_else(|| {
                Error::parse("Missing or invalid 'count' field in degree distribution")
            })? as usize;
            map.insert(degree, count);
        }
        vec.push(map);
    }
    Ok(vec)
}
