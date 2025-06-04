# QC-MDPC Markovian model

This repository contains a Rust implementation of the refined Markovian model of Section 6 in

> Sarah Arpin, Jun Bo Lau, Antoine Mesnard, Ray Perlner, Angela Robinson, Jean-Pierre Tillich & Valentin Vasseur: Error floor prediction with Markov models for QC-MDPC codes. <https://eprint.iacr.org/2025/153>

This model predicts the DFR of a simple variant of bit-flipping decoding, named step-by-step decoding. This decoder is particularly amenable to Markov chain analysis.

## Documentation

Run `cargo doc` for documentation, or view the online version at <https://vvasseur.github.io/qcmdpc_markov_ncw>
