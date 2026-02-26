pub mod base_binary;
// I'm not playing this game with clippy today -RB
#[allow(clippy::module_inception)]
pub mod builder;
pub mod cache;
pub mod cli;
pub mod config;
pub mod model;
pub mod templates;
/// Terminal logging utilities.
pub mod terminal;
pub mod tokenizer;
pub mod transforms;
