mod config;
mod embedding;
mod model_config;
mod model_metadata;
pub mod model_type;
mod sentence_embedding;
mod sequence_classification;
mod token;
mod token_classification;

// CV
mod image_classification;

pub use config::*;
pub use embedding::*;
pub use model_config::*;
pub use model_metadata::*;
pub use sentence_embedding::*;
pub use sequence_classification::*;
pub use token::*;
pub use token_classification::*;

// CV
pub use image_classification::*;
use std::io::Read;
use anyhow::Result;

pub trait FromCliInput {
    fn from_cli_input(inputs: Vec<String>) -> Self;
}

pub trait FromReadInput {
    fn from_read_input(input: Vec<&mut impl Read>) -> Result<Self>
    where Self: Sized;
}
