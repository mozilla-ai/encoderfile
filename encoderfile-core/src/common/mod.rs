mod embedding;
mod model_config;
mod model_metadata;
pub mod model_type;
mod sentence_embedding;
mod sequence_classification;
mod token;
mod token_classification;

pub use embedding::*;
pub use model_config::*;
pub use model_metadata::*;
pub use model_type::ModelType;
pub use sentence_embedding::*;
pub use sequence_classification::*;
pub use token::*;
pub use token_classification::*;

pub trait FromCliInput {
    fn from_cli_input(inputs: Vec<String>) -> Self;
}
