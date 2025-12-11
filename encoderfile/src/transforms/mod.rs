use crate::templates::transforms;
use anyhow::{Result, bail};

mod validation;

pub use validation::validate_transform;

pub fn new_transform(model_type: String) -> Result<()> {
    let template = match model_type.as_str() {
        "embedding" => transforms::EMBEDDING,
        "sequence_classification" => transforms::SEQUENCE_CLASSIFICATION,
        "token_classification" => transforms::TOKEN_CLASSIFICATION,
        "sentence_embedding" => transforms::SENTENCE_EMBEDDING,
        _ => bail!("Unknown model type: {}", model_type),
    };

    println!("{}\n", template.trim());

    Ok(())
}
