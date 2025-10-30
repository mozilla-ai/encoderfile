use crate::{
    error::ApiError,
    inference::{
        self, embedding::TokenEmbedding, inference::get_model,
        sequence_classification::SequenceClassificationResult,
        token_classification::TokenClassification, tokenizer::get_tokenizer,
    },
};

pub fn embedding(
    inputs: Vec<String>,
    normalize: bool,
) -> Result<Vec<Vec<TokenEmbedding>>, ApiError> {
    let tokenizer = get_tokenizer();
    let session = get_model();

    let encodings = inference::tokenizer::encode_text(tokenizer, inputs)?;

    inference::embedding::embedding(session, encodings, normalize)
}

pub fn sequence_classification(
    inputs: Vec<String>,
) -> Result<Vec<SequenceClassificationResult>, ApiError> {
    let tokenizer = get_tokenizer();
    let session = get_model();

    let encodings = inference::tokenizer::encode_text(tokenizer, inputs)?;

    inference::sequence_classification::sequence_classification(session, encodings)
}

pub fn token_classification(
    inputs: Vec<String>,
) -> Result<Vec<Vec<TokenClassification>>, ApiError> {
    let tokenizer = get_tokenizer();
    let session = get_model();

    let encodings = inference::tokenizer::encode_text(tokenizer, inputs)?;

    inference::token_classification::token_classification(session, encodings)
}
