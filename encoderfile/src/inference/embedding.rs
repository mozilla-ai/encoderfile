use ndarray::{Array3, Axis, Ix3};
use tokenizers::Encoding;

use crate::{
    common::{TokenEmbedding, TokenEmbeddingSequence, TokenInfo},
    error::ApiError,
    runtime::ModelConfig,
};

#[tracing::instrument(skip_all)]
pub fn embedding<'a>(
    mut session: crate::runtime::Model<'a>,
    _config: &ModelConfig,
    encodings: Vec<Encoding>,
) -> Result<Vec<TokenEmbeddingSequence>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let outputs = crate::run_model!(session, a_ids, a_mask, a_type_ids)?
        .get("last_hidden_state")
        .expect("Model does not return last_hidden_state")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix3>()
        .expect("Model does not return tensor of shape [n_batch, n_tokens, hidden_dim]")
        .into_owned();

    let embeddings = postprocess(outputs, encodings);

    Ok(embeddings)
}

#[tracing::instrument(skip_all)]
pub fn postprocess(
    outputs: Array3<f32>,
    encodings: Vec<Encoding>,
) -> Vec<TokenEmbeddingSequence> {
    let mut embeddings = Vec::new();

    for (encoding, embs) in encodings.iter().zip(outputs.axis_iter(Axis(0))) {
        let transformed = embs.into_owned();

        let mut results = Vec::new();

        for i in 0..encoding.len() {
            if encoding.get_special_tokens_mask()[i] == 1 {
                continue;
            }

            let (start, end) = encoding.get_offsets()[i];
            let token_info = TokenInfo {
                token: encoding.get_tokens()[i].clone(),
                token_id: encoding.get_ids()[i],
                start,
                end,
            };

            let e = transformed.index_axis(Axis(0), i);
            results.push(TokenEmbedding {
                embedding: e.to_owned().into_raw_vec_and_offset().0,
                token_info: Some(token_info),
            });
        }

        embeddings.push(TokenEmbeddingSequence {
            embeddings: results,
        })
    }

    embeddings
}
