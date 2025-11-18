use ndarray::{Array2, Axis, Ix2, Ix3};
use tokenizers::Encoding;

use crate::{common::SentenceEmbedding, error::ApiError, runtime::AppState};

#[tracing::instrument(skip_all)]
pub fn sentence_embedding<'a>(
    mut session: crate::runtime::Model<'a>,
    state: &AppState,
    encodings: Vec<Encoding>,
) -> Result<Vec<SentenceEmbedding>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let a_mask_arr = a_mask
        .try_extract_array::<i64>()
        .expect("Failed to extract attention mask into i64")
        .into_dimensionality::<Ix2>()
        .expect("a_mask is not in Ix2")
        .into_owned()
        .mapv(|i| i as f32);

    let outputs = crate::run_model!(session, a_ids, a_mask, a_type_ids)?
        .get("last_hidden_state")
        .expect("Model does not return last_hidden_state")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix3>()
        .expect("Model does not return tensor of shape [n_batch, n_tokens, hidden_dim]")
        .into_owned();

    let transform = state.transform();

    let pooled_outputs = transform.pool(outputs, a_mask_arr)?;

    let embeddings = postprocess(pooled_outputs, encodings);

    Ok(embeddings)
}

#[tracing::instrument(skip_all)]
pub fn postprocess(outputs: Array2<f32>, _encodings: Vec<Encoding>) -> Vec<SentenceEmbedding> {
    outputs
        .axis_iter(Axis(0))
        .map(|emb| SentenceEmbedding {
            embedding: emb.to_owned().into_raw_vec_and_offset().0,
        })
        .collect()
}
