use ndarray::{Array2, Axis, Ix1, Ix2};
use ort::value::TensorRef;
use tokenizers::Encoding;

use crate::{error::ApiError, inference::{inference::get_model, utils::requires_token_type_ids}};

pub async fn embedding(
    encodings: Vec<Encoding>,
    return_token_info: bool,
    normalize: bool,
) -> Result<Vec<Vec<TokenEmbedding>>, ApiError> {
    let mut session = get_model();

    // Get padded length of each encoding.
    let padded_token_length = encodings[0].len();

    // Get token IDs & mask as a flattened array.
    let ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
        .collect();
    let mask: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
        .collect();
    let type_ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
        .collect();

    // Convert flattened arrays into 2-dimensional tensors of shape [N, L].
    let a_ids =
        TensorRef::from_array_view(([encodings.len(), padded_token_length], &*ids)).unwrap();
    let a_mask =
        TensorRef::from_array_view(([encodings.len(), padded_token_length], &*mask)).unwrap();
    let a_type_ids =
        TensorRef::from_array_view(([encodings.len(), padded_token_length], &*type_ids)).unwrap();

    let outputs = match requires_token_type_ids(&session) {
        true => session.run(ort::inputs!(a_ids, a_mask, a_type_ids)),
        false => session.run(ort::inputs!(a_ids, a_mask)),
        }
        .map_err(|e| {
            tracing::error!("Error running model: {:?}", e);
            ApiError::InternalError("Error running model")
        })?;

    let outputs = outputs
        .get("last_hidden_state")
        .unwrap()
        .try_extract_array::<f32>()
        .unwrap();

    let mut embeddings = Vec::new();

    for (encoding, embs) in encodings.iter().zip(outputs.axis_iter(Axis(0))) {
        let mut transformed = embs.into_dimensionality::<Ix2>().unwrap().into_owned();

        if normalize {
            transformed = l2_normalize_rows(transformed);
        }

        let mut token_ids = encoding.get_ids().iter();
        let mut tokens = encoding.get_tokens().iter();
        let mut special_tokens_mask = encoding.get_special_tokens_mask().iter();
        let mut offsets = encoding.get_offsets().iter();
        let mut embeddings_iter = transformed.axis_iter(Axis(0));

        let mut results = Vec::new();

        while let (Some(token_id), Some(token), Some(special_tokens_mask), Some(offset), Some(e)) = (
            token_ids.next(),
            tokens.next(),
            special_tokens_mask.next(),
            offsets.next(),
            embeddings_iter.next(),
        ) {
            if *special_tokens_mask == 1 {
                continue;
            }

            let (start, end) = *offset;
            let embedding: Vec<f32> = e
                .into_dimensionality::<Ix1>()
                .unwrap()
                .iter()
                .map(|i| *i)
                .collect();

            let token_info = match return_token_info {
                true => Some(TokenInfo {
                    token: token.clone(),
                    token_id: *token_id,
                    start,
                    end,
                }),
                false => None,
            };

            results.push(TokenEmbedding {
                embedding,
                token_info,
            })
        }

        embeddings.push(results)
    }

    Ok(embeddings)

    // Err(ApiError::InternalError("Not Implemented"))
}

fn l2_normalize_rows(mut x: Array2<f32>) -> Array2<f32> {
    for mut row in x.axis_iter_mut(Axis(0)) {
        let norm = row.mapv(|v| v * v).sum().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|v| v / norm);
        }
    }
    x
}

#[derive(Debug)]
pub struct TokenEmbedding {
    pub embedding: Vec<f32>,
    pub token_info: Option<TokenInfo>,
}

impl From<TokenEmbedding> for crate::generated::encoderfile::TokenEmbedding {
    fn from(val: TokenEmbedding) -> Self {
        crate::generated::encoderfile::TokenEmbedding {
            embedding: val.embedding,
            token_info: val.token_info.map(|i| i.into()),
        }
    }
}

#[derive(Debug)]
pub struct TokenInfo {
    pub token: String,
    pub token_id: u32,
    pub start: usize,
    pub end: usize,
}

impl From<TokenInfo> for crate::generated::encoderfile::TokenInfo {
    fn from(val: TokenInfo) -> Self {
        crate::generated::encoderfile::TokenInfo {
            token: val.token,
            token_id: val.token_id,
            start: (val.start as u32),
            end: (val.end as u32),
        }
    }
}
