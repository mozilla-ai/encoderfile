use ndarray::{Axis, Ix2};
use tokenizers::Encoding;

use crate::{error::ApiError, inference::utils::requires_token_type_ids};

pub fn embedding<'a>(
    mut session: super::inference::Model<'a>,
    encodings: Vec<Encoding>,
    normalize: bool,
) -> Result<Vec<TokenEmbeddingSequence>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

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
            transformed = super::utils::l2_normalize(transformed, Axis(1));
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
            let embedding: Vec<f32> = e.iter().map(|i| *i).collect();

            let token_info = super::token_info::TokenInfo {
                token: token.clone(),
                token_id: *token_id,
                start,
                end,
            };

            results.push(TokenEmbedding {
                embedding,
                token_info: Some(token_info),
            })
        }

        embeddings.push(results)
    }

    Ok(embeddings)

    // Err(ApiError::InternalError("Not Implemented"))
}

#[derive(Debug, serde::Serialize)]
pub struct TokenEmbedding {
    pub embedding: Vec<f32>,
    pub token_info: Option<super::token_info::TokenInfo>,
}

impl From<TokenEmbedding> for crate::generated::embedding::TokenEmbedding {
    fn from(val: TokenEmbedding) -> Self {
        crate::generated::embedding::TokenEmbedding {
            embedding: val.embedding,
            token_info: val.token_info.map(|i| i.into()),
        }
    }
}

pub type TokenEmbeddingSequence = Vec<TokenEmbedding>;

impl From<TokenEmbeddingSequence> for crate::generated::embedding::TokenEmbeddingSequence {
    fn from(val: Vec<TokenEmbedding>) -> Self {
        Self {
            embeddings: val.into_iter().map(|i| i.into()).collect(),
        }
    }
}
