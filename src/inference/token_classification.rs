use crate::{config::get_model_config, error::ApiError, inference::utils::softmax};
use ndarray::{Axis, Ix2};
use ndarray_stats::QuantileExt;
use tokenizers::Encoding;

pub fn token_classification<'a>(
    mut session: super::inference::Model<'a>,
    encodings: Vec<Encoding>,
) -> Result<Vec<Vec<TokenClassification>>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let outputs = match super::utils::requires_token_type_ids(&session) {
        true => session.run(ort::inputs!(a_ids, a_mask, a_type_ids)),
        false => session.run(ort::inputs!(a_ids, a_mask)),
    }
    .map_err(|e| {
        tracing::error!("Error running model: {:?}", e);
        ApiError::InternalError("Error running model")
    })?;

    let outputs = outputs
        .get("logits")
        .unwrap()
        .try_extract_array::<f32>()
        .unwrap();

    let mut predictions = Vec::new();

    for (encoding, logits) in encodings.iter().zip(outputs.axis_iter(Axis(0))) {
        let logits = logits.into_dimensionality::<Ix2>().unwrap().to_owned();

        let scores = softmax(&logits, Axis(1));

        let mut token_ids = encoding.get_ids().iter();
        let mut tokens = encoding.get_tokens().iter();
        let mut special_tokens_mask = encoding.get_special_tokens_mask().iter();
        let mut offsets = encoding.get_offsets().iter();
        let mut logs_iter = logits.axis_iter(Axis(0));
        let mut scores_iter = scores.axis_iter(Axis(0));

        let mut results = Vec::new();

        let model_config = get_model_config();

        while let (
            Some(token_id),
            Some(token),
            Some(special_tokens_mask),
            Some(offset),
            Some(logs),
            Some(scores),
        ) = (
            token_ids.next(),
            tokens.next(),
            special_tokens_mask.next(),
            offsets.next(),
            logs_iter.next(),
            scores_iter.next(),
        ) {
            let argmax = scores.argmax().unwrap();
            let score = scores[argmax];
            let label = model_config.id2label(argmax as u32).unwrap().to_string();

            let (start, end) = *offset;

            if *special_tokens_mask == 1 {
                continue;
            }

            results.push(TokenClassification {
                token_info: super::token_info::TokenInfo {
                    token_id: *token_id,
                    token: token.clone(),
                    start,
                    end
                },
                score: score,
                label,
                logits: logs.iter().map(|i| *i).collect(),
                scores: scores.iter().map(|i| *i).collect(),
            })
        }

        predictions.push(results);
    }

    Ok(predictions)
}

#[derive(Debug, serde::Serialize)]
pub struct TokenClassificationResult {
    pub tokens: Vec<TokenClassification>,
}

impl From<TokenClassificationResult>
    for crate::generated::token_classification::TokenClassificationResult
{
    fn from(val: TokenClassificationResult) -> Self {
        Self {
            tokens: val.tokens.into_iter().map(|i| i.into()).collect(),
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct TokenClassification {
    pub token_info: super::token_info::TokenInfo,
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub label: String,
    pub score: f32,
}

impl From<TokenClassification> for crate::generated::token_classification::TokenClassification {
    fn from(val: TokenClassification) -> Self {
        Self {
            token_info: Some(val.token_info.into()),
            logits: val.logits,
            scores: val.scores,
            label: val.label,
            score: val.score,
        }
    }
}
