use crate::{
    common::{TokenClassification, TokenClassificationResult, TokenInfo},
    model::config::ModelConfig,
    error::ApiError,
    inference::utils::softmax,
};
use ndarray::{Axis, Ix3};
use ndarray_stats::QuantileExt;
use tokenizers::Encoding;

#[tracing::instrument(skip_all)]
pub fn token_classification<'a>(
    mut session: crate::model::model::Model<'a>,
    config: &ModelConfig,
    encodings: Vec<Encoding>,
) -> Result<Vec<TokenClassificationResult>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let outputs = crate::run_model!(session, a_ids, a_mask, a_type_ids)?;

    let outputs = outputs
        .get("logits")
        .expect("Model does not return logits")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix3>()
        .expect("Model does not return tensor of shape [n_batch, n_tokens, n_labels]");

    let mut predictions = Vec::new();

    for (encoding, logits) in encodings.iter().zip(outputs.axis_iter(Axis(0))) {
        let logits = logits.to_owned();

        let scores = softmax(&logits, Axis(1));

        let mut token_ids = encoding.get_ids().iter();
        let mut tokens = encoding.get_tokens().iter();
        let mut special_tokens_mask = encoding.get_special_tokens_mask().iter();
        let mut offsets = encoding.get_offsets().iter();
        let mut logs_iter = logits.axis_iter(Axis(0));
        let mut scores_iter = scores.axis_iter(Axis(0));

        let mut results = Vec::new();

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
            let argmax = scores.argmax().expect("Model has 0 labels");
            let score = scores[argmax];
            let label = match config.id2label(argmax as u32) {
                Some(l) => l.to_string(),
                None => {
                    panic!(
                        "FATAL: No label found for ID {}. Check to make sure that your config is correct.",
                        argmax
                    )
                }
            };

            let (start, end) = *offset;

            if *special_tokens_mask == 1 {
                continue;
            }

            results.push(TokenClassification {
                token_info: TokenInfo {
                    token_id: *token_id,
                    token: token.clone(),
                    start,
                    end,
                },
                score: score,
                label,
                logits: logs.iter().map(|i| *i).collect(),
                scores: scores.iter().map(|i| *i).collect(),
            })
        }

        predictions.push(TokenClassificationResult { tokens: results });
    }

    Ok(predictions)
}
