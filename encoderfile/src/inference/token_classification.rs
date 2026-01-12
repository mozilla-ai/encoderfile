use crate::{
    common::{ModelConfig, TokenClassification, TokenClassificationResult, TokenInfo},
    error::ApiError,
    transforms::{Postprocessor, TokenClassificationTransform},
};
use ndarray::{Array3, Axis, Ix3};
use ndarray_stats::QuantileExt;
use tokenizers::Encoding;

#[tracing::instrument(skip_all)]
pub fn token_classification<'a>(
    mut session: crate::runtime::Model<'a>,
    transform: &TokenClassificationTransform,
    config: &ModelConfig,
    encodings: Vec<Encoding>,
) -> Result<Vec<TokenClassificationResult>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let mut outputs = crate::run_model!(session, a_ids, a_mask, a_type_ids)?
        .get("logits")
        .expect("Model does not return logits")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix3>()
        .expect("Model does not return tensor of shape [n_batch, n_tokens, n_labels]")
        .into_owned();

    outputs = transform.postprocess(outputs)?;

    let predictions = postprocess(outputs, encodings, config);

    Ok(predictions)
}

#[tracing::instrument(skip_all)]
pub fn postprocess(
    outputs: Array3<f32>,
    encodings: Vec<Encoding>,
    config: &ModelConfig,
) -> Vec<TokenClassificationResult> {
    let mut predictions = Vec::new();

    for (encoding, logits) in encodings.iter().zip(outputs.axis_iter(Axis(0))) {
        let mut results = Vec::new();

        for i in 0..encoding.len() {
            let argmax = logits
                .index_axis(Axis(0), i)
                .argmax()
                .expect("Model has 0 labels");
            let score = logits.index_axis(Axis(0), i)[argmax];
            let label = match config.id2label(argmax as u32) {
                Some(l) => l.to_string(),
                None => {
                    panic!(
                        "FATAL: No label found for ID {argmax}. Check to make sure that your config is correct."
                    )
                }
            };
            let (start, end) = encoding.get_offsets()[i];

            if encoding.get_special_tokens_mask()[i] == 1 {
                continue;
            }

            results.push(TokenClassification {
                token_info: TokenInfo {
                    token_id: encoding.get_ids()[i],
                    token: encoding.get_tokens()[i].clone(),
                    start,
                    end,
                },
                score,
                label,
                scores: logits
                    .index_axis(Axis(0), i)
                    .to_owned()
                    .into_raw_vec_and_offset()
                    .0,
            })
        }

        predictions.push(TokenClassificationResult { tokens: results });
    }

    predictions
}
