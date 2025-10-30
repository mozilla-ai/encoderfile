use crate::{
    config::get_model_config,
    error::ApiError,
    inference::{inference::get_model, utils::requires_token_type_ids},
};
use ndarray::{Axis, Ix2};
use ndarray_stats::QuantileExt;
use tokenizers::Encoding;

pub async fn sequence_classification(
    encodings: Vec<Encoding>,
) -> Result<Vec<SequenceClassificationResult>, ApiError> {
    let mut session = get_model();

    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let outputs = match requires_token_type_ids(&session) {
        true => session.run(ort::inputs!(a_ids, a_mask, a_type_ids)),
        false => session.run(ort::inputs!(a_ids, a_mask)),
    }
    .map_err(|e| {
        tracing::error!("Error running model: {:?}", e);
        ApiError::InternalError("Error running model")
    })?;

    // get logits
    // will be in shape [N, L]
    let outputs = outputs
        .get("logits")
        .unwrap()
        .try_extract_array::<f32>()
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .into_owned();

    let probabilities = super::utils::softmax(&outputs, Axis(1));

    let model_config = get_model_config();

    let results = outputs
        .axis_iter(Axis(0))
        .zip(probabilities.axis_iter(Axis(0)))
        .map(|(logs, probs)| {
            let predicted_index = probs.argmax().unwrap();
            SequenceClassificationResult {
                logits: logs.iter().map(|i| *i).collect(),
                scores: probs.iter().map(|i| *i).collect(),
                predicted_index: (predicted_index as u32),
                predicted_label: model_config
                    .id2label(predicted_index as u32)
                    .map(|i| i.to_string()),
            }
        })
        .collect();

    Ok(results)
}

#[derive(Debug, serde::Serialize)]
pub struct SequenceClassificationResult {
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub predicted_index: u32,
    pub predicted_label: Option<String>,
}

impl From<SequenceClassificationResult>
    for crate::generated::sequence_classification::SequenceClassificationResult
{
    fn from(val: SequenceClassificationResult) -> Self {
        Self {
            logits: val.logits,
            scores: val.scores,
            predicted_index: val.predicted_index,
            predicted_label: val.predicted_label,
        }
    }
}
