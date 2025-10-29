use crate::{
    config::get_model_config,
    error::ApiError,
    inference::{inference::get_model, utils::requires_token_type_ids},
};
use ndarray::{Axis, Ix2};
use ndarray_stats::QuantileExt;
use ort::value::TensorRef;
use tokenizers::Encoding;

pub async fn sequence_classification(
    encodings: Vec<Encoding>,
) -> Result<Vec<SequenceClassificationResult>, ApiError> {
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

#[derive(Debug)]
pub struct SequenceClassificationResult {
    logits: Vec<f32>,
    scores: Vec<f32>,
    predicted_index: u32,
    predicted_label: Option<String>,
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
