use crate::{common::SequenceClassificationResult, config::ModelConfig, error::ApiError};
use ndarray::{Axis, Ix2};
use ndarray_stats::QuantileExt;
use tokenizers::Encoding;

pub fn sequence_classification<'a>(
    mut session: crate::model::Model<'a>,
    config: &ModelConfig,
    encodings: Vec<Encoding>,
) -> Result<Vec<SequenceClassificationResult>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let outputs = crate::run_model!(session, a_ids, a_mask, a_type_ids)?;

    // get logits
    // will be in shape [N, L]
    let outputs = outputs
        .get("logits")
        .expect("Model does not return logits")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix2>()
        .expect("Model does not return tensor of shape [n_batch, n_labels]")
        .into_owned();

    let probabilities = super::utils::softmax(&outputs, Axis(1));

    let results = outputs
        .axis_iter(Axis(0))
        .zip(probabilities.axis_iter(Axis(0)))
        .map(|(logs, probs)| {
            let predicted_index = probs.argmax().expect("Model has 0 labels");
            SequenceClassificationResult {
                logits: logs.iter().map(|i| *i).collect(),
                scores: probs.iter().map(|i| *i).collect(),
                predicted_index: (predicted_index as u32),
                predicted_label: config
                    .id2label(predicted_index as u32)
                    .map(|i| i.to_string()),
            }
        })
        .collect();

    Ok(results)
}
