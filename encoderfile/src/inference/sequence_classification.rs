use crate::{common::SequenceClassificationResult, error::ApiError, runtime::ModelConfig};
use ndarray::{Array2, Axis, Ix2};
use ndarray_stats::QuantileExt;
use ort::tensor::ArrayExtensions;
use tokenizers::Encoding;

#[tracing::instrument(skip_all)]
pub fn sequence_classification<'a>(
    mut session: crate::runtime::Model<'a>,
    config: &ModelConfig,
    encodings: Vec<Encoding>,
) -> Result<Vec<SequenceClassificationResult>, ApiError> {
    let (a_ids, a_mask, a_type_ids) = crate::prepare_inputs!(encodings);

    let outputs = crate::run_model!(session, a_ids, a_mask, a_type_ids)?
        .get("logits")
        .expect("Model does not return logits")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix2>()
        .expect("Model does not return tensor of shape [n_batch, n_labels]")
        .into_owned();

    let results = postprocess(outputs, config);

    Ok(results)
}

#[tracing::instrument(skip_all)]
pub fn postprocess(
    outputs: Array2<f32>,
    config: &ModelConfig,
) -> Vec<SequenceClassificationResult> {
    let probabilities = outputs.softmax(Axis(1));

    outputs
        .axis_iter(Axis(0))
        .zip(probabilities.axis_iter(Axis(0)))
        .map(|(logs, probs)| {
            let predicted_index = probs.argmax().expect("Model has 0 labels");
            SequenceClassificationResult {
                logits: logs.to_owned().into_raw_vec_and_offset().0,
                scores: probs.to_owned().into_raw_vec_and_offset().0,
                predicted_index: (predicted_index as u32),
                predicted_label: config
                    .id2label(predicted_index as u32)
                    .map(|i| i.to_string()),
            }
        })
        .collect()
}
