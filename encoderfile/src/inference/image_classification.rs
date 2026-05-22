use ndarray::{Array2, Array4, Axis, Ix2};

use crate::error::ApiError;

use crate::common::ImageLabelScore;

/*
fn logit_to_prob(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}
*/

#[tracing::instrument(skip_all)]
pub fn image_classification<'a>(
    mut session: crate::runtime::Model<'a>,
    // CHECK if this is a vec of flattened rgb images with num_channels X height X width
    images: Array4<f32>,
    classes: Vec<String>,
) -> Result<Vec<Vec<ImageLabelScore>>, ApiError> {
    let grouped_images = ort::value::TensorRef::from_array_view(&images)
        .unwrap()
        .to_owned();
    let raw_outputs = crate::run_cv_model!(session, grouped_images)?;
    let /*mut*/ outputs = raw_outputs
        .get("logits")
        .ok_or(ApiError::InternalError("Model does not return logits"))?
        .try_extract_array::<f32>()
        .map_err(|_| ApiError::InternalError("Model does not return tensor extractable to f32"))?
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            ApiError::InternalError("Model does not return tensor of shape [n_batch, n_classes]")
        })?
        .into_owned();
    // outputs.mapv_inplace(logit_to_prob);

    Ok(postprocess(outputs, classes))
}

#[tracing::instrument(skip_all)]
pub fn postprocess(outputs: Array2<f32>, classes: Vec<String>) -> Vec<Vec<ImageLabelScore>> {
    outputs
        .axis_iter(Axis(0))
        .map(|logs| {
            logs.iter()
                .enumerate()
                .map(|(idx, score)| {
                    ImageLabelScore {
                        label: classes[idx].to_string(), // TODO: get label from config
                        score: Some(*score),
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    // Add your test cases here
}
