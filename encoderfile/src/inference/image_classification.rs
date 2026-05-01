use ndarray::{Array2, Array3, Array4, Ix2, Axis};

use crate::{
    error::ApiError,
};

use crate::common::{ImageClassificationResult, ImageLabelScore};

#[tracing::instrument(skip_all)]
pub fn image_classification<'a>(
    mut session: crate::runtime::Model<'a>,
    // CHECK if this is a flattened rgb image with num_channels X height X width
    images: Vec<Array3<f32>>,
) -> Result<Vec<ImageLabelScore>, ApiError> {
    let grouped_images = ort::value::TensorRef::from_array_view(&*images)
        .unwrap()
        .to_owned();
    let mut outputs = crate::run_cv_model!(session, grouped_images)?
        .get("logits")
        .expect("Model does not return logits")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix2>()
        .expect("Model does not return tensor of shape [n_batch, n_classes]")
        .into_owned();


    Ok(outputs)
}

#[tracing::instrument(skip_all)]
pub fn postprocess(outputs: Array2<f32>) -> Vec<ImageLabelScore> {
    outputs
        .axis_iter(Axis(0))
        .map(|(logs)| {
            ImageLabelScore {
                label: "dummy".to_string(), // TODO: get label from config
                score: 1.0
            }
        })
        .collect()
}