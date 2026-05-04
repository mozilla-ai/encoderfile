use ndarray::{Array2, Array4, Ix2, Axis};

use crate::{
    error::ApiError,
};

use crate::common::{ImageLabelScore};

#[tracing::instrument(skip_all)]
pub fn image_classification<'a>(
    mut session: crate::runtime::Model<'a>,
    // CHECK if this is a vec of flattened rgb images with num_channels X height X width
    images: Array4<f32>,
    classes: Vec<String>,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<Vec<ImageLabelScore>>, ApiError> {
    let grouped_images = ort::value::TensorRef::from_array_view(
        &images)
        .unwrap()
        .to_owned();
    let outputs = crate::run_cv_model!(session, grouped_images)?
        .get("logits")
        .expect("Model does not return logits")
        .try_extract_array::<f32>()
        .expect("Model does not return tensor extractable to f32")
        .into_dimensionality::<Ix2>()
        .expect("Model does not return tensor of shape [n_batch, n_classes]")
        .into_owned();


    Ok(postprocess(outputs, classes))
}

#[tracing::instrument(skip_all)]
pub fn postprocess(outputs: Array2<f32>, classes: Vec<String>) -> Vec<Vec<ImageLabelScore>> {
    outputs
        .axis_iter(Axis(0))
        .map(|logs| {
            logs.iter().enumerate()
                .map(|(idx, score)| 
                    ImageLabelScore {
                        label: classes[idx].to_string(), // TODO: get label from config
                        score: *score
                    }
                )
                .collect()
        })
        .collect()
}