use crate::error::ApiError;

use super::{super::tensor::Tensor, EmbeddingTransform, Postprocessor};
use ndarray::{Array3, Ix3};

impl Postprocessor for EmbeddingTransform {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn postprocess(&self, data: Self::Input) -> Result<Self::Output, ApiError> {
        let func = match self.postprocessor() {
            Some(p) => p,
            None => return Ok(data),
        };

        let batch_size = data.shape()[0];
        let seq_len = data.shape()[1];

        let tensor = Tensor(data.into_dyn());

        let result = func
            .call::<Tensor>(tensor)
            .map_err(|e| ApiError::LuaError(e.to_string()))?
            .into_inner()
            .into_dimensionality::<Ix3>().map_err(|e| {
                tracing::error!("Failed to cast array into Ix3: {e}. Check your lua transform to make sure it returns a tensor of shape [batch_size, seq_len, *]");
                ApiError::InternalError("Error postprocessing embeddings")
            })?;

        let result_shape = result.shape();

        if batch_size != result_shape[0] || seq_len != result_shape[1] {
            tracing::error!(
                "Transform error: expected tensor of shape [{}, {}, *], got tensor of shape {:?}",
                batch_size,
                seq_len,
                result_shape
            );

            return Err(ApiError::InternalError("Error postprocessing embeddings"));
        }

        Ok(result)
    }
}
