use super::{
    TransformValidatorExt,
    utils::{
        BATCH_SIZE, HIDDEN_DIM, SEQ_LEN, create_dummy_attention_mask, random_tensor,
        validation_err, validation_err_ctx,
    },
};
use anyhow::{Context, Result};
use encoderfile_core::{
    common::ModelConfig,
    transforms::{Postprocessor, SentenceEmbeddingTransform},
};

impl TransformValidatorExt for SentenceEmbeddingTransform {
    fn dry_run(&self, _model_config: &ModelConfig) -> Result<()> {
        // create dummy hidden states with shape [batch_size, seq_len, hidden_dim]
        let dummy_hidden_states = random_tensor(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], (-1.0, 1.0))?;
        let dummy_attention_mask = create_dummy_attention_mask(BATCH_SIZE, SEQ_LEN, 3)?;
        let shape = dummy_hidden_states.shape().to_owned();

        let res = self.postprocess((dummy_hidden_states, dummy_attention_mask))
            .with_context(|| {
                validation_err_ctx(
                    format!(
                        "Failed to run postprocessing on dummy hidden states (randomly generated in range -1.0..1.0) of shape {:?}",
                        shape.as_slice(),
                    )
                )
            })?;

        // result must return tensor of rank 2
        if res.ndim() != 2 {
            validation_err(format!(
                "Transform must return tensor of rank 3. Got tensor of shape {:?}.",
                res.shape()
            ))?
        }

        // result must have same batch_size
        if res.shape()[0] != BATCH_SIZE {
            validation_err(format!(
                "Transform must preserve batch size [{}, *]. Got shape {:?}",
                BATCH_SIZE,
                res.shape()
            ))?
        }

        if res.shape()[1] < 1 {
            validation_err(format!(
                "Transform returned a tensor with last dimension 0. Shape: {:?}",
                res.shape()
            ))?
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{EncoderfileConfig, ModelPath};
    use encoderfile_core::common::ModelType;

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from("models/embedding")),
            model_type: ModelType::SentenceEmbedding,
            cache_dir: None,
            output_path: None,
            transform: None,
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
        }
    }

    fn test_model_config() -> ModelConfig {
        let config_json = include_str!("../../../../models/embedding/config.json");

        serde_json::from_str(config_json).unwrap()
    }

    #[test]
    fn test_successful_mean_pool() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        SentenceEmbeddingTransform::new(Some(
            "function Postprocess(arr, mask) return arr:mean_pool(mask) end".to_string(),
        ))
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config)
        .expect("Failed to validate");
    }

    #[test]
    fn test_bad_return_type() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        let result = SentenceEmbeddingTransform::new(Some(
            "function Postprocess(arr, mask) return 1 end".to_string(),
        ))
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config);

        assert!(result.is_err());
    }

    #[test]
    fn test_bad_dimensionality() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        let result = SentenceEmbeddingTransform::new(Some(
            "function Postprocess(arr, mask) return arr end".to_string(),
        ))
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config);

        assert!(result.is_err());
    }
}
