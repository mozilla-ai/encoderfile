use super::{
    TransformValidatorExt,
    utils::{BATCH_SIZE, HIDDEN_DIM, SEQ_LEN, random_tensor, validation_err, validation_err_ctx},
};
use crate::{
    common::ModelConfig,
    transforms::{EmbeddingTransform, Postprocessor},
};
use anyhow::{Context, Result};

impl TransformValidatorExt for EmbeddingTransform {
    fn dry_run(&self, _model_config: &ModelConfig) -> Result<()> {
        // create dummy hidden states with shape [batch_size, seq_len, hidden_dim]
        let dummy_hidden_states = random_tensor(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], (-1.0, 1.0))?;
        let shape = dummy_hidden_states.shape().to_owned();

        let res = self.postprocess(dummy_hidden_states)
            .with_context(|| {
                validation_err_ctx(
                    format!(
                        "Failed to run postprocessing on dummy logits (randomly generated in range -1.0..1.0) of shape {:?}",
                        shape.as_slice(),
                    )
                )
            })?;

        // result must return tensor of rank 3.
        if res.ndim() != 3 {
            validation_err(format!(
                "Transform must return tensor of rank 3. Got tensor of shape {:?}.",
                res.shape()
            ))?
        }

        // result must have same batch_size and seq_len
        if res.shape()[0] != BATCH_SIZE || res.shape()[1] != SEQ_LEN {
            validation_err(format!(
                "Transform must preserve batch and seq dims [{} {}, *]. Got shape {:?}",
                BATCH_SIZE,
                SEQ_LEN,
                res.shape()
            ))?
        }

        if res.shape()[2] < 1 {
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
    use crate::builder::config::{EncoderfileConfig, ModelPath};
    use crate::common::ModelType;
    use crate::transforms::DEFAULT_LIBS;

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from(
                "models/dummy_electra_token_embeddings",
            )),
            model_type: ModelType::Embedding,
            cache_dir: None,
            output_path: None,
            transform: None,
            lua_libs: None,
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
            target: None,
        }
    }

    fn test_model_config() -> ModelConfig {
        let config_json =
            include_str!("../../../../../models/dummy_electra_token_embeddings/config.json");

        serde_json::from_str(config_json).unwrap()
    }

    #[test]
    fn test_identity_validation() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        EmbeddingTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config)
        .expect("Failed to validate");
    }

    #[test]
    fn test_bad_return_type() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        let result = EmbeddingTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return 1 end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config);

        assert!(result.is_err());
    }

    #[test]
    fn test_bad_dimensionality() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        let result = EmbeddingTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr:sum_axis(1) end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config);

        assert!(result.is_err());
    }
}
