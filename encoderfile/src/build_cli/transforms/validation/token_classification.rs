use super::{
    TransformValidatorExt,
    utils::{BATCH_SIZE, SEQ_LEN, random_tensor, validation_err, validation_err_ctx},
};
use crate::{
    common::ModelConfig,
    transforms::{Postprocessor, TokenClassificationTransform},
};
use anyhow::{Context, Result};

impl TransformValidatorExt for TokenClassificationTransform {
    fn dry_run(&self, model_config: &ModelConfig) -> Result<()> {
        let num_labels = match model_config.num_labels() {
            Some(n) => n,
            None => validation_err(
                "Model config does not have `num_labels`, `id2label`, or `label2id` field. Please make sure you're using a TokenClassification model.",
            )?,
        };

        let dummy_logits = random_tensor(&[BATCH_SIZE, SEQ_LEN, num_labels], (-1.0, 1.0))?;
        let shape = dummy_logits.shape().to_owned();

        let res = self.postprocess(dummy_logits)
            .with_context(|| {
                validation_err_ctx(
                    format!(
                        "Failed to run postprocessing on dummy logits (randomly generated in range -1.0..1.0) of shape {:?}",
                        shape.as_slice(),
                    )
                )
            })?;

        // result must return tensor of rank 3
        if res.ndim() != 3 {
            validation_err(format!(
                "Transform must return tensor of rank 3. Got tensor of shape {:?}.",
                res.shape()
            ))?
        }

        // result must have same shape as original
        if res.shape() != shape {
            validation_err(format!(
                "Transform must return Tensor of shape [batch_size, seq_len, num_labels]. Expected shape [{}, {}, {}], got shape {:?}",
                BATCH_SIZE,
                SEQ_LEN,
                num_labels,
                res.shape()
            ))?
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::build_cli::config::{EncoderfileConfig, ModelPath};
    use crate::common::ModelType;
    use crate::transforms::DEFAULT_LIBS;

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from("models/token_classification")),
            model_type: ModelType::TokenClassification,
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
        let config_json = include_str!("../../../../../models/token_classification/config.json");

        serde_json::from_str(config_json).unwrap()
    }

    #[test]
    fn test_identity_validation() {
        let encoderfile_config = test_encoderfile_config();
        let model_config = test_model_config();

        TokenClassificationTransform::new(
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

        let result = TokenClassificationTransform::new(
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

        let result = TokenClassificationTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr:sum_axis(1) end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config, &model_config);

        assert!(result.is_err());
    }
}
