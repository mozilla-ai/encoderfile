use super::{
    TransformValidatorExt,
    utils::{BATCH_SIZE, random_tensor, validation_err, validation_err_ctx},
    get_text_input_state,
};
use crate::{
    common::ModelConfig,
    transforms::{Postprocessor, SequenceClassificationTransform},
};
use anyhow::{Context, Result};

use crate::runtime::TextInputState;

impl TransformValidatorExt<TextInputState> for SequenceClassificationTransform {
    fn dry_run(&self, model_config: &ModelConfig, _input_state: &TextInputState) -> Result<()> {
        let num_labels = match model_config.num_labels() {
            Some(n) => n,
            None => validation_err(
                "Model config does not have `num_labels`, `id2label`, or `label2id` field. Please make sure you're using a SequenceClassification model.",
            )?,
        };

        let dummy_logits = random_tensor(&[BATCH_SIZE, num_labels], (-1.0, 1.0))?;
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

        // result must return tensor of rank 2
        if res.ndim() != 2 {
            validation_err(format!(
                "Transform must return tensor of rank 2. Got tensor of shape {:?}.",
                res.shape()
            ))?
        }

        // result must have same shape as original
        if res.shape() != shape {
            validation_err(format!(
                "Transform must return Tensor of shape [batch_size, num_labels]. Expected shape [{}, {}], got shape {:?}",
                BATCH_SIZE,
                num_labels,
                res.shape()
            ))?
        }

        Ok(())
    }

    fn get_input_state(&self, encoderfile_config: &crate::builder::config::EncoderfileConfig) -> Result<TextInputState> {
        get_text_input_state(encoderfile_config)
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::config::{EncoderfileConfig, ModelPath};
    use crate::common::model_type::ModelType;
    use crate::transforms::DEFAULT_LIBS;

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from(
                "../models/sequence_classification",
            )),
            model_type: ModelType::SequenceClassification,
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

    #[test]
    fn test_identity_validation() {
        let encoderfile_config = test_encoderfile_config();

        SequenceClassificationTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config)
        .expect("Failed to validate");
    }

    #[test]
    fn test_bad_return_type() {
        let encoderfile_config = test_encoderfile_config();

        let result = SequenceClassificationTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return 1 end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config);

        assert!(result.is_err());
    }

    #[test]
    fn test_bad_dimensionality() {
        let encoderfile_config = test_encoderfile_config();

        let result = SequenceClassificationTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr:sum_axis(1) end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config);

        assert!(result.is_err());
    }
}
