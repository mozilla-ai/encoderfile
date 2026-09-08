use super::{
    TransformValidatorExt,
    utils::{BATCH_SIZE, random_tensor, validation_err, validation_err_ctx},
};
use crate::{
    common::ModelConfig, runtime::{ImageConfig, ImageInputState, ImagePreprocessing}, transforms::{Image, ImageClassificationTransform, Postprocessor, Preprocessor},
};
use anyhow::{Context, Result, Error};
use image::{ImageBuffer, Rgb};
use rand::{RngCore, SeedableRng, rngs::StdRng};

const TEST_NUM_LABELS: usize = 16;
const VALIDATION_RNG_SEED: u64 = 13;

impl TransformValidatorExt<ImageInputState> for ImageClassificationTransform 
    where Self: Preprocessor {
    fn dry_run(&self, model_config: &ModelConfig, _input_state: &ImageInputState) -> Result<()> {
        let num_labels = match model_config.num_labels() {
            Some(n) => n,
            None => validation_err(
                "Model config does not have `num_labels`, `id2label`, or `label2id` field. Please make sure you're using an ImageClassification model.",
            )?,
        };

        let dummy_logits = random_tensor(&[BATCH_SIZE, TEST_NUM_LABELS], (-1.0, 1.0))?;
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
        
        let channels_u32 = model_config
            .num_channels()
            .ok_or_else(|| Error::msg("Model config does not have `num_channels` field. Please make sure you're using an ImageClassification model."))?;
        let channels = channels_u32 as usize;
        let height_u32 = model_config.height().ok_or_else(|| Error::msg("Model config does not have `height` field. Please make sure you're using an ImageClassification model."))?;
        let width_u32 = model_config.width().ok_or_else(|| Error::msg("Model config does not have `width` field. Please make sure you're using an ImageClassification model."))?;
        let height = height_u32 as usize;
        let width = width_u32 as usize;
        let less_height = height / 2;
        let less_width = width / 2;
        let more_height = height * 2;
        let more_width = width * 2;

        // check preprocessing
        let dummy_image = {
            let mut rgb_data: Vec<u8> = vec![0u8; width * height * channels];
            let mut rng = StdRng::seed_from_u64(VALIDATION_RNG_SEED);
            rng.fill_bytes(&mut rgb_data);

            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width_u32, height_u32, rgb_data)
                .expect("Failed to build RGB8 image from random buffer")
        };
        let dummy_config = ImageInputState {
            config: ImageConfig {
                num_channels: channels_u32,
                image_size: None,
            },
            preprocessing: ImagePreprocessing {
                rescale_factor: Some(1.0),
                do_rescale: Some(false),
                do_resize: Some(true),
                do_normalize: Some(true),
                image_mean: Some(vec![0.485, 0.456, 0.406]),
                image_std: Some(vec![0.229, 0.224, 0.225]),
                image_processor_type: None,
                size: Some(crate::runtime::ImageSize {
                    shortest_edge: None,
                    height: Some(height_u32),
                    width: Some(width_u32),
                }),
            },
        };

        self.preprocess((
            Image(image::DynamicImage::ImageRgb8(dummy_image)),
            dummy_config,
        ))?;


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

    fn get_input_state(&self, encoderfile_config: &crate::builder::config::EncoderfileConfig) -> Result<ImageInputState> {
        super::get_image_input_state(encoderfile_config)
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::config::{EncoderfileConfig, ModelPath};
    use crate::builder::transforms::validation::{get_image_input_state, get_text_input_state};
use crate::common::model_type::ModelType;
    use crate::transforms::DEFAULT_LIBS;

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from("../models/image_classification")),
            model_type: ModelType::ImageClassification,
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

        ImageClassificationTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr end\n\nfunction Preprocess(img) return img:resize(224,224):to_array(3) end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config)
        .expect("Failed to validate");
    }

    #[test]
    fn test_bad_return_type() {
        let encoderfile_config = test_encoderfile_config();

        let result = ImageClassificationTransform::new(
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

        let result = ImageClassificationTransform::new(
            DEFAULT_LIBS.to_vec(),
            Some("function Postprocess(arr) return arr:sum_axis(1) end".to_string()),
        )
        .expect("Failed to create transform")
        .validate(&encoderfile_config);

        assert!(result.is_err());
    }
}
