// IMPORTANT NOTE:
//
// Image preprocessor configuration is NOT a stable, self-contained artifact (see tokenizer situation).
//
// It seems to vary widely between models and is often not even explicitly defined anywhere, so for now we just
// require users to provide the config for the model they are using, and we will deal with new
// models on a case-by-case basis as they come in.

use crate::format::assets::{AssetKind, AssetSource, PlannedAsset};
use anyhow::Result;

use super::config::EncoderfileConfig;
use crate::runtime::ImagePreprocessing;

pub fn validate_image_preprocessor<'a>(
    encoderfile_config: &'a EncoderfileConfig,
) -> Result<PlannedAsset<'a>> {
    let config = match encoderfile_config.path.preprocessor_config_path()? {
        // if preprocessor_config.json is provided, use that
        Some(preprocessor_config_path) => {
            // open preprocessor_config
            let contents = std::fs::read_to_string(preprocessor_config_path)?;
            let preprocessor_config: ImagePreprocessing = serde_json::from_str(contents.as_str())?;
            preprocessor_config
        }
        // some values may be present in config.json
        None => {
            // from_model_config(&image_preprocessing.config)?;
            anyhow::bail!("FATAL: No preprocessor_config.json provided");
        }
    };
    let model_config = encoderfile_config.model_config()?;
    let serialized = serde_json::to_vec(&config)?;

    // num_channels must be same as len for mean and std
    if let Some(num_channels) = model_config.num_channels {
        if let Some(image_mean) = config.image_mean.as_ref()
            && image_mean.len() != num_channels as usize
        {
            anyhow::bail!("num_channels must match length of image_mean");
        }
        if let Some(image_std) = config.image_std.as_ref()
            && image_std.len() != num_channels as usize
        {
            anyhow::bail!("num_channels must match length of image_std");
        }
    }

    PlannedAsset::from_asset_source(
        AssetSource::InMemory(std::borrow::Cow::Owned(serialized)),
        AssetKind::ImagePreprocessor,
    )
}

#[cfg(test)]
mod tests {
    use crate::builder::config::ModelPath;
    use crate::common::model_type::ModelType;

    use super::*;

    #[test]
    fn test_validate_preprocessor_config() {
        let config = EncoderfileConfig {
            name: "my-model".into(),
            version: "0.0.1".into(),
            path: ModelPath::Directory("../models/image_classification".into()),
            model_type: ModelType::Embedding,
            output_path: None,
            cache_dir: None,
            transform: None,
            lua_libs: None,
            tokenizer: None,
            validate_transform: false,
            base_binary_path: None,
            target: None,
        };

        let preprocessor_config = validate_image_preprocessor(&config)
            .expect("Failed to validate image preprocessor config");

        println!(
            "Validated image preprocessor config: {:?}",
            preprocessor_config
        );
    }
}
