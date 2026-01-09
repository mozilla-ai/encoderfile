use super::EncoderfileCodec;
use crate::{
    format::{container::Encoderfile, footer::EncoderfileFooter},
    generated::manifest::EncoderfileManifest,
};
use anyhow::{Context, Result, bail};
use prost::Message;
use std::io::{Read, Seek, SeekFrom};

impl EncoderfileCodec {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Encoderfile> {
        let footer =
            EncoderfileFooter::read_from(reader).context("failed to read encoderfile footer")?;

        // Determine file length
        let file_len = reader.seek(SeekFrom::End(0))?;

        let manifest_end = footer
            .metadata_offset
            .checked_add(footer.metadata_length)
            .context("metadata offset overflow")?;

        if manifest_end > file_len {
            bail!(
                "truncated encoderfile: manifest ends at byte {}, file length is {}",
                manifest_end,
                file_len
            );
        }

        // Seek and read manifest
        reader.seek(SeekFrom::Start(footer.metadata_offset))?;

        let mut buf = vec![0u8; footer.metadata_length as usize];
        reader
            .read_exact(&mut buf)
            .context("truncated encoderfile while reading manifest")?;

        let manifest =
            EncoderfileManifest::decode(&*buf).context("failed to decode manifest protobuf")?;

        Ok(Encoderfile::new(manifest, footer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::model_type::ModelType,
        format::assets::{AssetKind, AssetPlan, AssetSource, PlannedAsset},
        generated::manifest::Backend,
    };
    use std::{
        borrow::Cow,
        io::{Cursor, Seek, SeekFrom},
    };

    fn planned(kind: AssetKind, bytes: &'static [u8]) -> PlannedAsset<'static> {
        PlannedAsset::from_asset_source(AssetSource::InMemory(Cow::Borrowed(bytes)), kind).unwrap()
    }

    fn valid_plan() -> AssetPlan<'static> {
        AssetPlan::new(vec![
            planned(AssetKind::ModelWeights, b"weights"),
            planned(AssetKind::ModelConfig, b"config"),
            planned(AssetKind::Tokenizer, b"tokenizer"),
        ])
        .unwrap()
    }

    #[test]
    fn read_round_trips_written_encoderfile() {
        let codec = EncoderfileCodec { absolute_offset: 0 };

        let plan = valid_plan();
        let mut buf = Vec::new();

        codec
            .write::<_>(
                "test-model".to_string(),
                "0.1.0".to_string(),
                ModelType::Embedding,
                Backend::Cpu,
                &plan,
                &mut buf,
            )
            .unwrap();

        let mut cursor = Cursor::new(buf);
        let encoderfile = EncoderfileCodec::read(&mut cursor).unwrap();

        let manifest = encoderfile.manifest();

        assert_eq!(manifest.name, "test-model");
        assert_eq!(manifest.version, "0.1.0");
        assert!(manifest.model_config.is_some());
        assert!(manifest.weights.is_some());
        assert!(manifest.tokenizer.is_some());
    }

    #[test]
    fn read_respects_nonzero_absolute_offset() {
        let codec = EncoderfileCodec {
            absolute_offset: 128,
        };

        let plan = valid_plan();

        let mut buf = vec![0u8; 128]; // fake prefix (llamafile-style)
        codec
            .write::<_>(
                "offset-test".to_string(),
                "1.0.0".to_string(),
                ModelType::Embedding,
                Backend::Cpu,
                &plan,
                &mut buf,
            )
            .unwrap();

        let mut cursor = Cursor::new(buf);
        cursor.seek(SeekFrom::End(0)).unwrap();

        let encoderfile = EncoderfileCodec::read(&mut cursor).unwrap();
        let manifest = encoderfile.manifest();

        assert_eq!(manifest.name, "offset-test");
    }

    #[test]
    fn read_fails_on_truncated_input() {
        let mut buf = vec![0u8; 8]; // too small to contain footer
        let mut cursor = Cursor::new(&mut buf);

        let err = EncoderfileCodec::read(&mut cursor).unwrap_err();
        let msg = err.to_string();

        assert!(
            msg.contains("footer") || msg.contains("decode"),
            "unexpected error: {msg}"
        );
    }
}
