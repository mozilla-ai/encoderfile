use anyhow::Result;
use std::io::{Read, Seek, SeekFrom};

use crate::{
    common::ModelType,
    format::{assets::AssetKind, footer::EncoderfileFooter},
    generated::manifest::{Artifact, EncoderfileManifest},
};

#[derive(Debug)]
pub struct Encoderfile {
    manifest: EncoderfileManifest,
    footer: EncoderfileFooter,
}

impl Encoderfile {
    pub fn new(manifest: EncoderfileManifest, footer: EncoderfileFooter) -> Self {
        Self { manifest, footer }
    }

    pub fn manifest(&self) -> &EncoderfileManifest {
        &self.manifest
    }

    pub fn footer(&self) -> &EncoderfileFooter {
        &self.footer
    }

    pub fn name(&self) -> &str {
        self.manifest.name.as_ref()
    }

    pub fn version(&self) -> &str {
        self.manifest.version.as_str()
    }

    pub fn model_type(&self) -> ModelType {
        self.manifest.model_type().into()
    }

    pub fn open_required<'a, R: Read + Seek>(
        &self,
        reader: &'a mut R,
        kind: AssetKind,
    ) -> Result<ArtifactReader<'a, R>> {
        self.open_optional(reader, kind)
            .ok_or_else(|| anyhow::anyhow!("Missing required artifact: {kind:?}"))
    }

    pub fn open_optional<'a, R: Read + Seek>(
        &self,
        reader: &'a mut R,
        kind: AssetKind,
    ) -> Option<ArtifactReader<'a, R>> {
        match self.manifest.get_slot(&kind) {
            Some(a) => Some(ArtifactReader::new(self.footer.metadata_offset, reader, a)),
            None => None,
        }
    }
}

#[derive(Debug)]
pub struct ArtifactReader<'a, R: Read + Seek> {
    reader: &'a mut R,
    /// Absolute file offset of artifact start
    start: u64,
    /// Current position within artifact
    pos: u64,
    /// Total artifact length
    length: u64,
}

impl<'a, R: Read + Seek> ArtifactReader<'a, R> {
    pub fn new(manifest_offset: u64, reader: &'a mut R, artifact: &Artifact) -> Self {
        Self {
            reader,
            start: manifest_offset + artifact.offset,
            pos: 0,
            length: artifact.length,
        }
    }

    pub fn len(&self) -> u64 {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, R: Read + Seek> Read for ArtifactReader<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.length {
            return Ok(0);
        }

        let max = buf.len().min((self.length - self.pos) as usize);

        self.reader.seek(SeekFrom::Start(self.start + self.pos))?;
        let n = self.reader.read(&mut buf[..max])?;

        self.pos += n as u64;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::model_type::{Embedding, ModelType},
        format::{
            assets::{AssetPlan, AssetSource, PlannedAsset},
            codec::EncoderfileCodec,
        },
        generated::manifest::Backend,
    };
    use std::{
        borrow::Cow,
        io::{Cursor, Read},
    };

    fn planned(kind: AssetKind, bytes: &'static [u8]) -> PlannedAsset<'static> {
        PlannedAsset::from_asset_source(AssetSource::InMemory(Cow::Borrowed(bytes)), kind).unwrap()
    }

    fn encoderfile_with_assets() -> (Encoderfile, Cursor<Vec<u8>>) {
        let plan = AssetPlan::new(vec![
            planned(AssetKind::ModelWeights, b"weights"),
            planned(AssetKind::ModelConfig, b"config"),
            planned(AssetKind::Tokenizer, b"tokenizer"),
        ])
        .unwrap();

        let codec = EncoderfileCodec::new(0);
        let mut buf = Vec::new();

        codec
            .write::<Embedding, _>(
                "test-model".into(),
                "1.0.0".into(),
                ModelType::Embedding,
                Backend::Cpu,
                &plan,
                &mut buf,
            )
            .unwrap();

        let mut cursor = Cursor::new(buf);
        let ef = EncoderfileCodec::read(&mut cursor).unwrap();

        (ef, cursor)
    }

    #[test]
    fn encoderfile_accessors_work() {
        let (ef, _) = encoderfile_with_assets();

        assert_eq!(ef.name(), "test-model");
        assert_eq!(ef.version(), "1.0.0");
        assert_eq!(ef.model_type(), ModelType::Embedding);
    }

    #[test]
    fn open_optional_returns_none_when_missing() {
        let (ef, mut cursor) = encoderfile_with_assets();

        let reader = ef.open_optional(&mut cursor, AssetKind::Transform);
        assert!(reader.is_none());
    }

    #[test]
    fn open_required_errors_when_missing() {
        let (ef, mut cursor) = encoderfile_with_assets();

        let err = ef
            .open_required(&mut cursor, AssetKind::Transform)
            .unwrap_err();

        assert!(err.to_string().contains("Missing required artifact"));
    }

    #[test]
    fn open_required_reads_entire_artifact() {
        let (ef, mut cursor) = encoderfile_with_assets();

        let mut reader = ef
            .open_required(&mut cursor, AssetKind::ModelWeights)
            .unwrap();

        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        assert_eq!(buf, b"weights");
        assert_eq!(reader.len(), 7);
        assert!(!reader.is_empty());
    }

    #[test]
    fn artifact_reader_respects_length_and_partial_reads() {
        let (ef, mut cursor) = encoderfile_with_assets();

        let mut reader = ef
            .open_required(&mut cursor, AssetKind::ModelConfig)
            .unwrap();

        let mut buf = [0u8; 2];

        let n1 = reader.read(&mut buf).unwrap();
        assert_eq!(n1, 2);
        assert_eq!(&buf, b"co");

        let n2 = reader.read(&mut buf).unwrap();
        assert_eq!(n2, 2);
        assert_eq!(&buf, b"nf");

        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"ig");

        let n3 = reader.read(&mut buf).unwrap();
        assert_eq!(n3, 0); // EOF
    }

    #[test]
    fn artifact_reader_is_empty_when_length_zero() {
        let artifact = Artifact {
            offset: 0,
            length: 0,
            sha256: [0u8; 32].to_vec(),
        };

        let mut data = Cursor::new(Vec::new());
        let reader = ArtifactReader::new(0, &mut data, &artifact);

        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
    }
}
