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
