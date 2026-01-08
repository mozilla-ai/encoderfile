use anyhow::{Result, bail};
use std::io::{Read, Seek, SeekFrom};

use crate::{
    format::{assets::AssetKind, footer::EncoderfileFooter},
    generated::manifest::{Artifact, EncoderfileManifest},
};

#[derive(Debug)]
pub struct Encoderfile {
    manifest: EncoderfileManifest,
    _footer: EncoderfileFooter,
}

impl Encoderfile {
    pub fn new(manifest: EncoderfileManifest, footer: EncoderfileFooter) -> Self {
        Self {
            manifest,
            _footer: footer,
        }
    }

    pub fn open_required<'a, R: Read + Seek>(
        &self,
        reader: &'a mut R,
        kind: AssetKind,
    ) -> Result<ArtifactReader<'a, R>> {
        let artifact = match self.manifest.get_slot(&kind) {
            Some(s) => s,
            None => bail!("missing required artifact: {kind:?}"),
        };

        Ok(ArtifactReader::new(reader, artifact))
    }
}

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
    pub fn new(reader: &'a mut R, artifact: &Artifact) -> Self {
        Self {
            reader,
            start: artifact.offset,
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
