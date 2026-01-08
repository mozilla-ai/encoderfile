use anyhow::{Result, bail};
use std::io::{Read, Seek};

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

        Ok(ArtifactReader::new(reader, &artifact))
    }
}

pub struct ArtifactReader<'a, R: Read + Seek> {
    reader: &'a mut R,
    offset: u64,
    remaining: u64,
}

impl<'a, R: Read + Seek> ArtifactReader<'a, R> {
    fn new(reader: &'a mut R, artifact: &Artifact) -> Self {
        Self {
            reader,
            offset: artifact.offset,
            remaining: artifact.length,
        }
    }
}
