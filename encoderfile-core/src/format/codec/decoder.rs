use super::EncoderfileCodec;
use crate::{
    format::{container::Encoderfile, footer::EncoderfileFooter},
    generated::manifest::EncoderfileManifest,
};
use anyhow::Result;
use prost::Message;
use std::io::{Read, Seek, SeekFrom};

impl EncoderfileCodec {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Encoderfile> {
        let footer = EncoderfileFooter::read_from(reader)?;

        // seek to manifest
        reader.seek(SeekFrom::Start(footer.metadata_offset))?;

        let mut buf = vec![0u8; footer.metadata_length as usize];
        reader.read_exact(&mut buf)?;

        let manifest = EncoderfileManifest::decode(&*buf)?;

        Ok(Encoderfile::new(manifest, footer))
    }
}
