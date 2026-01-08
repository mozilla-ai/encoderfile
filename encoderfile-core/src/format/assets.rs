use anyhow::Result;
use ring::digest;
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
};

#[derive(Debug)]
pub enum AssetSource<'a> {
    File(&'a Path),
    InMemory(Cow<'a, [u8]>),
}

impl<'a> AssetSource<'a> {
    fn open(&'a self) -> std::io::Result<Box<dyn Read + 'a>> {
        match self {
            AssetSource::File(path) => Ok(Box::new(File::open(path)?)),
            AssetSource::InMemory(bytes) => Ok(Box::new(Cursor::new(bytes.as_ref()))),
        }
    }

    pub fn hash_and_len(&self) -> Result<(u64, [u8; 32])> {
        let reader = self.open()?;
        let mut reader = BufReader::new(reader);

        let mut ctx = digest::Context::new(&digest::SHA256);
        let mut buf = [0u8; 64 * 1024];
        let mut len: u64 = 0;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            ctx.update(&buf[..n]);
            len += n as u64;
        }

        let digest = ctx.finish();
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_ref());

        Ok((len, out))
    }

    pub fn planned_asset(&self, kind: AssetKind) -> Result<PlannedAsset> {
        let (length, sha256) = self.hash_and_len()?;

        Ok(PlannedAsset {
            kind,
            length,
            sha256,
        })
    }
}

#[derive(Debug)]
pub struct PlannedAsset {
    pub kind: AssetKind,
    pub length: u64,
    pub sha256: [u8; 32],
}

#[derive(Debug)]
pub enum AssetKind {
    ModelWeights,
    Transform,
    ModelConfig,
    Tokenizer,
}
