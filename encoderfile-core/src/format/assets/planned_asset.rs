use super::{kind::AssetKind, source::AssetSource};
use anyhow::Result;

#[derive(Debug)]
pub struct PlannedAsset<'a> {
    pub kind: AssetKind,
    pub length: u64,
    pub sha256: [u8; 32],
    pub source: AssetSource<'a>,
}

impl<'a> PlannedAsset<'a> {
    pub fn from_asset_source(source: AssetSource<'a>, kind: AssetKind) -> Result<Self> {
        let (length, sha256) = source.hash_and_len()?;

        Ok(Self {
            kind,
            length,
            sha256,
            source,
        })
    }
}
