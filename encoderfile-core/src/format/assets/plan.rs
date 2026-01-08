use super::{kind::AssetKind, source::AssetSource};
use anyhow::Result;

#[derive(Debug)]
pub struct AssetPlan<'a> {
    assets: Vec<PlannedAsset<'a>>,
}

impl<'a> AssetPlan<'a> {
    pub fn new(mut assets: Vec<PlannedAsset<'a>>) -> Result<Self> {
        assets.sort_by_key(|a| a.kind);

        for w in assets.windows(2) {
            if w[0].kind == w[1].kind {
                anyhow::bail!("duplicate asset kind {:?}", w[0].kind);
            }
        }

        Ok(Self { assets })
    }

    pub fn assets(&self) -> &[PlannedAsset<'a>] {
        &self.assets
    }
}

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
