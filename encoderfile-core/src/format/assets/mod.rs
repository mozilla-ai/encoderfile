use anyhow::Result;

mod kind;
mod planned_asset;
mod source;

pub use self::{kind::AssetKind, planned_asset::PlannedAsset, source::AssetSource};

pub struct AssetPlan<'a> {
    assets: Vec<PlannedAsset<'a>>,
}

impl<'a> AssetPlan<'a> {
    pub fn new(mut assets: Vec<PlannedAsset<'a>>) -> Result<Self> {
        assets.sort_by_key(|a| a.kind.clone());

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
