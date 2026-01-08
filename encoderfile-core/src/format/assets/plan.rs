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

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn planned(
        kind: AssetKind,
        bytes: &'static [u8],
    ) -> PlannedAsset<'static> {
        PlannedAsset::from_asset_source(
            AssetSource::InMemory(Cow::Borrowed(bytes)),
            kind,
        )
        .unwrap()
    }

    #[test]
    fn planned_asset_from_source_sets_fields() {
        let data = b"test bytes";
        let asset = PlannedAsset::from_asset_source(
            AssetSource::InMemory(Cow::Borrowed(data)),
            AssetKind::ModelConfig,
        )
        .unwrap();

        assert_eq!(asset.kind, AssetKind::ModelConfig);
        assert_eq!(asset.length, data.len() as u64);

        // sanity-check hash without reimplementing SHA here
        let (len, hash) = asset.source.hash_and_len().unwrap();
        assert_eq!(asset.length, len);
        assert_eq!(asset.sha256, hash);
    }

    #[test]
    fn asset_plan_sorts_by_kind() {
        let a = planned(AssetKind::Tokenizer, b"a");
        let b = planned(AssetKind::ModelWeights, b"b");
        let c = planned(AssetKind::Transform, b"c");

        let plan = AssetPlan::new(vec![a, b, c]).unwrap();
        let kinds: Vec<_> = plan.assets().iter().map(|a| a.kind).collect();

        assert_eq!(
            kinds,
            vec![
                AssetKind::ModelWeights,
                AssetKind::Transform,
                AssetKind::Tokenizer,
            ]
        );
    }

    #[test]
    fn asset_plan_rejects_duplicate_kinds() {
        let a1 = planned(AssetKind::ModelConfig, b"a");
        let a2 = planned(AssetKind::ModelConfig, b"b");

        let err = AssetPlan::new(vec![a1, a2]).unwrap_err();
        let msg = err.to_string();

        assert!(
            msg.contains("duplicate asset kind"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn assets_returns_internal_slice() {
        let a = planned(AssetKind::ModelWeights, b"x");
        let plan = AssetPlan::new(vec![a]).unwrap();

        let assets = plan.assets();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].kind, AssetKind::ModelWeights);
    }
}
