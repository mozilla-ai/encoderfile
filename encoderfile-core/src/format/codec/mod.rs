use anyhow::{Result, bail};
use std::{collections::HashSet, marker::PhantomData};

use crate::format::assets::{AssetPolicySpec, PlannedAsset};

pub struct EncoderfileCodec<T: AssetPolicySpec> {
    _data: PhantomData<T>,
}

impl<T: AssetPolicySpec> EncoderfileCodec<T> {
    pub fn validate_assets(assets: &[PlannedAsset]) -> Result<()> {
        let mut kinds: Vec<_> = assets.iter().map(|a| a.kind).collect();
        kinds.sort();

        // Enforce exactly-once per kind
        for w in kinds.windows(2) {
            if w[0] == w[1] {
                bail!("duplicate asset kind {:?}", w[0]);
            }
        }

        let present: HashSet<_> = kinds.iter().copied().collect();

        // Required must exist
        for req in T::required_assets() {
            if !present.contains(&req) {
                bail!("missing required asset {:?}. This should not happen.", req);
            }
        }

        // Present must be either required or optional
        for kind in &present {
            if !(T::required_assets().contains(&kind) || T::optional_assets().contains(&kind)) {
                bail!("asset {:?} not permitted for this model type", kind);
            }
        }

        Ok(())
    }
}
