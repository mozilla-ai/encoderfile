use super::EncoderfileCodec;
use anyhow::{Result, bail};

use crate::{
    common::model_type::ModelType,
    format::{
        assets::{AssetPlan, AssetPolicySpec},
        footer::EncoderfileFooter,
    },
    generated::manifest::{Artifact, Backend, EncoderfileManifest},
};

use prost::Message;
use std::{collections::HashSet, io::Write};

impl EncoderfileCodec {
    pub fn write<T, W>(
        &self,
        name: String,
        version: String,
        model_type: ModelType,
        backend: Backend,
        plan: &AssetPlan,
        out: &mut W,
    ) -> Result<()>
    where
        T: AssetPolicySpec,
        W: Write,
    {
        // validate assets
        Self::validate_assets::<T>(plan)?;

        let model_type: crate::generated::metadata::ModelType = From::from(model_type);

        let assets = plan.assets();

        let mut manifest = EncoderfileManifest {
            name,
            version,
            model_type: model_type.into(),
            backend: backend.into(),
            model_config: None,
            weights: None,
            transform: None,
            tokenizer: None,
        };

        for asset in assets {
            let artifact = Artifact::new(0, asset.length, asset.sha256);

            manifest.set_artifact(&asset.kind, artifact);
        }

        // compute manifest size
        let manifest_bytes = manifest.encode_to_vec();
        let manifest_len = manifest_bytes.len() as u64;

        // assign offsets
        let mut offset = manifest_len;

        for asset in assets {
            manifest.set_offset(&asset.kind, offset)?;
            offset += asset.length;
        }

        // write manifest
        let manifest_bytes = manifest.encode_to_vec();
        out.write_all(&manifest_bytes)?;

        // write assets
        for asset in assets {
            let written_len = asset.source.write_to(out)?;
            debug_assert_eq!(written_len, asset.length);
        }

        // write footer
        let footer = EncoderfileFooter::new(
            self.absolute_offset,
            manifest_bytes.len() as u64,
            true, // metadata is protobuf
        );

        footer.write_to(out)?;

        Ok(())
    }

    pub fn validate_assets<T: AssetPolicySpec>(plan: &AssetPlan) -> Result<()> {
        let kinds: Vec<_> = plan.assets().iter().map(|i| i.kind).collect();

        // Enforce exactly-once per kind
        for w in kinds.windows(2) {
            if w[0] == w[1] {
                bail!("duplicate asset kind {:?}", w[0]);
            }
        }

        let present: HashSet<_> = kinds.iter().copied().collect();

        // Required must exist
        for req in T::required_assets() {
            if !present.contains(req) {
                bail!("missing required asset {:?}. This should not happen.", req);
            }
        }

        // Present must be either required or optional
        for kind in &present {
            if !(T::required_assets().contains(kind) || T::optional_assets().contains(kind)) {
                bail!("asset {:?} not permitted for this model type", kind);
            }
        }

        Ok(())
    }
}
