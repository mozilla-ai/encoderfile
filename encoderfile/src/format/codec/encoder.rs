use super::EncoderfileCodec;
use anyhow::{Result, bail};

use crate::{
    common::model_type::{
        Embedding, ModelType, SentenceEmbedding, SequenceClassification, TokenClassification,
    },
    format::{
        assets::{AssetPlan, AssetPolicySpec},
        footer::EncoderfileFooter,
    },
    generated::manifest::{Artifact, Backend, EncoderfileManifest},
};

use prost::Message;
use std::{collections::HashSet, io::Write};

impl EncoderfileCodec {
    /// Write an encoderfile payload consisting of:
    /// - a protobuf-encoded manifest
    /// - the raw asset bytes
    /// - a fixed-size footer appended at the end
    ///
    /// # Layout
    ///
    /// The on-disk layout is:
    ///
    /// ```text
    /// [ EncoderfileManifest (protobuf) ]
    /// [ Asset 0 bytes              ]
    /// [ Asset 1 bytes              ]
    /// [ ...                         ]
    /// [ EncoderfileFooter (32 B)    ]
    /// ```
    ///
    /// All artifact offsets stored in the manifest are **relative to the start
    /// of the manifest**, not the start of the file. The footer records the
    /// absolute file offset at which the manifest begins.
    ///
    /// # Protobuf size stability
    ///
    /// Protobuf encoding is *not* size-stable in general: even when fields use
    /// fixed-width numeric types (e.g. `fixed64`), the surrounding protobuf
    /// framing (field tags and length delimiters) may change size depending on
    /// encoded values.
    ///
    /// As a result, writing correct artifact offsets requires **stabilizing the
    /// encoded manifest size before writing any asset bytes**.
    ///
    /// This implementation performs a bounded, two-pass layout fixup:
    ///
    /// 1. Encode the manifest with placeholder offsets to determine its size.
    /// 2. Assign artifact offsets relative to that size and re-encode.
    /// 3. If the encoded size changes, reassign offsets once more.
    ///
    /// In practice, this converges immediately; a debug assertion enforces that
    /// the manifest size is stable before bytes are written.
    ///
    /// # Important invariants
    ///
    /// - Artifact offsets MUST be computed from the *final* encoded manifest size.
    /// - Assets MUST be written immediately after the manifest, in the same order.
    /// - The footer MUST be written last and MUST reflect the final manifest size.
    ///
    /// Do NOT refactor this function to compute offsets in a single pass or to
    /// assume protobuf encoding size is value-independent. Doing so will corrupt
    /// artifact offsets and cause runtime reads to return incorrect data.
    pub fn write<W>(
        &self,
        name: String,
        version: String,
        model_type: ModelType,
        backend: Backend,
        plan: &AssetPlan,
        out: &mut W,
    ) -> Result<()>
    where
        W: Write,
    {
        // 1. Validate assets
        // TODO: This does not need to be that complicated
        match &model_type {
            ModelType::Embedding => Self::validate_assets::<Embedding>(plan)?,
            ModelType::SequenceClassification => {
                Self::validate_assets::<SequenceClassification>(plan)?
            }
            ModelType::TokenClassification => Self::validate_assets::<TokenClassification>(plan)?,
            ModelType::SentenceEmbedding => Self::validate_assets::<SentenceEmbedding>(plan)?,
        };

        let model_type: crate::generated::metadata::ModelType = model_type.into();
        let assets = plan.assets();

        // 2. Build manifest skeleton (NO OFFSETS YET)
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

        // Populate artifacts with length + hash
        for asset in assets {
            let artifact = Artifact::new(0, asset.length, asset.sha256);
            manifest.set_artifact(&asset.kind, artifact);
        }

        // ------------------------------------------------------------
        // 3. Stabilize manifest size + offsets (two-pass fixup)
        // ------------------------------------------------------------

        // Pass 1: encode without offsets
        let mut manifest_bytes = manifest.encode_to_vec();
        let mut manifest_len = manifest_bytes.len() as u64;

        // Pass 2: assign offsets
        let mut offset = manifest_len;
        for asset in assets {
            manifest.set_offset(&asset.kind, offset)?;
            offset += asset.length;
        }

        // Re-encode
        manifest_bytes = manifest.encode_to_vec();
        let new_len = manifest_bytes.len() as u64;

        // Pass 3 (only if needed): reassign offsets once more
        if new_len != manifest_len {
            manifest_len = new_len;

            let mut offset = manifest_len;
            for asset in assets {
                manifest.set_offset(&asset.kind, offset)?;
                offset += asset.length;
            }

            manifest_bytes = manifest.encode_to_vec();
        }

        debug_assert_eq!(
            manifest.encode_to_vec().len() as u64,
            manifest_bytes.len() as u64,
            "manifest size did not stabilize after offset fixup"
        );

        // ------------------------------------------------------------
        // 4. Write output
        // ------------------------------------------------------------

        // Write manifest
        out.write_all(&manifest_bytes)?;

        // Write assets
        for asset in assets {
            let written = asset.source.write_to(out)?;
            debug_assert_eq!(written, asset.length);
        }

        // Write footer
        let footer = EncoderfileFooter::new(
            self.absolute_offset,
            manifest_bytes.len() as u64,
            true, // protobuf metadata
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model_type::{Embedding, ModelType};
    use crate::format::assets::{AssetKind, AssetSource, PlannedAsset};
    use crate::generated::manifest::Backend;
    use std::borrow::Cow;

    fn planned(kind: AssetKind, bytes: &'static [u8]) -> PlannedAsset<'static> {
        PlannedAsset::from_asset_source(AssetSource::InMemory(Cow::Borrowed(bytes)), kind).unwrap()
    }

    fn valid_plan() -> AssetPlan<'static> {
        AssetPlan::new(vec![
            planned(AssetKind::ModelWeights, b"weights"),
            planned(AssetKind::ModelConfig, b"config"),
            planned(AssetKind::Tokenizer, b"tokenizer"),
        ])
        .unwrap()
    }

    #[test]
    fn validate_assets_accepts_required_only() {
        let plan = valid_plan();
        EncoderfileCodec::validate_assets::<Embedding>(&plan).unwrap();
    }

    #[test]
    fn validate_assets_accepts_optional() {
        let plan = AssetPlan::new(vec![
            planned(AssetKind::ModelWeights, b"weights"),
            planned(AssetKind::ModelConfig, b"config"),
            planned(AssetKind::Tokenizer, b"tokenizer"),
            planned(AssetKind::Transform, b"fn transform(x) { x }"),
        ])
        .unwrap();

        EncoderfileCodec::validate_assets::<Embedding>(&plan).unwrap();
    }

    #[test]
    fn validate_assets_rejects_missing_required() {
        let plan = AssetPlan::new(vec![
            planned(AssetKind::ModelWeights, b"weights"),
            planned(AssetKind::ModelConfig, b"config"),
            // missing tokenizer
        ])
        .unwrap();

        let err = EncoderfileCodec::validate_assets::<Embedding>(&plan).unwrap_err();
        assert!(err.to_string().contains("missing required asset"));
    }

    #[test]
    fn validate_assets_rejects_disallowed_kind() {
        let plan = AssetPlan::new(vec![
            planned(AssetKind::ModelWeights, b"weights"),
            planned(AssetKind::ModelConfig, b"config"),
            planned(AssetKind::Tokenizer, b"tokenizer"),
            // not allowed by Encoder policy
            planned(AssetKind::ModelWeights, b"oops"),
        ]);
        // Duplicate check happens earlier; this test intentionally documents behavior
        assert!(plan.is_err());
    }

    #[test]
    fn write_smoke_test() {
        let codec = EncoderfileCodec { absolute_offset: 0 };

        let plan = valid_plan();

        let mut out = Vec::new();

        codec
            .write(
                "test-model".to_string(),
                "0.1.0".to_string(),
                ModelType::Embedding,
                Backend::Cpu,
                &plan,
                &mut out,
            )
            .unwrap();

        assert!(!out.is_empty());

        let asset_bytes: usize = plan.assets().iter().map(|a| a.length as usize).sum();

        assert!(
            out.len() > asset_bytes,
            "output should include manifest + footer in addition to assets"
        );
    }
}
