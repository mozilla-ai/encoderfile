use crate::{
    format::assets::AssetKind,
    generated::manifest::{Artifact, EncoderfileManifest},
};
use anyhow::{Result, bail};

pub mod decoder;
pub mod encoder;

#[derive(Debug)]
pub struct EncoderfileCodec {
    absolute_offset: u64,
}

impl EncoderfileCodec {
    pub fn new(absolute_offset: u64) -> Self {
        Self { absolute_offset }
    }
}

impl EncoderfileManifest {
    fn set_artifact(&mut self, kind: &AssetKind, artifact: Artifact) {
        let slot = self.get_mut_slot(kind);

        *slot = Some(artifact);
    }

    fn set_offset(&mut self, kind: &AssetKind, offset: u64) -> Result<()> {
        let slot = self.get_mut_slot(kind);

        match slot {
            Some(s) => s.offset = offset,
            None => bail!("Cannot set offset for {kind:?}: artifact unset"),
        }
        Ok(())
    }

    fn get_mut_slot<'a>(&'a mut self, kind: &AssetKind) -> &'a mut Option<Artifact> {
        match kind {
            AssetKind::ModelWeights => &mut self.weights,
            AssetKind::ModelConfig => &mut self.model_config,
            AssetKind::Transform => &mut self.transform,
            AssetKind::Tokenizer => &mut self.tokenizer,
        }
    }

    pub fn get_slot(&self, kind: &AssetKind) -> &Option<Artifact> {
        match kind {
            AssetKind::ModelWeights => &self.weights,
            AssetKind::ModelConfig => &self.model_config,
            AssetKind::Transform => &self.transform,
            AssetKind::Tokenizer => &self.tokenizer,
        }
    }

    pub fn artifacts_iter(&self) -> impl Iterator<Item = (AssetKind, &Artifact)> {
        AssetKind::ORDERED
            .iter()
            .filter_map(|kind| self.get_slot(kind).as_ref().map(|a| (*kind, a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated::manifest::Artifact;

    fn empty_manifest() -> EncoderfileManifest {
        EncoderfileManifest {
            name: "test".into(),
            version: "0.0.0".into(),
            model_type: 0,
            backend: 0,
            model_config: None,
            weights: None,
            transform: None,
            tokenizer: None,
        }
    }

    fn artifact(len: u64) -> Artifact {
        Artifact {
            offset: 0,
            length: len,
            sha256: [0u8; 32].to_vec(),
        }
    }

    #[test]
    fn set_and_get_artifact() {
        let mut m = empty_manifest();
        let a = artifact(123);

        m.set_artifact(&AssetKind::ModelWeights, a.clone());

        let slot = m.get_slot(&AssetKind::ModelWeights);
        assert!(slot.is_some());
        assert_eq!(slot.as_ref().unwrap().length, 123);
    }

    #[test]
    fn set_offset_updates_existing_artifact() {
        let mut m = empty_manifest();
        m.set_artifact(&AssetKind::Tokenizer, artifact(10));

        m.set_offset(&AssetKind::Tokenizer, 42).unwrap();

        let a = m.get_slot(&AssetKind::Tokenizer).as_ref().unwrap();
        assert_eq!(a.offset, 42);
    }

    #[test]
    fn set_offset_fails_if_artifact_unset() {
        let mut m = empty_manifest();

        let err = m.set_offset(&AssetKind::Transform, 10).unwrap_err();
        let msg = err.to_string();

        assert!(
            msg.contains("artifact unset"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn get_slot_returns_none_for_unset() {
        let m = empty_manifest();

        assert!(m.get_slot(&AssetKind::ModelConfig).is_none());
        assert!(m.get_slot(&AssetKind::Transform).is_none());
    }

    #[test]
    fn artifacts_iter_yields_only_present_in_order() {
        let mut m = empty_manifest();

        m.set_artifact(&AssetKind::Tokenizer, artifact(1));
        m.set_artifact(&AssetKind::ModelWeights, artifact(2));
        m.set_artifact(&AssetKind::Transform, artifact(3));

        let kinds: Vec<_> = m.artifacts_iter().map(|(kind, _)| kind).collect();

        // Must follow AssetKind::ORDERED, not insertion order
        assert_eq!(
            kinds,
            AssetKind::ORDERED
                .iter()
                .copied()
                .filter(|k| matches!(
                    k,
                    AssetKind::ModelWeights | AssetKind::Transform | AssetKind::Tokenizer
                ))
                .collect::<Vec<_>>()
        );
    }
}
