use crate::{
    format::assets::AssetKind,
    generated::manifest::{Artifact, EncoderfileManifest},
};
use anyhow::{Result, bail};

pub mod decoder;
pub mod encoder;

pub struct EncoderfileCodec;

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
