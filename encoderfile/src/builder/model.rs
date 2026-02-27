use crate::format::assets::{AssetKind, AssetSource, PlannedAsset};
use anyhow::{Result, bail};
use ort::{
    session::{Output, Session},
    tensor::Shape,
};
use std::path::Path;

pub trait ModelTypeExt {
    fn validate_model<'a>(&self, path: &'a Path) -> Result<PlannedAsset<'a>>;
}

impl ModelTypeExt for crate::common::ModelType {
    fn validate_model<'a>(&self, path: &'a Path) -> Result<PlannedAsset<'a>> {
        let model = load_model(path)?;

        match self {
            Self::Embedding => validate_embedding_model(model),
            Self::SequenceClassification => validate_sequence_classification_model(model),
            Self::TokenClassification => validate_token_classification_model(model),
            Self::SentenceEmbedding => validate_sentence_embedding_model(model),
        }?;

        PlannedAsset::from_asset_source(AssetSource::File(path), AssetKind::ModelWeights)
    }
}

fn validate_sentence_embedding_model(model: Session) -> Result<()> {
    let shape = get_outp_dim(model.outputs.as_slice(), "last_hidden_state")?;

    // sentence embedding output shape must be identical to token embedding output shape.
    // mean pooling is taken care of at postprocessing step.
    if shape.len() != 3 {
        bail!("Model must return tensor of shape [batch_size, n_labels]")
    }

    Ok(())
}

fn validate_embedding_model(model: Session) -> Result<()> {
    let shape = get_outp_dim(model.outputs.as_slice(), "last_hidden_state")?;

    if shape.len() != 3 {
        bail!("Model must return tensor of shape [batch_size, seq_len, hidden_dim]")
    }

    Ok(())
}

fn validate_sequence_classification_model(model: Session) -> Result<()> {
    let shape = get_outp_dim(model.outputs.as_slice(), "logits")?;

    if shape.len() != 2 {
        bail!("Model must return tensor of shape [batch_size, n_labels]")
    }

    Ok(())
}

fn validate_token_classification_model(model: Session) -> Result<()> {
    let shape = get_outp_dim(model.outputs.as_slice(), "logits")?;

    if shape.len() != 3 {
        bail!("Model must return tensor of shape [batch_size, seq_len, n_labels]")
    }

    Ok(())
}

fn get_outp_dim<'a>(outputs: &'a [Output], outp_name: &str) -> Result<&'a Shape> {
    outputs
        .iter()
        .find(|i| i.name == outp_name)
        .ok_or(anyhow::anyhow!(format!("Model must return {}", outp_name)))?
        .output_type
        .tensor_shape()
        .ok_or(anyhow::anyhow!("Model must return tensor"))
}

fn load_model(file: &Path) -> Result<Session> {
    Ok(Session::builder()?.commit_from_file(file)?)
}
