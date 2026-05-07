use parking_lot::Mutex;
use std::{
    error::Error, fs::File, io::{BufReader, Read, Seek, Error as IOError, ErrorKind}, sync::Arc,
};

use anyhow::Result;
use clap::Parser;
use encoderfile::{
    common::{ModelConfig, model_type::{
        Embedding, ImageClassification, ModelType, SentenceEmbedding, SequenceClassification, TokenClassification
    }},
    runtime::{ClassifierState, EncoderfileLoader, EncoderfileState, FeatureExtractorState, ImageInputState, TextInputState, load_assets},
    transport::cli::{TextCli, ImageCli},
};

#[tokio::main]
async fn main() -> Result<()> {
    // open current executable
    let path = std::env::current_exe()?;
    let file = File::open(path)?;
    let mut file = BufReader::new(file);
    // load encoderfile
    let mut loader = load_assets(&mut file)?;

    // entrypoint
    entrypoint(&mut loader).await
}

macro_rules! run_cli {
    ($model_type:ident, $cli:expr, $config:expr, $session:expr, $input_state:expr, $task_state:expr) => {{
        let state = Arc::new(EncoderfileState::<$model_type>::new(
            $config,
            $session,
            $input_state,
            $task_state,
        ));
        $cli.command.execute(state).await
    }};
}

async fn entrypoint<'a, R: Read + Seek>(loader: &mut EncoderfileLoader<'a, R>) -> Result<()> {
    let session = Mutex::new(loader.session()?);
    let model_config = loader.model_config()?;
    let config = loader.encoderfile_config()?;
    // TODO clear out lifetimes in state and loader to avoid

    fn class_task_state(model_config: &ModelConfig) -> ClassifierState {
        // if num_labels, make a vector of labels
        // if id2label, make sure it's 0..n-1
        ClassifierState {
            id2label: model_config.id2label.clone(),
            label2id: model_config.label2id.clone(),
            num_labels: model_config.num_labels,
        }
    }

    match loader.model_type() {
        ModelType::Embedding => run_cli!(
            Embedding,
            TextCli::parse(),
            config,
            session,
            TextInputState { tokenizer: loader.tokenizer()?, model_config },
            FeatureExtractorState {}
        ),
        ModelType::SequenceClassification => run_cli!(
            SequenceClassification,
            TextCli::parse(),
            config,
            session,
            TextInputState { tokenizer: loader.tokenizer()?, model_config: model_config.clone() },
            class_task_state(&model_config)
        ),
        ModelType::TokenClassification => run_cli!(
            TokenClassification,
            TextCli::parse(),
            config,
            session,
            TextInputState { tokenizer: loader.tokenizer()?, model_config: model_config.clone() },
            class_task_state(&model_config)
        ),
        ModelType::SentenceEmbedding => run_cli!(
            SentenceEmbedding,
            TextCli::parse(),
            config,
            session,
            TextInputState { tokenizer: loader.tokenizer()?, model_config },
            FeatureExtractorState {}
        ),
        ModelType::ImageClassification => run_cli!(
            ImageClassification,
            ImageCli::parse(),
            config,
            session,
            ImageInputState { 
                height: model_config.height(),
                width: model_config.width(),
                num_channels: model_config.num_channels().ok_or(IOError::new(ErrorKind::InvalidData, "Missing required configuration field"))?,
                image_size: model_config.image_size,
            },
            class_task_state(&model_config)
        ),
    }
}
