use parking_lot::Mutex;
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    sync::Arc,
};

use anyhow::Result;
use clap::Parser;
use encoderfile::{
    common::model_type::{
        Embedding, SentenceEmbedding, SequenceClassification, TokenClassification, ModelType,
    },
    common::ModelConfig,
    runtime::{EncoderfileLoader, EncoderfileState, load_assets, TextInputState, FeatureExtractorState, ClassifierState},
    transport::cli::Cli,
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
    let cli = Cli::parse();
    let session = Mutex::new(loader.session()?);
    let model_config = loader.model_config()?;
    let tokenizer = loader.tokenizer()?;
    let config = loader.encoderfile_config()?;

    fn class_task_state(model_config: &ModelConfig) -> ClassifierState {
        ClassifierState {
            id2label: model_config.id2label.clone(),
            label2id: model_config.label2id.clone(),
            num_labels: model_config.num_labels,
        }
    }

    match loader.model_type() {
        ModelType::Embedding => run_cli!(
            Embedding,
            cli,
            config,
            session,
            TextInputState { tokenizer, model_config },
            FeatureExtractorState {}
        ),
        ModelType::SequenceClassification => run_cli!(
            SequenceClassification,
            cli,
            config,
            session,
            TextInputState { tokenizer, model_config: model_config.clone() },
            class_task_state(&model_config)
        ),
        ModelType::TokenClassification => run_cli!(
            TokenClassification,
            cli,
            config,
            session,
            TextInputState { tokenizer, model_config: model_config.clone() },
            class_task_state(&model_config)
        ),
        ModelType::SentenceEmbedding => run_cli!(
            SentenceEmbedding,
            cli,
            config,
            session,
            TextInputState { tokenizer, model_config },
            FeatureExtractorState {}
        ),
        ModelType::ImageClassification => panic!("ImageClassification is not yet supported in the CLI"),
    }
}
