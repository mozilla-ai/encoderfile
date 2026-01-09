use clap::Parser;
use parking_lot::Mutex;
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    sync::Arc,
};

use anyhow::Result;
use encoderfile_core::{
    AppState,
    common::{
        ModelType,
        model_type::{Embedding, SentenceEmbedding, SequenceClassification, TokenClassification},
    },
    format::codec::EncoderfileCodec,
    runtime::EncoderfileLoader,
};

use encoderfile_runtime::cli::Cli;

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
    ($model_type:ident, $cli:expr, $config:expr, $session:expr, $tokenizer:expr, $model_config:expr) => {{
        let state = AppState::<$model_type>::new($config, $session, $tokenizer, $model_config);
        $cli.command.execute(state).await
    }};
}

async fn entrypoint<'a, R: Read + Seek>(loader: &mut EncoderfileLoader<'a, R>) -> Result<()> {
    let cli = Cli::parse();
    let session = Arc::new(Mutex::new(loader.session()?));
    let model_config = Arc::new(loader.model_config()?);
    let tokenizer = Arc::new(loader.tokenizer()?);
    let config = Arc::new(loader.encoderfile_config()?);

    match loader.model_type() {
        ModelType::Embedding => run_cli!(Embedding, cli, config, session, tokenizer, model_config),
        ModelType::SequenceClassification => run_cli!(
            SequenceClassification,
            cli,
            config,
            session,
            tokenizer,
            model_config
        ),
        ModelType::TokenClassification => run_cli!(
            TokenClassification,
            cli,
            config,
            session,
            tokenizer,
            model_config
        ),
        ModelType::SentenceEmbedding => run_cli!(
            SentenceEmbedding,
            cli,
            config,
            session,
            tokenizer,
            model_config
        ),
    }
}

fn load_assets<'a, R: Read + Seek>(file: &'a mut R) -> Result<EncoderfileLoader<'a, R>> {
    let encoderfile = EncoderfileCodec::read(file)?;
    let loader = EncoderfileLoader::new(encoderfile, file);

    Ok(loader)
}
