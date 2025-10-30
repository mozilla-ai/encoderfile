use anyhow::Result;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")), // default to "info" level
        )
        .with_target(false) // hide module path
        .compact() // short, pretty output
        .init();

    let encodings = encoderfile::inference::tokenizer::encode_text(vec![
        "hello my name is raz besaleli".to_string(),
    ])
    .unwrap();

    let classifications =
        encoderfile::inference::token_classification::token_classification(encodings)
            .await
            .unwrap();

    println!("{:?}", classifications);

    Ok(())
}
