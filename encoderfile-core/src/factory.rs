use crate::{
    cli::Cli,
    common::model_type::ModelTypeSpec,
    runtime::{AppState, get_model, get_model_config, get_tokenizer},
    services::Inference,
    transport::{grpc::GrpcRouter, http::HttpRouter, mcp::McpRouter},
};

#[macro_export]
macro_rules! embed_in_section {
    ($name:ident, $path:expr, $section:expr, Bytes) => {
        $crate::embed_in_section!(
            $name,
            [u8; include_bytes!($path).len()],
            $section,
            *include_bytes!($path)
        );
    };

    ($name:ident, $path:expr, $section:expr, String) => {
        $crate::embed_in_section!($name, &str, $section, include_str!($path));
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr) => {
        $crate::embed_in_section!($name, $dtype, concat!("__DATA,", $section), $res, "macos");
        $crate::embed_in_section!($name, $dtype, concat!(".", $section), $res, "linux");
        $crate::embed_in_section!($name, $dtype, concat!(".rdata$", $section), $res, "windows");
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr, $target_os:expr) => {
        #[cfg(target_os = $target_os)]
        #[unsafe(link_section = $section)]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: $dtype = $res;
    };
}

#[macro_export]
macro_rules! factory {
    {
        $model_weights_path:expr,
        $tokenizer_path:expr,
        $model_config_path:expr,
        $model_type:ident,
        $model_id:expr,
        $transform:expr,
    } => {
        mod assets {
            $crate::embed_in_section!(MODEL_WEIGHTS, $model_weights_path, "model_weights", Bytes);
            $crate::embed_in_section!(TOKENIZER_JSON, $tokenizer_path, "tokenizer", String);
            $crate::embed_in_section!(MODEL_CONFIG_JSON, $model_config_path, "model_config", String);

            pub const MODEL_ID: &'static str = $model_id;

            #[allow(dead_code)]
            pub const TRANSFORM: Option<&'static str> = $transform;
        }

        fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            $crate::entrypoint::<$crate::common::model_type::$model_type>(
                &assets::MODEL_WEIGHTS,
                assets::MODEL_CONFIG_JSON,
                assets::TOKENIZER_JSON,
                assets::MODEL_ID,
                assets::TRANSFORM,
            )
        }
    };
}

pub fn entrypoint<T: ModelTypeSpec + GrpcRouter + McpRouter + HttpRouter>(
    model_bytes: &[u8],
    config_str: &str,
    tokenizer_json: &str,
    model_id: &str,
    transform_str: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    AppState<T>: Inference,
{
    use anyhow::Context;
    use clap::Parser;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .with_context(|| "Failed to create Tokio runtime")?;

    let cli = Cli::parse();

    let session = get_model(model_bytes);
    let config = get_model_config(config_str);
    let tokenizer = get_tokenizer(tokenizer_json, &config);
    let transform_str = transform_str.map(|t| t.to_string());
    let model_id = model_id.to_string();

    let state = AppState::new(session, tokenizer, config, model_id, transform_str);

    rt.block_on(cli.command.execute::<T>(state))?;

    Ok(())
}
