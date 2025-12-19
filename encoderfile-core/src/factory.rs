use crate::{
    cli::Cli,
    common::model_type::ModelTypeSpec,
    runtime::{AppState, get_config, get_model, get_model_config, get_tokenizer},
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
        $config:expr,
        $model_weights_path:expr,
        $tokenizer_path:expr,
        $model_config_path:expr,
        $model_type:ident
    } => {
        mod assets {
            $crate::embed_in_section!(MODEL_WEIGHTS, $model_weights_path, "model_weights", Bytes);
            $crate::embed_in_section!(TOKENIZER_JSON, $tokenizer_path, "tokenizer", String);
            $crate::embed_in_section!(MODEL_CONFIG_JSON, $model_config_path, "model_config", String);

            pub const CONFIG: &'static str = $config;
        }

        fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            $crate::entrypoint::<$crate::common::model_type::$model_type>(
                assets::CONFIG,
                &assets::MODEL_WEIGHTS,
                assets::MODEL_CONFIG_JSON,
                assets::TOKENIZER_JSON,
            )
        }
    };
}

pub fn entrypoint<T: ModelTypeSpec + GrpcRouter + McpRouter + HttpRouter>(
    config_str: &str,
    model_bytes: &[u8],
    model_config_str: &str,
    tokenizer_json: &str,
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
    let config = get_config(config_str);
    let session = get_model(model_bytes);
    let model_config = get_model_config(model_config_str);
    let tokenizer = get_tokenizer(tokenizer_json, &config);

    let state = AppState::new(config, session, tokenizer, model_config);

    rt.block_on(cli.command.execute::<T>(state))?;

    Ok(())
}
