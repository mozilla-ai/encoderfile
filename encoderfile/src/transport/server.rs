use crate::{
    AppState,
    common::{ModelType, model_type::ModelTypeSpec},
    services::Inference,
    transport::{grpc::GrpcRouter, http::HttpRouter},
};
use anyhow::Result;
use axum::extract::connect_info::IntoMakeServiceWithConnectInfo;
use axum_server::tls_rustls::RustlsConfig;
use std::{net::SocketAddr, path::Path};
use tower_http::trace::DefaultOnResponse;

pub async fn run_grpc<T: ModelTypeSpec + GrpcRouter>(
    hostname: String,
    port: String,
    maybe_cert_file: Option<String>,
    maybe_key_file: Option<String>,
    state: AppState<T>,
) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    let model_type = T::enum_val();

    let router = T::grpc_router(state)
        .layer(
            tower_http::trace::TraceLayer::new_for_grpc()
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    serve_with_optional_tls(
        addr.as_str(),
        router,
        maybe_cert_file,
        maybe_key_file,
        model_type,
        "gRPC",
    ).await
}

pub async fn run_http<T: ModelTypeSpec + HttpRouter>(
    hostname: String,
    port: String,
    maybe_cert_file: Option<String>,
    maybe_key_file: Option<String>,
    state: AppState<T>,
) -> Result<()>
where
    AppState<T>: Inference,
{
    let addr = format!("{}:{}", &hostname, &port);

    let model_type = T::enum_val();

    let router = T::http_router(state)
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    serve_with_optional_tls(
        addr.as_str(),
        router,
        maybe_cert_file,
        maybe_key_file,
        model_type,
        "HTTP",
    ).await
}

pub async fn run_mcp<T: ModelTypeSpec + crate::transport::mcp::McpRouter>(
    hostname: String,
    port: String,
    maybe_cert_file: Option<String>,
    maybe_key_file: Option<String>,
    state: AppState<T>,
) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    // FIXME add otel around here
    let model_type = T::enum_val();

    let router = T::mcp_router(state)
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
                // TODO check if otel is enabled
                // .make_span_with(crate::middleware::format_span)
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    serve_with_optional_tls(
        addr.as_str(),
        router,
        maybe_cert_file,
        maybe_key_file,
        model_type,
        "MCP",
    ).await
}

async fn serve_with_optional_tls(
    addr: &str,
    router: IntoMakeServiceWithConnectInfo<axum::Router, SocketAddr>,
    maybe_cert_file: Option<String>,
    maybe_key_file: Option<String>,
    model_type: ModelType,
    server_type_str: &str
) -> Result<()> {
    match (maybe_cert_file, maybe_key_file) {
        (Some(cert), Some(key)) => {
            tracing::debug!(
                "TLS configuration: cert file: {:?}, cert key: {:?}",
                cert.as_str(),
                key.as_str()
            );

            tracing::info!("Running {:?} {:?} server with TLS on {}", model_type, server_type_str, &addr);

            let config =
                RustlsConfig::from_pem_file(Path::new(&cert), Path::new(&key)).await?;
            let socket_addr = addr.parse()?;
            axum_server::bind_rustls(socket_addr, config)
                .serve(router)
                .await?;
        }
        (None, None) => {
            tracing::info!("Running {:?} {:?} server on {}", model_type, server_type_str, &addr);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, router).await?;
        }
        _ => {
            anyhow::bail!("Both cert and key file must be set when TLS is enabled");
        }
    }

    Ok(())
}
