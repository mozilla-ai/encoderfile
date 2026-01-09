use crate::{
    AppState,
    common::model_type::ModelTypeSpec,
    services::Inference,
    transport::{grpc::GrpcRouter, http::HttpRouter},
};
use anyhow::Result;
use axum_server::tls_rustls::RustlsConfig;
use std::path::Path;
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

    let router = T::router(state)
        .layer(
            tower_http::trace::TraceLayer::new_for_grpc()
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    tracing::debug!(
        "TLS configuration: cert file: {:?}, cert key: {:?}",
        maybe_cert_file,
        maybe_key_file
    );

    tracing::info!("Running {:?} gRPC server on {}", model_type, &addr);

    match (maybe_cert_file, maybe_key_file) {
        (Some(cert_file), Some(key_file)) => {
            // ref: https://github.com/tokio-rs/axum/blob/main/examples/tls-rustls/src/main.rs#L45
            // configure certificate and private key used by https
            let config =
                RustlsConfig::from_pem_file(Path::new(&cert_file), Path::new(&key_file)).await?;

            let socket_addr = addr.parse()?;

            axum_server::bind_rustls(socket_addr, config)
                .serve(router)
                .await?
        }
        (None, Some(_)) => {
            return Err(crate::error::ApiError::ConfigError(
                "Both cert and key file need to be set. Only the key file has been set.",
            ))?;
        }
        (Some(_), None) => {
            return Err(crate::error::ApiError::ConfigError(
                "Both cert and key file need to be set. Only the cert file has been set.",
            ))?;
        }
        (None, None) => {
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("Invalid address: {addr}");

            axum::serve(listener, router).await?;
        }
    }

    Ok(())
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

    tracing::info!(
        "TLS configuration: cert file: {:?}, cert key: {:?}",
        maybe_cert_file,
        maybe_key_file
    );

    match (maybe_cert_file, maybe_key_file) {
        (Some(cert_file), Some(key_file)) => {
            tracing::info!("Running {:?} HTTPS server on {}", model_type, &addr);

            // ref: https://github.com/tokio-rs/axum/blob/main/examples/tls-rustls/src/main.rs#L45
            // configure certificate and private key used by https
            let config = RustlsConfig::from_pem_file(Path::new(&cert_file), Path::new(&key_file))
                .await
                .unwrap();

            let socket_addr = addr.parse()?;

            axum_server::bind_rustls(socket_addr, config)
                .serve(router)
                .await?
        }
        _ => {
            tracing::info!("Running {:?} HTTP server on {}", model_type, &addr);

            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("Invalid address: {addr}");

            axum::serve(listener, router).await?;
        }
    }

    Ok(())
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

    tracing::info!(
        "TLS configuration: cert file: {:?}, cert key: {:?}",
        maybe_cert_file,
        maybe_key_file
    );

    match (maybe_cert_file, maybe_key_file) {
        (Some(cert_file), Some(key_file)) => {
            tracing::info!("Running {:?} MPC-on-TLS server on {}", model_type, &addr);

            // ref: https://github.com/tokio-rs/axum/blob/main/examples/tls-rustls/src/main.rs#L45
            // configure certificate and private key used by https
            let config = RustlsConfig::from_pem_file(Path::new(&cert_file), Path::new(&key_file))
                .await
                .unwrap();

            let socket_addr = addr.parse()?;

            axum_server::bind_rustls(socket_addr, config)
                .serve(router)
                .await?
        }
        _ => {
            tracing::info!("Running {:?} MCP server on {}", model_type, &addr);

            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("Invalid address: {addr}");

            axum::serve(listener, router).await?;
        }
    }

    Ok(())
}
