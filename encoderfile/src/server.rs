use crate::{
    config::get_model_type,
    transport::{grpc, http},
};
use anyhow::Result;
use tower_http::trace::DefaultOnResponse;

#[cfg(not(tarpaulin_include))]
pub async fn run_grpc(hostname: String, port: String) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    let router = grpc::router()
        .layer(
            tower_http::trace::TraceLayer::new_for_grpc()
                .make_span_with(crate::middleware::format_span)
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Invalid address: {addr}");

    tracing::info!("Running {:?} gRPC server on {}", get_model_type(), &addr);

    axum::serve(listener, router).await?;

    Ok(())
}

#[cfg(not(tarpaulin_include))]
pub async fn run_http(hostname: String, port: String) -> Result<()> {
    use crate::state::AppState;

    let addr = format!("{}:{}", &hostname, &port);

    let state = AppState::default();

    let router = http::router(state)
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
                .make_span_with(crate::middleware::format_span)
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Invalid address: {addr}");

    tracing::info!("Running {:?} HTTP server on {}", get_model_type(), &addr);

    axum::serve(listener, router).await?;

    Ok(())
}



#[cfg(not(tarpaulin_include))]
pub async fn run_mcp(hostname: String, port: String) -> Result<()> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService,
        session::local::LocalSessionManager,
    };
    use crate::state::AppState;
    use crate::transport::mcp::Encoder;

    let addr = format!("{}:{}", &hostname, &port);

    // FIXME add otel around here

    let service = StreamableHttpService::new(
        move || Ok(Encoder::new(AppState::default())),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    // FIXME consolidate routes with existing axum
    let router = axum::Router::new().nest_service("/mcp", service);
    let tcp_listener = tokio::net::TcpListener::bind(addr).await?;
    let _ = axum::serve(tcp_listener, router)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.unwrap() })
        .await;
    Ok(())
}