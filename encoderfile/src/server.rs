use crate::{
    runtime::config::get_model_type,
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
