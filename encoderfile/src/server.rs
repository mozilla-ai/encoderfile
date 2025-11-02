use crate::config::get_model_type;
use anyhow::Result;

pub async fn run_grpc(hostname: String, port: String) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    let router = crate::grpc::router()
        .layer(tower_http::trace::TraceLayer::new_for_grpc())
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    tracing::info!("Running {:?} gRPC server on {}", get_model_type(), &addr);

    axum::serve(listener, router).await?;

    Ok(())
}

pub async fn run_http(hostname: String, port: String) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    let router = crate::http::router()
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    tracing::info!("Running {:?} HTTP server on {}", get_model_type(), &addr);

    axum::serve(listener, router).await?;

    Ok(())
}
