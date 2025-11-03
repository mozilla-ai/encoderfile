use axum::http::Request;
use tracing::{Span, info_span};

pub fn format_span<T>(req: &Request<T>) -> Span {
    let req_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("no-id");

    info_span!(
        "request",
        method = %req.method(),
        uri = %req.uri(),
        request_id = %req_id,
    )
}
