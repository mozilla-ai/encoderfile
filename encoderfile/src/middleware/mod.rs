use axum::{
    body::Body,
    http::{HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use tracing::info_span;

pub async fn request_id(req: Request<Body>, next: Next) -> Response {
    // check for incoming header
    let headers = req.headers();
    let req_id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .and_then(sanitize_request_id)
        .unwrap_or_else(|| {
            uuid::Uuid::now_v7().to_string()
        })
        .to_string();

    // attach to tracing span
    let span = info_span!("request", %req_id);
    let _enter = span.enter();

    // put it in request extensions so handlers can use it
    let mut req = req;
    req.extensions_mut().insert(req_id.clone());

    let mut res = next.run(req).await;

    // echo it back in response
    res.headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&req_id).unwrap());

    res
}

fn sanitize_request_id(header_val: &str) -> Option<String> {
    if header_val.len() > 64 {
        return None;
    }
    let is_valid = header_val
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !is_valid {
        tracing::warn!("Invalid X-Request-ID, generating new one");
        return None;
    }
    Some(header_val.to_string())
}
