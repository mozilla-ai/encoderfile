use axum::{Json, extract::State, http::{Request, StatusCode}};
use encoderfile::{
    common::{EmbeddingRequest, SequenceClassificationRequest, TokenClassificationRequest},
    test_utils::{embedding_state, sequence_classification_state, token_classification_state},
    transport::http::{
        embedding, router, sequence_classification, token_classification
    },
};
use tower::ServiceExt;

macro_rules! test_empty_input {
    ($state:expr, $req:expr, $route_fn:expr) => {{
        let (code, _msg) = $route_fn(State($state), Json($req))
            .await
            .err()
            .expect("Did not error on empty input");

        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    }};
}

macro_rules! test_router {
    ($fn_name:ident, $state_func:ident) => {
        #[tokio::test]
        async fn $fn_name() {
            let state = $state_func();
            let router = router(state);

            // health should exist
            let request = Request::get("/health")
                .body(axum::body::Body::from("{}"))
                .unwrap();

            let resp = router.clone().oneshot(request).await.unwrap();

            assert_eq!(resp.status(), StatusCode::OK);

            // openapi should exist
            let request = Request::get("/openapi.json")
                .body(axum::body::Body::from("{}"))
                .unwrap();

            let resp = router.clone().oneshot(request).await.unwrap();

            assert_eq!(resp.status(), StatusCode::OK);

            // model should exist
            let request = Request::get("/model")
                .body(axum::body::Body::from("{}"))
                .unwrap();

            let resp = router.clone().oneshot(request).await.unwrap();

            assert_eq!(resp.status(), StatusCode::OK);

            // predict route should exist
            let request = Request::get("/predict")
                .body(axum::body::Body::from("{}"))
                .unwrap();
            
            let resp = router.clone().oneshot(request).await.unwrap();

            assert_ne!(resp.status(), StatusCode::NOT_FOUND);
        }
    }
}

macro_rules! test_successful_route {
    ($state:expr, $req:expr, $route_fn:expr) => {{
        let req = $req;
        let n_inputs = req.inputs.len();
        let metadata_is_none = req.metadata.is_none();

        let Json(resp) = $route_fn(State($state), Json(req))
            .await
            .expect("Expected successful call");

        assert_eq!(resp.metadata.is_none(), metadata_is_none);
        assert_eq!(resp.results.len(), n_inputs);
    }};
}

#[tokio::test]
async fn test_embedding_route() {
    test_successful_route!(
        embedding_state(),
        EmbeddingRequest {
            inputs: vec![
                "This is a test".to_string(),
                "This is also a test".to_string()
            ],
            normalize: true,
            metadata: None,
        },
        embedding::embedding
    );
}

#[tokio::test]
async fn test_embedding_route_empty() {
    test_empty_input!(
        embedding_state(),
        EmbeddingRequest {
            inputs: vec![],
            normalize: true,
            metadata: None,
        },
        embedding::embedding
    );
}

#[tokio::test]
async fn test_sequence_classification_route() {
    test_successful_route!(
        sequence_classification_state(),
        SequenceClassificationRequest {
            inputs: vec![
                "this is a test".to_string(),
                "this is also a test".to_string()
            ],
            metadata: None,
        },
        sequence_classification::sequence_classification
    )
}

#[tokio::test]
async fn test_sequence_classification_route_empty() {
    test_empty_input!(
        sequence_classification_state(),
        SequenceClassificationRequest {
            inputs: vec![],
            metadata: None,
        },
        sequence_classification::sequence_classification
    );
}

#[tokio::test]
async fn test_token_classification_route() {
    test_successful_route!(
        token_classification_state(),
        TokenClassificationRequest {
            inputs: vec![
                "this is a test".to_string(),
                "this is also a test".to_string()
            ],
            metadata: None,
        },
        token_classification::token_classification
    )
}

#[tokio::test]
async fn test_token_classification_route_empty() {
    test_empty_input!(
        token_classification_state(),
        TokenClassificationRequest {
            inputs: vec![],
            metadata: None,
        },
        token_classification::token_classification
    );
}

test_router!(test_embedding_router, embedding_state);
test_router!(test_sequence_cls_router, sequence_classification_state);
test_router!(test_token_cls_router, token_classification_state);
