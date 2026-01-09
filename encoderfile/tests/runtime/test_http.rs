macro_rules! test_router_mod {
    ($model_type:ident, $mod_name:ident, $state_func:ident, $test_input:expr) => {
        mod $mod_name {
            use axum::http::{Request, StatusCode};
            use encoderfile::runtime::transport::http::HttpRouter;
            use encoderfile_core::{common::*, dev_utils::*};
            use tower::ServiceExt;

            fn router() -> axum::Router {
                let state = $state_func();
                model_type::$model_type::http_router(state)
            }

            #[tokio::test]
            async fn test_health_route() {
                let router = router();

                // health should exist
                let request = Request::get("/health")
                    .body(axum::body::Body::from("{}"))
                    .unwrap();

                let resp = router.oneshot(request).await.unwrap();

                assert_eq!(resp.status(), StatusCode::OK);
            }

            #[tokio::test]
            async fn test_openapi_route() {
                let router = router();

                // openapi should exist
                let request = Request::get("/openapi.json")
                    .body(axum::body::Body::from("{}"))
                    .unwrap();

                let resp = router.oneshot(request).await.unwrap();

                assert_eq!(resp.status(), StatusCode::OK);
            }

            #[tokio::test]
            async fn test_model_config_route() {
                let router = router();

                // model should exist
                let request = Request::get("/model")
                    .body(axum::body::Body::from("{}"))
                    .unwrap();

                let resp = router.oneshot(request).await.unwrap();

                assert_eq!(resp.status(), StatusCode::OK);
            }

            #[tokio::test]
            async fn test_predict_route() {
                let router = router();

                let body = serde_json::to_string(&$test_input).unwrap();

                let request = Request::post("/predict")
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap();

                let resp = router.oneshot(request).await.unwrap();

                if resp.status() != StatusCode::OK {
                    panic!("{} {:#?}", resp.status(), resp.body())
                }

                assert_eq!(resp.status(), StatusCode::OK);
            }

            #[tokio::test]
            async fn test_predict_route_empty() {
                let router = router();

                let mut inp = $test_input;
                inp.inputs = vec![];

                let body = serde_json::to_string(&inp).unwrap();

                let request = Request::post("/predict")
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap();

                let resp = router.oneshot(request).await.unwrap();

                assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
            }
        }
    };
}

test_router_mod!(
    Embedding,
    embedding_tests,
    embedding_state,
    EmbeddingRequest {
        inputs: vec!["Test sentence 1".to_string(), "Test sentence 2".to_string()],
        metadata: None,
    }
);
test_router_mod!(
    SequenceClassification,
    sequence_classification_tests,
    sequence_classification_state,
    SequenceClassificationRequest {
        inputs: vec!["Test sentence 1".to_string(), "Test sentence 2".to_string()],
        metadata: None,
    }
);
test_router_mod!(
    TokenClassification,
    token_classification_tests,
    token_classification_state,
    TokenClassificationRequest {
        inputs: vec!["Test sentence 1".to_string(), "Test sentence 2".to_string()],
        metadata: None,
    }
);
test_router_mod!(
    SentenceEmbedding,
    sentence_embedding_tests,
    sentence_embedding_state,
    SentenceEmbeddingRequest {
        inputs: vec!["Test sentence 1".to_string(), "Test sentence 2".to_string()],
        metadata: None,
    }
);
