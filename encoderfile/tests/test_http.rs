macro_rules! test_router_mod {
    ($model_type:ident, $mod_name:ident, $state_func:ident, $test_input:expr) => {
        mod $mod_name {
            use axum::http::{Request, StatusCode};
            use encoderfile::{common::*, dev_utils::*, transport::http::HttpRouter};
            use tower::ServiceExt;

            fn router() -> axum::Router {
                let state = $state_func();
                state.http_router()
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

mod image_classification_tests {
    use axum::http::{Request, StatusCode};
    use encoderfile::{dev_utils, transport::http::HttpRouter};
    use tower::ServiceExt;

    fn router() -> axum::Router {
        let state = dev_utils::image_classification_state();
        state.http_router()
    }

    #[tokio::test]
    async fn test_predict_route() {
        let router = router();
        let img_loc1 = "../test-pictures/yoga01.jpg";
        let img_loc2 = "../test-pictures/yoga02.jpg";
        let img_bytes1 = std::fs::read(img_loc1).unwrap();
        let img_bytes2 = std::fs::read(img_loc2).unwrap();
        let payload = serde_json::json!({
            "inputs": ["yoga01.jpg", "yoga02.jpg"],
            "metadata": {}
        });

        let boundary = "----encoderfile-boundary";
        let mut multipart_body = Vec::new();

        multipart_body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"payload\"\r\nContent-Type: application/json\r\n\r\n{}\r\n",
                payload
            )
            .as_bytes(),
        );

        multipart_body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"yoga01.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
            )
            .as_bytes(),
        );
        multipart_body.extend_from_slice(&img_bytes1);
        multipart_body.extend_from_slice(b"\r\n");

        multipart_body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"yoga02.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
            )
            .as_bytes(),
        );
        multipart_body.extend_from_slice(&img_bytes2);
        multipart_body.extend_from_slice(b"\r\n");

        multipart_body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let request = Request::post("/predict/multipart")
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(axum::body::Body::from(multipart_body))
            .unwrap();

        let resp = router.oneshot(request).await.unwrap();

        if resp.status() != StatusCode::OK {
            panic!("{} {:#?}", resp.status(), resp.body())
        }

        assert_eq!(resp.status(), StatusCode::OK);

        // gather the body into a single bytes object and convert it into a string for easier debugging if the test fails
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_string = String::from_utf8(body_bytes.to_vec()).unwrap();
        println!("Response body: {}", body_string);
    }
}
