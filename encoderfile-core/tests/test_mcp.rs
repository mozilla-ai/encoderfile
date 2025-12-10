use anyhow::Result;
use encoderfile_core::common::model_type::ModelTypeSpec;
use encoderfile_core::transport::mcp::McpRouter;
use encoderfile_core::{AppState, transport::mcp};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tower_http::trace::DefaultOnResponse;

async fn run_mcp<T: ModelTypeSpec + McpRouter>(
    addr: String,
    state: AppState<T>,
    receiver: oneshot::Receiver<()>,
) -> Result<()> {
    let router = mcp::make_router(state).layer(
        tower_http::trace::TraceLayer::new_for_http()
            // TODO check if otel is enabled
            // .make_span_with(crate::middleware::format_span)
            .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
    );
    tracing::info!("Running {:?} MCP server on {}", T::enum_val(), &addr);
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            receiver.await.ok();
            tracing::info!("Received shutdown signal, shutting down");
        })
        .await
        .expect("Error while shutting down server");
    Ok(())
}

macro_rules! test_mcp_server_impl {
    ($mod_name:ident, $state_func:ident, $req_type:ident, $resp_type:ident) => {
        pub mod $mod_name {
            use encoderfile_core::{
                common::{$req_type, $resp_type},
                dev_utils::$state_func,
            };
            use rmcp::{
                ServiceExt,
                model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
                transport::StreamableHttpClientTransport,
            };
            use tokio::sync::oneshot;

            const LOCALHOST: &str = "localhost";
            const PORT: i32 = 9100;

            pub async fn $mod_name() {
                let addr = format!("{}:{}", LOCALHOST, PORT);
                let dummy_state = $state_func();
                let (sender, receiver) = oneshot::channel();
                let _mcp_server = tokio::spawn(super::run_mcp(addr, dummy_state, receiver));
                // Client usage copied over from https://github.com/modelcontextprotocol/rust-sdk/blob/main/examples/clients/src/streamable_http.rs
                let client_transport = StreamableHttpClientTransport::from_uri(format!(
                    "http://{}:{}/mcp",
                    LOCALHOST, PORT
                ));
                let client_info = ClientInfo {
                    protocol_version: Default::default(),
                    capabilities: ClientCapabilities::default(),
                    client_info: Implementation {
                        name: "test sse client".to_string(),
                        title: None,
                        version: "0.0.1".to_string(),
                        website_url: None,
                        icons: None,
                    },
                };
                let client = client_info
                    .serve(client_transport)
                    .await
                    .inspect_err(|e| {
                        tracing::error!("client error: {:?}", e);
                    })
                    .unwrap();
                // Initialize
                let server_info = client.peer_info();
                tracing::info!("Connected to server: {server_info:#?}");

                // List tools
                let tools = client
                    .list_tools(Default::default())
                    .await
                    .expect("list tools failed");
                tracing::info!("Available tools: {tools:#?}");

                assert_eq!(tools.tools.len(), 1);
                assert_eq!(tools.tools[0].name, "run_encoder");

                let test_params = $req_type {
                    inputs: vec![
                        "This is a test.".to_string(),
                        "This is another test.".to_string(),
                    ],
                    metadata: None,
                };
                let tool_result = client
                    .call_tool(CallToolRequestParam {
                        name: "run_encoder".into(),
                        arguments: serde_json::json!(test_params).as_object().cloned(),
                    })
                    .await
                    .expect("call tool failed");
                tracing::info!("Tool result: {tool_result:#?}");
                let embeddings_response: $resp_type = serde_json::from_value(
                    tool_result
                        .structured_content
                        .expect("No structured content found"),
                )
                .expect("failed to parse tool result");
                assert_eq!(embeddings_response.results.len(), 2);
                client.cancel().await.expect("Error cancelling the agent");
                sender.send(()).expect("Error sending end of test signal");
            }
        }
    };
}

test_mcp_server_impl!(
    test_mcp_embedding,
    embedding_state,
    EmbeddingRequest,
    EmbeddingResponse
);

test_mcp_server_impl!(
    test_mcp_sentence_embedding,
    sentence_embedding_state,
    SentenceEmbeddingRequest,
    SentenceEmbeddingResponse
);

test_mcp_server_impl!(
    test_mcp_token_classification,
    token_classification_state,
    TokenClassificationRequest,
    TokenClassificationResponse
);

test_mcp_server_impl!(
    test_mcp_sequence_classification,
    sequence_classification_state,
    SequenceClassificationRequest,
    SequenceClassificationResponse
);

#[tokio::test]
#[test_log::test]
async fn test_mcp_servers() {
    self::test_mcp_embedding::test_mcp_embedding().await;
    tracing::info!("Testing embedding");
    self::test_mcp_sentence_embedding::test_mcp_sentence_embedding().await;
    tracing::info!("Testing sentence embedding");
    self::test_mcp_token_classification::test_mcp_token_classification().await;
    tracing::info!("Testing token classification");
    self::test_mcp_sequence_classification::test_mcp_sequence_classification().await;
    tracing::info!("Testing sequence classification");
}
