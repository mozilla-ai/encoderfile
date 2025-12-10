use crate::common::model_type::ModelTypeSpec;
use crate::runtime::AppState;
use rmcp::ServerHandler;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

mod error;

pub trait McpRouter: ModelTypeSpec + 'static {
    type Tool: ServerHandler;
    const NEW_TOOL: fn(AppState<Self>) -> Self::Tool;

    fn router(state: AppState<Self>) -> axum::Router
    where
        <Self as McpRouter>::Tool: rmcp::ServerHandler,
    {
        let service = StreamableHttpService::new(
            move || Ok(Self::NEW_TOOL(state.clone())),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        axum::Router::new().nest_service("/mcp", service)
    }
}

// TODO figure out the lifetimes of a state so a ref can be safely passed
pub fn make_router<T: ModelTypeSpec>(state: AppState<T>) -> axum::Router
where
    T: McpRouter,
{
    T::router(state)
}

macro_rules! generate_mcp {
    ($model_type:ident, $tool_name:ident, $fn_name:ident, $request_body:ident, $return_model:ident, $short_desc:literal, $long_desc:literal) => {
        mod $fn_name {
            use crate::services::Inference;
            use $crate::common::$request_body;
            use $crate::runtime::AppState;
            use $crate::transport::mcp::error::to_mcp_error;
            use rmcp::{
                ErrorData as McpError,
                ServerHandler,
                RoleServer,
                handler::server::{
                    router::tool::ToolRouter,
                    wrapper::Parameters
                },
                service::RequestContext,
                model::{
                    CallToolResult,
                    Implementation,
                    ProtocolVersion,
                    ServerInfo,
                    ServerCapabilities,
                    InitializeRequestParam,
                    InitializeResult
                },
                tool, tool_handler, tool_router
            };

            #[derive(Clone)]
            pub struct $tool_name {
                state: AppState<$crate::common::model_type::$model_type>,
                tool_router: ToolRouter<$tool_name>,
            }

            impl super::McpRouter for $crate::common::model_type::$model_type {
                type Tool = $tool_name;
                const NEW_TOOL: fn(AppState<Self>) -> Self::Tool = Self::Tool::new;
            }

            #[tool_router]
            impl $tool_name {
                pub fn new(state: AppState<crate::common::model_type::$model_type>) -> Self {
                    Self {
                        state,
                        tool_router: Self::tool_router(),
                    }
                }

                #[tool(description = $short_desc)]
                fn run_encoder(&self, Parameters(object): Parameters<$request_body>) -> Result<CallToolResult, McpError> {
                    let response = self.state.inference(object)?;
                    let result = CallToolResult::structured(serde_json::to_value(response).map_err(to_mcp_error)?);
                    Ok(result)
                }
            }
            #[tool_handler]
            impl ServerHandler for $tool_name {
                fn get_info(&self) -> ServerInfo {
                    ServerInfo {
                        protocol_version: ProtocolVersion::V_2025_06_18,
                        capabilities: ServerCapabilities::builder()
                            .enable_tools()
                            .build(),
                        server_info: Implementation::from_build_env(),
                        instructions: Some($long_desc.to_string()),
                    }
                }

                async fn initialize(
                    &self,
                    _request: InitializeRequestParam,
                    context: RequestContext<RoleServer>,
                ) -> Result<InitializeResult, McpError> {
                    if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
                        let initialize_headers = &http_request_part.headers;
                        let initialize_uri = &http_request_part.uri;
                        tracing::info!(?initialize_headers, %initialize_uri, "initialize mcp server");
                    }
                    Ok(self.get_info())
                }
            }
        }
    };
}

generate_mcp!(
    Embedding,
    EmbedderTool,
    embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    "Performs embeddings for input text sequences.",
    "This tool will provide a vector embedding for the input text sequence."
);

generate_mcp!(
    SequenceClassification,
    SequenceClassificationTool,
    sequence_classification,
    SequenceClassificationRequest,
    SequenceClassificationResponse,
    "Performs sequence classification of input text sequences.",
    "This tool will classify an input text sequence."
);

generate_mcp!(
    TokenClassification,
    TokenClassificationTool,
    token_classification,
    TokenClassificationRequest,
    TokenClassificationResponse,
    "Performs token classification of input text sequences.",
    "This tool will classify each token of an input text sequence."
);

generate_mcp!(
    SentenceEmbedding,
    SentenceEmbeddingTool,
    sentence_embedding,
    SentenceEmbeddingRequest,
    SentenceEmbeddingResponse,
    "Performs sentence embedding of input text sequences.",
    "This tool will embed a sequence of texts."
);
