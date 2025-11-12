use crate::{
    common::ModelType,
};
use crate::runtime::AppState;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService,
    session::local::LocalSessionManager,
};

// TODO figure out the lifetimes of a state so a ref can be safely passed
pub fn make_router(state: AppState) -> axum::Router {
    match state.model_type {
        ModelType::Embedding => {
            let service = StreamableHttpService::new(
                move || Ok(embedding::EmbedderTool::new(state.clone())),
                LocalSessionManager::default().into(),
                Default::default());
            axum::Router::new().nest_service("/mcp", service)
        }
        ModelType::SequenceClassification => {
            let service = StreamableHttpService::new(
                move || Ok(sequence_classification::SequenceClassificationTool::new(state.clone())),
                LocalSessionManager::default().into(),
                Default::default());
            axum::Router::new().nest_service("/mcp", service)
        }
        ModelType::TokenClassification => {
            let service = StreamableHttpService::new(
                move || Ok(token_classification::TokenClassificationTool::new(state.clone())),
                LocalSessionManager::default().into(),
                Default::default());
            axum::Router::new().nest_service("/mcp", service)
        }
    }
}

macro_rules! generate_mcp {
    ($tool_name:ident, $fn_name:ident, $request_body:ident, $return_model:ident, $short_desc:literal, $long_desc:literal) => {
        mod $fn_name {
            use $crate::services::$fn_name;
            use $crate::common::$request_body;
            use $crate::runtime::AppState;
            use $crate::error::to_mcp_error;
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
                state: AppState,
                tool_router: ToolRouter<$tool_name>,
            }

            #[tool_router]
            impl $tool_name {
                pub fn new(state: AppState) -> Self {
                    Self {
                        state,
                        tool_router: Self::tool_router(),
                    }
                }

                #[tool(description = $short_desc)]
                fn run_encoder(&self, Parameters(object): Parameters<$request_body>) -> Result<CallToolResult, McpError> {
                    let response = $fn_name(object, &self.state)?;
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
    EmbedderTool,
    embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    "Performs embeddings for input text sequences.",
    "This tool will provide a vector embedding for the input text sequence."
);

generate_mcp!(
    SequenceClassificationTool,
    sequence_classification,
    SequenceClassificationRequest,
    SequenceClassificationResponse,
    "Performs sequence classification of input text sequences.",
    "This tool will classify an input text sequence."
);

generate_mcp!(
    TokenClassificationTool,
    token_classification,
    TokenClassificationRequest,
    TokenClassificationResponse,
    "Performs token classification of input text sequences.",
    "This tool will classify each token of an input text sequence."
);