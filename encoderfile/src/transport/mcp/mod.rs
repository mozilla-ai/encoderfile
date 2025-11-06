use crate::{
    common::EmbeddingRequest,
    services::embedding,
    error::ApiError::{self, InputError, InternalError, ConfigError},
    state::AppState,
};
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
        AnnotateAble,
        CallToolResult,
        ErrorCode,
        Implementation,
        ProtocolVersion,
        Resource,
        RawResource,
        ServerInfo,
        ServerCapabilities,
        InitializeRequestParam,
        InitializeResult
    },
    tool, tool_handler, tool_router
};

#[derive(Clone)]
pub struct Encoder {
    state: AppState,
    tool_router: ToolRouter<Encoder>,
}


impl From<ApiError> for McpError {
     fn from(api_error: ApiError) -> McpError {
        match api_error {
            InputError(str) => McpError {code: ErrorCode::INVALID_REQUEST, message: std::borrow::Cow::Borrowed(str), data: None},
            InternalError(str) => McpError {code: ErrorCode::INTERNAL_ERROR, message: std::borrow::Cow::Borrowed(str), data: None},
            ConfigError(str) => McpError {code: ErrorCode::INTERNAL_ERROR, message: std::borrow::Cow::Borrowed(str), data: None},
        }
     }
}

fn to_mcp_error(serde_err: serde_json::Error) -> McpError {
    return McpError {code: ErrorCode::INVALID_REQUEST, message: serde_err.to_string(), data: None};
}

#[tool_router]
impl Encoder {
    #[allow(dead_code)]
    pub fn new(state: AppState) -> Self {
        Self {
            state: state,
            tool_router: Self::tool_router(),
        }
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    // TODO The desc is maybe a candidate to export into elf too
    #[tool(description = "Run the embedded encoder")]
    fn run_encoder(&self, Parameters(object): Parameters<EmbeddingRequest>) -> Result<CallToolResult, McpError> {
        let response = embedding(object, &self.state)?;
        let result = CallToolResult::structured(serde_json::to_value(response).map_err(to_mcp_error));
        return Ok(result);
    }
}


#[tool_handler]
impl ServerHandler for Encoder {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_06_18,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("This server provides an embedded encoder to which a list of inputs can be sent.".to_string()),
        }
    }

    // Do we want to expose config via resources? I assume we don't?
    /*
    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text("str:////Users/to/some/path/", "cwd"),
                self._create_resource_text("memo://insights", "memo-name"),
            ],
            next_cursor: None,
        })
    }
    */
    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
            let initialize_headers = &http_request_part.headers;
            let initialize_uri = &http_request_part.uri;
            tracing::info!(?initialize_headers, %initialize_uri, "initialize from http server");
        }
        Ok(self.get_info())
    }
}

