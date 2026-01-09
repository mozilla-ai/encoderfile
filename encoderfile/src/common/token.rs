use schemars::JsonSchema;

#[derive(Debug, serde::Serialize, serde::Deserialize, utoipa::ToSchema, JsonSchema)]
pub struct TokenInfo {
    pub token: String,
    pub token_id: u32,
    pub start: usize,
    pub end: usize,
}
