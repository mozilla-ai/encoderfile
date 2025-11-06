use schemars::JsonSchema;

#[derive(Debug, serde::Serialize, serde::Deserialize, utoipa::ToSchema, JsonSchema)]
pub struct TokenInfo {
    pub token: String,
    pub token_id: u32,
    pub start: usize,
    pub end: usize,
}

impl From<TokenInfo> for crate::generated::token::TokenInfo {
    fn from(val: TokenInfo) -> Self {
        crate::generated::token::TokenInfo {
            token: val.token,
            token_id: val.token_id,
            start: (val.start as u32),
            end: (val.end as u32),
        }
    }
}
