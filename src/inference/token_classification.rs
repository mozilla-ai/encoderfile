#[derive(Debug, serde::Serialize)]
pub struct TokenClassificationResult {
    pub tokens: Vec<TokenClassification>
}

impl From<TokenClassificationResult> for crate::generated::token_classification::TokenClassificationResult {
    fn from(val: TokenClassificationResult) -> Self {
        Self {
            tokens: val.tokens.into_iter().map(|i| i.into()).collect()
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct TokenClassification {
    pub token_id: u32,
    pub token: String,
    pub start: u32,
    pub end: u32,
    pub logits: Option<Vec<f32>>,
    pub scores: Vec<f32>,
    pub label: String,
    pub score: f32,
}

impl From<TokenClassification> for crate::generated::token_classification::TokenClassification {
    fn from(val: TokenClassification) -> Self {
        Self {
            token_id: val.token_id,
            token: val.token,
            start: val.start,
            end: val.end,
            logits: val.logits.unwrap_or(Vec::new()),
            scores: val.scores,
            label: val.label,
            score: val.score
        }
    }
}
