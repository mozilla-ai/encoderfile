pub mod embedding;
pub mod model;
pub mod sequence_classification;
pub mod token_classification;
pub mod tokenizer;
pub mod utils;

pub mod token_info {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
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
}
