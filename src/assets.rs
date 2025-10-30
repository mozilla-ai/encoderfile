pub const MODEL_TYPE_STR: &str = env!("MODEL_TYPE");
pub const MODEL_WEIGHTS: &[u8] = include_bytes!(env!("MODEL_WEIGHTS_PATH"));
pub const TOKENIZER_JSON: &str = include_str!(env!("TOKENIZER_PATH"));
pub const MODEL_CONFIG_JSON: &str = include_str!(env!("MODEL_CONFIG_PATH"));

pub const BANNER: &'static str = include_str!("../assets/banner.txt");
pub const MODEL_ID: &'static str = env!("MODEL_NAME");
