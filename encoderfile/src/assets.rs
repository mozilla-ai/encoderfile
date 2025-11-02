pub const MODEL_TYPE_STR: &str = env!("MODEL_TYPE");
pub const MODEL_WEIGHTS: &[u8] = include_bytes!(env!("MODEL_WEIGHTS_PATH"));
pub const TOKENIZER_JSON: &str = include_str!(env!("TOKENIZER_PATH"));
pub const MODEL_CONFIG_JSON: &str = include_str!(env!("MODEL_CONFIG_PATH"));

pub const BANNER: &'static str = include_str!("../../assets/banner.txt");
const MODEL_ID: &'static str = env!("MODEL_NAME");

pub fn get_banner() -> String {
    let model_id_len = MODEL_ID.len();
    let signature = " | Mozilla.ai";
    let total_len: usize = 65;
    let remaining_len = total_len - model_id_len - signature.len();

    let spaces = " ".repeat(remaining_len);

    format!(
        "{}\nModel ID: {}{}{}\n",
        BANNER, MODEL_ID, spaces, signature
    )
}

pub fn get_model_id() -> String {
    MODEL_ID.to_string()
}
