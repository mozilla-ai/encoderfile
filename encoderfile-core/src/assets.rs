crate::factory! {
    env!("MODEL_WEIGHTS_PATH"),
    env!("TOKENIZER_PATH"),
    env!("MODEL_CONFIG_PATH"),
    env!("MODEL_TYPE"),
    env!("MODEL_NAME"),
    Some(include_str!("/Users/rbesaleli/encoderfile/transforms/embedding/l2_normalize_embeddings.lua")),
}

pub use assets::{MODEL_ID, TRANSFORM};
pub use config::{get_model_config, get_model_type};
pub use model::get_model;
pub use tokenizer::get_tokenizer;

pub const BANNER: &str = include_str!("../../assets/banner.txt");

pub fn get_banner() -> String {
    let model_id_len = MODEL_ID.len();
    let signature = " | Mozilla.ai";
    let total_len: usize = 65;
    let remaining_len = total_len - model_id_len - signature.len();

    let spaces = " ".repeat(remaining_len);

    format!("{BANNER}\nModel ID: {MODEL_ID}{spaces}{signature}\n")
}

pub fn get_model_id() -> String {
    MODEL_ID.to_string()
}
