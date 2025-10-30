mod assets;
pub mod cli;
pub mod config;
pub mod error;
pub mod generated;
pub mod grpc;
pub mod inference;
pub mod services;

pub use assets::{BANNER, MODEL_ID};

pub fn get_banner() -> String {
    let model_id_len = MODEL_ID.len();
    let signature = " | Mozilla.ai";
    let total_len: usize = 73;
    let remaining_len = total_len - model_id_len - signature.len();

    let spaces = " ".repeat(remaining_len);

    format!("{}\nModel ID: {}{}{}", BANNER, MODEL_ID, spaces, signature)
}
