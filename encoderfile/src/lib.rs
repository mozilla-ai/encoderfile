#[cfg(not(tarpaulin_include))]
mod assets;
pub mod cli;
pub mod config;
pub mod error;
#[cfg(not(tarpaulin_include))]
#[rustfmt::skip]
pub mod generated;
pub mod grpc;
pub mod http;
pub mod inference;
pub mod middleware;
pub mod model;
pub mod server;
pub mod services;
pub mod state;
pub mod tokenizer;

pub use assets::get_banner;
