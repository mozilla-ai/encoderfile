mod assets;
pub mod cli;
pub mod config;
pub mod error;
#[cfg(not(tarpaulin_include))]
pub mod generated;
pub mod grpc;
pub mod http;
pub mod inference;
pub mod server;
pub mod services;
pub mod state;

pub use assets::get_banner;
