#[cfg(not(tarpaulin_include))]
mod assets;
pub mod cli;
pub mod common;
pub mod error;
#[cfg(not(tarpaulin_include))]
#[rustfmt::skip]
pub mod generated;
pub mod inference;
pub mod runtime;
pub mod server;
pub mod services;
pub mod state;
pub mod transport;

pub use assets::get_banner;
