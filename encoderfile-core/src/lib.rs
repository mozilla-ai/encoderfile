#[cfg(not(tarpaulin_include))]
mod assets;
pub mod cli;
pub mod common;
pub mod error;
#[cfg(not(tarpaulin_include))]
#[rustfmt::skip]
pub mod generated;
pub mod factory;
pub mod inference;
pub mod runtime;
pub mod server;
pub mod services;
pub mod transforms;
pub mod transport;

#[cfg(any(test, feature = "dev-utils"))]
pub mod dev_utils;

pub use assets::get_banner;
pub use runtime::AppState;
