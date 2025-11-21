#[cfg(not(tarpaulin_include))]
mod assets;
#[cfg(feature = "transport")]
pub mod cli;
pub mod common;
pub mod error;
#[cfg(not(tarpaulin_include))]
#[rustfmt::skip]
#[cfg(feature = "transport")]
pub mod generated;
pub mod factory;
#[cfg(feature = "runtime")]
pub mod inference;
#[cfg(feature = "runtime")]
pub mod runtime;
#[cfg(feature = "transport")]
pub mod server;
#[cfg(feature = "runtime")]
pub mod services;
#[cfg(feature = "transforms")]
pub mod transforms;
#[cfg(feature = "transport")]
pub mod transport;

#[cfg(feature = "dev-utils")]
pub mod dev_utils;

pub use assets::get_banner;
#[cfg(feature = "runtime")]
pub use runtime::AppState;
