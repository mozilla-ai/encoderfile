use crate::common::Config;
use std::sync::{Arc, OnceLock};

static CONFIG: OnceLock<Arc<Config>> = OnceLock::new();

pub fn get_config(config_str: &str) -> Arc<Config> {
    CONFIG
        .get_or_init(|| match serde_json::from_str::<Config>(config_str) {
            Ok(c) => Arc::new(c),
            Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
        })
        .clone()
}
