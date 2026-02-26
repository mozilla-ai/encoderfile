use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::Serialize;
use serde_json::to_string_pretty;

use anyhow::Result;

use crate::{
    common::{Config, ModelConfig},
    runtime::load_assets,
};

// inspect struct with info

#[derive(Debug, Serialize)]
pub struct InspectInfo {
    pub model_config: ModelConfig,
    pub encoderfile_config: Config,
}

pub fn inspect_encoderfile(path_str: &String) -> Result<String> {
    let file = File::open(Path::new(&path_str))?;
    let mut file = BufReader::new(file);
    let mut loader = load_assets(&mut file)?;

    let config = loader.encoderfile_config()?;
    let model_config = loader.model_config()?;

    Ok(to_string_pretty(&InspectInfo {
        model_config,
        encoderfile_config: config,
    })?)
}
