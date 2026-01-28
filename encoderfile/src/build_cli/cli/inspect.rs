use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use serde::Serialize;
use serde_json::to_string_pretty;

use anyhow::Result;

use crate::{
    runtime::load_assets,
    common::{ModelConfig, Config},
};

// inspect struct with info

#[derive(Debug, Serialize)]
pub struct InspectInfo {
    pub model_config: ModelConfig,
    pub encoderfile_config: Config,
}


pub fn inspect_encoderfile(path_str: String) -> Result<String> {
    let file = File::open(Path::new(&path_str))?;
    let mut file = BufReader::new(file);
    let mut loader = load_assets(&mut file)?;

    let config = loader.encoderfile_config()?;
    let model_config = loader.model_config()?;
    
    Ok(to_string_pretty(&InspectInfo { model_config, encoderfile_config: config })?)
}
