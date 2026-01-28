use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use serde_json::to_string_pretty;

use anyhow::Result;

use crate::{
    runtime::load_assets,
};

// inspect struct with info

pub fn inspect_encoderfile(path_str: String) -> Result<()> {
    let path = Path::new(&path_str);
    let file = File::open(path)?;
    let mut file = BufReader::new(file);
    let mut loader = load_assets(&mut file)?;
    let config = loader.encoderfile_config()?;
    let model_config = loader.model_config()?;
    let transform = loader.transform()?;
    println!("Encoderfile Config:\n{}", to_string_pretty(&config)?);
    println!("Model Config:\n{}", to_string_pretty(&model_config)?);
    println!("Transform:\n{}", transform.unwrap_or("<None>".to_string()));

    Ok(())
}