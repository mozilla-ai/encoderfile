use anyhow::{Context, Result, bail};

use encoderfile::build_cli::cli::{
    GlobalArguments,
    inspect_encoderfile,
};
use std::{
    fs,
    path::Path,
    process::{Command},
};
use tempfile::tempdir;

const BINARY_NAME: &str = "test.encoderfile";

fn config(model_name: &String, model_path: &Path, output_path: &Path) -> String {
    format!(
        r##"
encoderfile:
  name: {:?}
  path: {:?}
  model_type: token_classification
  output_path: {:?}
  transform: |
    --- Applies a softmax across token classification logits.
    --- Each token classification is normalized independently.
    --- 
    --- Args:
    ---   arr (Tensor): A tensor of shape [batch_size, n_tokens, n_labels].
    ---                 The softmax is applied along the third axis (n_labels).
    ---
    --- Returns:
    ---   Tensor: The input tensor with softmax-normalized embeddings.
    ---@param arr Tensor
    ---@return Tensor
    function Postprocess(arr)
        return arr:softmax(3)
    end
        "##,
        model_name, model_path, output_path
    )
}

const MODEL_ASSETS_PATH: &str = "../models/token_classification";

#[test]
fn test_inspect_encoderfile() -> Result<()> {
    let dir = tempdir()?;
    let path = dir
        .path()
        .canonicalize()
        .expect("Failed to canonicalize temp path");

    let tmp_model_path = path.join("models").join("token_classification");

    let ef_config_path = path.join("encoderfile.yml");
    let encoderfile_path = path.join(BINARY_NAME);
    let model_name = String::from("some-custom-name");

    // copy model assets to temp dir
    copy_dir_all(MODEL_ASSETS_PATH, tmp_model_path.as_path())
        .expect("Failed to copy model assets to temp directory");

    if !tmp_model_path.join("model.onnx").exists() {
        bail!(
            "Path {:?} does not exist",
            tmp_model_path.join("model.onnx")
        );
    }

    // compile base binary and copy to temp dir
    let _ = Command::new("cargo")
        .args(["build", "-p", "encoderfile-runtime"])
        .status()
        .expect("Failed to build encoderfile-runtime");

    let base_binary_path = fs::canonicalize("../target/debug/encoderfile-runtime")
        .expect("Failed to canonicalize base binary path");

    // write encoderfile config
    let config = config(&model_name, tmp_model_path.as_path(), encoderfile_path.as_path());

    fs::write(ef_config_path.as_path(), config.as_bytes())
        .expect("Failed to write encoderfile config");

    let build_args =
        encoderfile::build_cli::cli::test_build_args(ef_config_path.as_path(), base_binary_path);

    // build encoderfile
    let global_args = GlobalArguments::default();

    build_args
        .run(&global_args)
        .context("Failed to build encoderfile")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(encoderfile_path.as_path())
            .expect("Failed to get path for built encoderfile")
            .permissions();

        perms.set_mode(0o755);
        fs::set_permissions(encoderfile_path.as_path(), perms).expect("Failed to set permissions");
    }

    let inspect_output = inspect_encoderfile(
        String::from(
            encoderfile_path.as_os_str().to_str().ok_or_else(
                || anyhow::anyhow!("Encoderfile path name failed to convert to string"))?
        )
    )?;

    let inspect_output_json = serde_json::from_str::<serde_json::Value>(&inspect_output)
        .context("Failed to parse inspect output as JSON")?;
    inspect_output_json
        .get("encoderfile_config")
        .and_then(|efc| efc.get("name"))
        .and_then(|name| name.as_str())
        .filter(|name_str| *name_str == model_name.as_str())
        .ok_or_else(|| anyhow::anyhow!("Model name in inspect output does not match expected"))?;

    Ok(())
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> anyhow::Result<()> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    fs::create_dir_all(dst).context(format!("Failed to create directory {:?}", &dst))?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());

        if ty.is_dir() {
            copy_dir_all(entry.path(), dest_path.as_path()).context(format!(
                "Failed to copy {:?} to {:?}",
                entry.path(),
                dest_path.as_path()
            ))?;
        } else {
            fs::copy(entry.path(), dest_path.as_path()).context(format!(
                "Failed to copy {:?} to {:?}",
                entry.path(),
                dest_path.as_path()
            ))?;
        }
    }

    Ok(())
}
