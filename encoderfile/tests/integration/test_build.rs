use anyhow::{Context, Result, bail};

use encoderfile::build_cli::cli::GlobalArguments;
use std::{
    fs, io,
    path::Path,
    process::{Child, Command},
    thread::sleep,
    time::{Duration, Instant},
};
use tempfile::tempdir;

const BINARY_NAME: &str = "test.encoderfile";

fn config(model_path: &Path, output_path: &Path) -> String {
    format!(
        r##"
encoderfile:
  name: test-model
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
        model_path, output_path
    )
}

const MODEL_ASSETS_PATH: &str = "../models/token_classification";

#[test]
fn test_build_encoderfile() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path();

    let tmp_model_path = path.join("models").join("token_classification");

    let base_binary_path = fs::canonicalize("../target/debug/encoderfile-runtime")?;
    let ef_config_path = fs::canonicalize(path)?.join("encoderfile.yml");
    let encoderfile_path = fs::canonicalize(path)?.join(BINARY_NAME);

    // copy model assets to temp dir
    copy_dir_all(MODEL_ASSETS_PATH, tmp_model_path.as_path())?;

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

    fs::copy(
        "../target/debug/encoderfile-runtime",
        base_binary_path.as_path(),
    )
    .expect("Failed to write base binary");

    // write encoderfile config
    let config = config(tmp_model_path.as_path(), encoderfile_path.as_path());

    fs::write(ef_config_path.as_path(), config.as_bytes())
        .expect("Failed to write encoderfile config");

    let build_args =
        encoderfile::build_cli::cli::test_build_args(ef_config_path.as_path(), base_binary_path);

    // build encoderfile
    let global_args = GlobalArguments::default();

    build_args.run(&global_args)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(encoderfile_path.as_path())
            .expect("Failed to get path for built encoderfile")
            .permissions();

        perms.set_mode(0o755);
        fs::set_permissions(encoderfile_path.as_path(), perms).expect("Failed to set permissions");
    }

    // serve encoderfile
    let mut child = spawn_encoderfile(
        encoderfile_path
            .to_str()
            .expect("Failed to create encoderfile binary path"),
    )?;

    wait_for_http("http://localhost:8080/health", Duration::from_secs(10))?;

    child.kill()?;
    child.wait().ok();

    Ok(())
}

fn wait_for_http(url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let start = Instant::now();

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!("server did not become ready in time");
        }

        if let Ok(resp) = client.get(url).send()
            && resp.status().is_success()
        {
            return Ok(());
        }

        sleep(Duration::from_millis(200));
    }
}

fn spawn_encoderfile(path: &str) -> Result<Child> {
    Command::new(path)
        .arg("serve")
        .arg("--disable-grpc")
        .arg("--http-port")
        .arg("8080")
        .spawn()
        .context("failed to spawn encoderfile process")
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());

        if ty.is_dir() {
            copy_dir_all(entry.path(), dest_path)?;
        } else {
            fs::copy(entry.path(), dest_path)?;
        }
    }

    Ok(())
}
