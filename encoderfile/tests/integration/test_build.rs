use anyhow::{Context, Result, bail};

use encoderfile::build_cli::cli::GlobalArguments;
use std::{
    fs,
    path::Path,
    process::{Child, Command},
    thread::sleep,
    time::{Duration, Instant},
};
use tempfile::tempdir;

use encoderfile::common;

tonic::include_proto!("encoderfile.metadata");

use encoderfile::generated::token_classification;

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

#[tokio::test]
async fn test_build_encoderfile() -> Result<()> {
    let dir = tempdir()?;
    let path = dir
        .path()
        .canonicalize()
        .expect("Failed to canonicalize temp path");

    let tmp_model_path = path.join("models").join("token_classification");

    let ef_config_path = path.join("encoderfile.yml");
    let encoderfile_path = path.join(BINARY_NAME);

    let http_port = "8080";
    let grpc_port = "9090";
    let sample_text =
        "Hugging Face is a technology company based in New York and Paris.".to_string();

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
    let config = config(
        &String::from("test-model"),
        tmp_model_path.as_path(),
        // use relative directory here to test the working_dir option
        Path::new(".").join(BINARY_NAME).as_path(), // encoderfile_path.as_path(),
    );

    fs::write(ef_config_path.as_path(), config.as_bytes())
        .expect("Failed to write encoderfile config");

    // make dummy dir and cd into it to test the working_dir option
    let dummy_dir = path.join("dummy");
    fs::create_dir(&dummy_dir).expect("Failed to create dummy directory");
    std::env::set_current_dir(&dummy_dir)
        .expect("Failed to change current directory to dummy directory");

    // without to working_dir option, the previous dir change would result
    // in the encoderfile being generated at the wrong path because of the rel output
    let build_args = encoderfile::build_cli::cli::test_build_args_working_dir(
        ef_config_path.as_path(),
        base_binary_path,
        &path,
    );

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

    // serve encoderfile
    let mut child = spawn_encoderfile(
        encoderfile_path
            .to_str()
            .expect("Failed to create encoderfile binary path"),
        http_port,
        grpc_port,
    )?;

    wait_for_http(
        format!("http://localhost:{http_port}/health").as_str(),
        Duration::from_secs(10),
    )
    .await?;
    send_http_inference(&sample_text, http_port.to_string()).await?;
    send_grpc_inference(&sample_text, grpc_port.to_string()).await?;

    child.kill()?;
    child.wait().ok();

    Ok(())
}

async fn wait_for_http(url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::new();
    let start = Instant::now();

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!("server did not become ready in time");
        }

        if let Ok(resp) = client.get(url).send().await
            && resp.status().is_success()
        {
            return Ok(());
        }

        sleep(Duration::from_millis(200));
    }
}

async fn send_http_inference(sample_text: &str, http_port: String) -> Result<()> {
    let client = reqwest::Client::new();
    let req = common::TokenClassificationRequest {
        inputs: vec![sample_text.to_owned()],
        metadata: None,
    };
    client
        .post(format!("http://localhost:{http_port}/predict"))
        .json(&req)
        .send()
        .await?;
    Ok(())
}

async fn send_grpc_inference(sample_text: &str, grpc_port: String) -> Result<()> {
    let mut client = token_classification::token_classification_inference_client::TokenClassificationInferenceClient::connect(format!("http://[::]:{grpc_port}/predict")).await?;
    let req = token_classification::TokenClassificationRequest {
        inputs: vec![sample_text.to_owned()],
        metadata: std::collections::HashMap::new(),
    };
    client.predict(req).await?;
    Ok(())
}

fn spawn_encoderfile(path: &str, http_port: &str, grpc_port: &str) -> Result<Child> {
    Command::new(path)
        .arg("serve")
        .arg("--grpc-port")
        .arg(grpc_port)
        .arg("--http-port")
        .arg(http_port)
        .spawn()
        .context("failed to spawn encoderfile process")
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
