fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    tonic_prost_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(true)
        .out_dir("src/generated")
        .compile_protos(&["proto/encoderfile.proto"], &["proto/encoderfile"])?;

    let onnx_weights_path = std::env::var("MODEL_WEIGHTS_PATH").unwrap();
    let tokenizer_path = std::env::var("TOKENIZER_PATH").unwrap();
    let model_config_path = std::env::var("MODEL_CONFIG_PATH").unwrap();
    let model_type = std::env::var("MODEL_TYPE").unwrap();

    println!("cargo:rustc-env=MODEL_WEIGHTS_PATH={}", &onnx_weights_path);
    println!("cargo:rustc-env=TOKENIZER_PATH={}", &tokenizer_path);
    println!("cargo:rustc-env=MODEL_CONFIG_PATH={}", &model_config_path);
    println!("cargo:rustc-env=MODEL_TYPE={}", &model_type);

    Ok(())
}
