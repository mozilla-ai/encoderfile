fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    tonic_prost_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(true)
        .out_dir("src/generated")
        .compile_protos(&["proto/encoderfile.proto"], &["proto/encoderfile"])?;

    println!(
        "cargo:rustc-env=MODEL_WEIGHTS_PATH={}",
        std::env::var("MODEL_WEIGHTS_PATH").unwrap()
    );
    println!(
        "cargo:rustc-env=TOKENIZER_PATH={}",
        std::env::var("TOKENIZER_PATH").unwrap()
    );
    println!(
        "cargo:rustc-env=MODEL_CONFIG_PATH={}",
        std::env::var("MODEL_CONFIG_PATH").unwrap()
    );
    println!("cargo:rustc-env=MODEL_TYPE={}", std::env::var("MODEL_TYPE").unwrap());
    println!("cargo:rustc-env=MODEL_NAME={}", std::env::var("MODEL_NAME").unwrap());

    Ok(())
}
