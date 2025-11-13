const BUILD_VARS: &[&str] = &[
    "MODEL_WEIGHTS_PATH",
    "TOKENIZER_PATH",
    "MODEL_CONFIG_PATH",
    "MODEL_TYPE",
    "MODEL_NAME",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    tonic_prost_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(true)
        // .out_dir("src/generated")
        .compile_protos(
            &[
                "proto/embedding.proto",
                "proto/sequence_classification.proto",
                "proto/token_classification.proto",
            ],
            &[
                "proto/embedding",
                "proto/sequence_classification",
                "proto/token_classification",
            ],
        )?;

    for var in BUILD_VARS {
        let val = std::env::var(var).expect("Missing required environment variable: {var}");

        println!("cargo:rustc-env={var}={val}");
    }

    Ok(())
}
