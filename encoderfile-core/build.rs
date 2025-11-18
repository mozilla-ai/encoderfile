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
                "proto/sentence_embedding.proto",
            ],
            &[
                "proto/embedding",
                "proto/sequence_classification",
                "proto/token_classification",
                "proto/sentence_embedding",
            ],
        )?;

    Ok(())
}
