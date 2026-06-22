fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    tonic_prost_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(true)
        .build_client(true)
        // .out_dir("src/generated")
        .compile_protos(
            &[
                "proto/embedding.proto",
                "proto/sequence_classification.proto",
                "proto/token_classification.proto",
                "proto/sentence_embedding.proto",
                "proto/image_classification.proto",
                "proto/manifest.proto",
                "proto/image_types.proto",
            ],
            &[
                "proto/embedding",
                "proto/sequence_classification",
                "proto/token_classification",
                "proto/sentence_embedding",
                "proto/image_classification",
                "proto/manifest",
                "proto/image_types",
            ],
        )?;

    Ok(())
}
