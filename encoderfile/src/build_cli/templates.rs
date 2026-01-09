pub mod transforms {
    macro_rules! embed_transform_template {
        ($name:ident, $path:expr) => {
            pub const $name: &'static str =
                include_str!(concat!("../../templates/transforms/", $path, ".lua"));
        };
    }

    embed_transform_template!(EMBEDDING, "embedding");
    embed_transform_template!(SEQUENCE_CLASSIFICATION, "sequence_classification");
    embed_transform_template!(TOKEN_CLASSIFICATION, "token_classification");
    embed_transform_template!(SENTENCE_EMBEDDING, "sentence_embedding");
}
