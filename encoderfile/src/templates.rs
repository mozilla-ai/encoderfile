use lazy_static::lazy_static;
use tera::Tera;

pub mod transforms {
    macro_rules! embed_transform_template {
        ($name:ident, $path:expr) => {
            pub const $name: &'static str =
                include_str!(concat!("../templates/transforms/", $path, ".lua"));
        };
    }

    embed_transform_template!(EMBEDDING, "embedding");
    embed_transform_template!(SEQUENCE_CLASSIFICATION, "sequence_classification");
    embed_transform_template!(TOKEN_CLASSIFICATION, "token_classification");
    embed_transform_template!(SENTENCE_EMBEDDING, "sentence_embedding");
}

const MAIN_RS: &str = include_str!("../templates/main.rs.tera");
const CARGO_TOML: &str = include_str!("../templates/Cargo.toml.tera");

lazy_static! {
    pub static ref TEMPLATES: Tera = {
        let mut tera = Tera::default();

        tera.add_raw_template("main.rs.tera", MAIN_RS)
            .expect("failed to load main.rs.tera");

        tera.add_raw_template("Cargo.toml.tera", CARGO_TOML)
            .expect("failed to load Cargo.toml.tera");

        tera
    };
}
