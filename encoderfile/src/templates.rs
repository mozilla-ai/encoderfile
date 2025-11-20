use lazy_static::lazy_static;
use tera::Tera;

const MAIN_RS: &'static str = include_str!("../templates/main.rs.tera");
const CARGO_TOML: &'static str = include_str!("../templates/Cargo.toml.tera");

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
