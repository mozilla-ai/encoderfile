use anyhow::Result;
use lazy_static::lazy_static;
use tera::Tera;

mod config;

lazy_static! {
    pub static ref TEMPLATES: Tera = {
        
        // tera.autoescape_on(vec![".html", ".sql"]);
        // tera.register_filter("do_nothing", do_nothing_filter);
        match Tera::new("encoderfile/templates/*") {
            Ok(t) => t,
            Err(e) => {
                println!("Parsing error(s): {}", e);
                ::std::process::exit(1);
            }
        }
    };
}

fn main() -> Result<()> {
    let path = std::path::PathBuf::from("test_config.yml");

    let config = config::Config::load(&path)?;

    let write_dir = config.encoderfile.get_write_dir();
    std::fs::create_dir_all(&write_dir)?;

    // create src/ and target/ directory
    std::fs::create_dir(write_dir.join("src/"))?;
    std::fs::create_dir(write_dir.join("target/"))?;

    let ctx = config.encoderfile.to_tera_ctx()?;

    render("main.rs.tera", &ctx, &write_dir, "src/main.rs")?;
    render("Cargo.toml.tera", &ctx, &write_dir, "Cargo.toml")?;

    if config.encoderfile.build {
        let cargo_toml_path = write_dir.join("Cargo.toml").canonicalize()?;

        let manifest_dir = cargo_toml_path.to_str().unwrap();

        std::process::Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(manifest_dir)
            .status()?;
    }

    Ok(())
}

fn render(template_name: &str, ctx: &tera::Context, write_dir: &std::path::PathBuf, out_path: &str) -> Result<()> {
    let rendered = TEMPLATES.render(template_name, ctx)?;

    let file = write_dir.join(out_path);

    std::fs::write(file, rendered)?;

    Ok(())
}
