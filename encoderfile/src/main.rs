use anyhow::Result;
use lazy_static::lazy_static;
use tera::Tera;

mod config;

lazy_static! {
    pub static ref TEMPLATES: Tera = {
        let tera = match Tera::new("encoderfile/templates/*") {
            Ok(t) => t,
            Err(e) => {
                println!("Parsing error(s): {}", e);
                ::std::process::exit(1);
            }
        };
        // tera.autoescape_on(vec![".html", ".sql"]);
        // tera.register_filter("do_nothing", do_nothing_filter);
        tera
    };
}

fn main() -> Result<()> {
    let path = std::path::PathBuf::from("test_config.yml");

    let config = config::Config::load(&path)?;
    println!("{:?}", config);

    let ctx = config.encoderfile.to_tera_ctx()?;

    println!("{:?}", ctx);

    println!(
        "{:?}",
        TEMPLATES.get_template_names().collect::<Vec<&str>>()
    );

    let rendered = TEMPLATES.render("main.rs.tera", &ctx)?;

    std::fs::write("encoderfile-test/src/main.rs", rendered)?;

    Ok(())
}
