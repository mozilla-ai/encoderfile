use lazy_static::lazy_static;
use tera::Tera;

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
