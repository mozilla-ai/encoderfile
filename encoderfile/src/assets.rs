macro_rules! embed_in_section {
    ($name:ident, $path:expr, $section:expr, Bytes) => {
        embed_in_section!(
            $name,
            [u8; include_bytes!(env!($path)).len()],
            $section,
            *include_bytes!(env!($path))
        );
    };

    ($name:ident, $path:expr, $section:expr, String) => {
        embed_in_section!($name, &str, $section, include_str!(env!($path)));
    };

    ($name:ident, $path:expr, $section:expr, Env) => {
        embed_in_section!($name, &str, $section, env!($path));
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr) => {
        embed_in_section!($name, $dtype, concat!("__DATA,", $section), $res, "macos");
        embed_in_section!($name, $dtype, concat!(".", $section), $res, "linux");
        embed_in_section!($name, $dtype, concat!(".rdata$", $section), $res, "windows");
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr, $target_os:expr) => {
        #[cfg(target_os = $target_os)]
        #[unsafe(link_section = $section)]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: $dtype = $res;
    };
}

embed_in_section!(MODEL_WEIGHTS, "MODEL_WEIGHTS_PATH", "model_weights", Bytes);
embed_in_section!(TOKENIZER_JSON, "TOKENIZER_PATH", "model_tokenizer", String);
embed_in_section!(
    MODEL_CONFIG_JSON,
    "MODEL_CONFIG_PATH",
    "model_config",
    String
);
embed_in_section!(MODEL_TYPE_STR, "MODEL_TYPE", "model_type", Env);
embed_in_section!(MODEL_ID, "MODEL_NAME", "model_id", Env);

pub const BANNER: &str = include_str!("../../assets/banner.txt");

pub fn get_banner() -> String {
    let model_id_len = MODEL_ID.len();
    let signature = " | Mozilla.ai";
    let total_len: usize = 65;
    let remaining_len = total_len - model_id_len - signature.len();

    let spaces = " ".repeat(remaining_len);

    format!("{BANNER}\nModel ID: {MODEL_ID}{spaces}{signature}\n")
}

pub fn get_model_id() -> String {
    MODEL_ID.to_string()
}
