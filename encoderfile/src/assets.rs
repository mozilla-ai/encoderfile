macro_rules! embed_in_section {
    ($name:ident, $path:expr, $section:expr, Bytes) => {
        #[cfg(target_os = "macos")]
        #[unsafe(link_section = concat!("__DATA,", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: [u8; include_bytes!(env!($path)).len()] = *include_bytes!(env!($path));

        #[cfg(target_os = "linux")]
        #[unsafe(link_section = concat!(".", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: [u8; include_bytes!(env!($path)).len()] = *include_bytes!(env!($path));

        #[cfg(target_os = "windows")]
        #[unsafe(link_section = concat!(".rdata$", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: [u8; include_bytes!(env!($path)).len()] = *include_bytes!(env!($path));
    };

    ($name:ident, $path:expr, $section:expr, String) => {
        #[cfg(target_os = "macos")]
        #[unsafe(link_section = concat!("__DATA,", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: &str = include_str!(env!($path));

        #[cfg(target_os = "linux")]
        #[unsafe(link_section = concat!(".", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: &str = include_str!(env!($path));

        #[cfg(target_os = "windows")]
        #[unsafe(link_section = concat!(".rdata$", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: &str = include_str!(env!($path));
    };

    ($name:ident, $path:expr, $section:expr, Env) => {
        #[cfg(target_os = "macos")]
        #[unsafe(link_section = concat!("__DATA,", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: &str = env!($path);

        #[cfg(target_os = "linux")]
        #[unsafe(link_section = concat!(".", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: &str = env!($path);

        #[cfg(target_os = "windows")]
        #[unsafe(link_section = concat!(".rdata$", $section))]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: &str = env!($path);
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

pub const BANNER: &'static str = include_str!("../../assets/banner.txt");

pub fn get_banner() -> String {
    let model_id_len = MODEL_ID.len();
    let signature = " | Mozilla.ai";
    let total_len: usize = 65;
    let remaining_len = total_len - model_id_len - signature.len();

    let spaces = " ".repeat(remaining_len);

    format!(
        "{}\nModel ID: {}{}{}\n",
        BANNER, MODEL_ID, spaces, signature
    )
}

pub fn get_model_id() -> String {
    MODEL_ID.to_string()
}
