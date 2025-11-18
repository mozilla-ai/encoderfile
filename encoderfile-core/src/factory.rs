#[macro_export]
macro_rules! embed_in_section {
    ($name:ident, $path:expr, $section:expr, Bytes) => {
        $crate::embed_in_section!(
            $name,
            [u8; include_bytes!($path).len()],
            $section,
            *include_bytes!($path)
        );
    };

    ($name:ident, $path:expr, $section:expr, String) => {
        $crate::embed_in_section!($name, &str, $section, include_str!($path));
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr) => {
        $crate::embed_in_section!($name, $dtype, concat!("__DATA,", $section), $res, "macos");
        $crate::embed_in_section!($name, $dtype, concat!(".", $section), $res, "linux");
        $crate::embed_in_section!($name, $dtype, concat!(".rdata$", $section), $res, "windows");
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr, $target_os:expr) => {
        #[cfg(target_os = $target_os)]
        #[unsafe(link_section = $section)]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: $dtype = $res;
    };
}

#[macro_export]
macro_rules! factory {
    {
        $model_weights_path:expr,
        $tokenizer_path:expr,
        $model_config_path:expr,
        $model_type:expr,
        $model_id:expr,
        $transform:expr,
    } => {
        mod assets {
            $crate::embed_in_section!(MODEL_WEIGHTS, $model_weights_path, "model_weights", Bytes);
            $crate::embed_in_section!(TOKENIZER_JSON, $tokenizer_path, "tokenizer", String);
            $crate::embed_in_section!(MODEL_CONFIG_JSON, $model_config_path, "model_config", String);

            pub const MODEL_TYPE_STR: &'static str = $model_type;
            pub const MODEL_ID: &'static str = $model_id;

            #[allow(dead_code)]
            pub const TRANSFORM: Option<&'static str> = $transform;
        }
    };
}
