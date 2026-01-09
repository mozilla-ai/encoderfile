pub fn default_cache_dir() -> std::path::PathBuf {
    directories::ProjectDirs::from("com", "mozilla-ai", "encoderfile")
        .expect("Cannot locate")
        .cache_dir()
        .to_path_buf()
}
