use std::path::PathBuf;

fn main() {
    let encoderfile_core_dep_str = match env!("ENCODERFILE_DEV") {
        // include local path if in dev mode
        "true" => get_local_encoderfile_dep(),
        // otherwise use coupled version. encoderfile and encoderfile-core
        // should ALWAYS have the same version.
        "false" => get_versioned_encoderfile_dep(),
        _ => panic!("ENCODERFILE_DEV must either be \"true\" or \"false\""),
    };

    println!(
        "cargo:rustc-env=ENCODERFILE_CORE_DEP_STR={}",
        encoderfile_core_dep_str
    );
}

fn get_local_encoderfile_dep() -> String {
    let encoderfile_dir: PathBuf = PathBuf::from("../encoderfile-core")
        .canonicalize()
        .expect("Failed to find encoderfile-core directory. This should not happen.");

    format!("path = {:?}", encoderfile_dir.to_str().unwrap())
}

fn get_versioned_encoderfile_dep() -> String {
    format!("version = {}", env!("CARGO_PKG_VERSION"))
}
