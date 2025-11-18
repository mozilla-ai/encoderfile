pub const BANNER: &str = include_str!("../../assets/banner.txt");

pub fn get_banner(model_id: &str) -> String {
    let model_id_len = model_id.len();
    let signature = " | Mozilla.ai";
    let total_len: usize = 65;
    let remaining_len = total_len - model_id_len - signature.len();

    let spaces = " ".repeat(remaining_len);

    format!("{BANNER}\nModel ID: {model_id}{spaces}{signature}\n")
}
