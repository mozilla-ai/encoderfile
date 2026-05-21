mod engine;
mod tensor;
mod image;
mod utils;

pub use engine::*;
pub use tensor::Tensor;
pub use image::Image;

pub const DEFAULT_LIBS: [mlua::StdLib; 3] = [
    mlua::StdLib::TABLE,
    mlua::StdLib::STRING,
    mlua::StdLib::MATH,
];
