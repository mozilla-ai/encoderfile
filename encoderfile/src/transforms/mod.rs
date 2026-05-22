mod engine;
mod image;
mod tensor;
mod utils;

pub use engine::*;
pub use image::Image;
pub use tensor::Tensor;

pub const DEFAULT_LIBS: [mlua::StdLib; 3] = [
    mlua::StdLib::TABLE,
    mlua::StdLib::STRING,
    mlua::StdLib::MATH,
];
