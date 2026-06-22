use super::Tensor;
use image::{DynamicImage, GenericImageView};
use mlua::prelude::*;
use ndarray::Array3;

const DEFAULT_FILTER_TYPE: image::imageops::FilterType = image::imageops::FilterType::Triangle;

#[derive(Debug, Clone)]
pub struct Image(pub DynamicImage);

impl Image {
    pub fn into_inner(&self) -> &DynamicImage {
        &self.0
    }
}

impl FromLua for Image {
    fn from_lua(value: LuaValue, _lua: &Lua) -> Result<Image, LuaError> {
        match value {
            LuaValue::UserData(data) => data.borrow::<Image>().map(|i| i.to_owned()),
            _ => Err(LuaError::external(
                format!("Unknown type: {}", value.type_name()).as_str(),
            )),
        }
    }
}

fn dyn_image_to_array3(image: &DynamicImage, num_channels: u32) -> Array3<f32> {
    // TODO num_channels is tied to the format we convert to
    let raw = image.to_rgb8().into_raw();
    let (h_us, w_us) = image.dimensions();
    let h_us: usize = h_us as usize;
    let w_us: usize = w_us as usize;
    let nc_us: usize = num_channels as usize;

    // Build CHW array directly from raw HWC bytes, avoiding an intermediate array and transpose.
    Array3::from_shape_fn((nc_us, h_us, w_us), |(c, y, x)| {
        raw[y * w_us * nc_us + x * nc_us + c] as f32
    })
}

fn resize_image(image: &DynamicImage, height: u32, width: u32) -> DynamicImage {
    image.resize_exact(width, height, DEFAULT_FILTER_TYPE)
}

impl LuaUserData for Image {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        // tensor ops
        methods.add_method("to_array", |_, this, num_channels| {
            Ok(Tensor(
                dyn_image_to_array3(this.into_inner(), num_channels).into_dyn(),
            ))
        });
        methods.add_method("resize", |_, this, (height, width)| {
            Ok(Image(resize_image(this.into_inner(), height, width)))
        });
    }
}

#[cfg(test)]
fn load_env() -> Lua {
    Lua::new()
}

#[test]
fn test_resize_image() {
    use image::GenericImageView;
    let img = image::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
    assert_ne!(img.dimensions(), (224, 224));
    let lua = load_env();
    let img_val = Image(img);
    lua.globals().set("img", img_val).unwrap();
    let resized: Image = lua.load("return img:resize(224, 224)").eval().unwrap();
    assert_eq!(resized.into_inner().dimensions(), (224, 224));
}

#[test]
fn test_image_to_array() {
    let img = image::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
    let lua = load_env();
    let img_val = Image(img);
    lua.globals().set("img", img_val).unwrap();
    let array: Tensor = lua
        .load("return img:resize(224,224):to_array(3)")
        .eval()
        .unwrap();
    assert_eq!(array.into_inner().shape(), &[3, 224, 224]);
}
