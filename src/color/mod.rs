mod rgb;
pub use rgb::RGB;

pub fn rgb(r: f64, g: f64, b: f64) -> RGB {
    RGB::new(r, g, b)
}

mod rgbf;
pub use rgbf::RGBf;

pub fn rgbf(r: f32, g: f32, b: f32) -> RGBf {
    RGBf::new(r, g, b)
}

mod rgba;
pub use rgba::RGBA;

pub fn rgba(r: f64, g: f64, b: f64, a: f64) -> RGBA {
    RGBA::new(r, g, b, a)
}

mod rgbaf;
pub use rgbaf::RGBAf;

pub fn rgbaf(r: f32, g: f32, b: f32, a: f32) -> RGBAf {
    RGBAf::new(r, g, b, a)
}

mod rgb24;
pub use rgb24::RGB24;

pub fn rgb24(r: u8, g: u8, b: u8) -> RGB24 {
    RGB24::new(r, g, b)
}

mod rgba32;
pub use rgba32::RGBA32;

pub fn rgba32(r: u8, g: u8, b: u8, a: u8) -> RGBA32 {
    RGBA32::new(r, g, b, a)
}
