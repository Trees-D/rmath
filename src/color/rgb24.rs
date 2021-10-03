#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

use super::{RGBAf, RGBf, RGB, RGBA, RGBA32};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RGB24 {
    r: u8,
    g: u8,
    b: u8,
}

impl Display for RGB24 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RGB24(r: {}, g: {}, b: {})", self.r, self.g, self.b)
    }
}

impl Default for RGB24 {
    fn default() -> Self {
        Self { r: 0, g: 0, b: 0 }
    }
}

impl Index<usize> for RGB24 {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("`rmath::color::RGB24::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for RGB24 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("`rmath::color::RGB24::index_mut`: index out of bounds."),
        }
    }
}

impl From<f64> for RGB24 {
    fn from(rgb: f64) -> Self {
        let rgb = (rgb * 255.0) as u8;
        Self::new(rgb, rgb, rgb)
    }
}

impl From<(f64, f64, f64)> for RGB24 {
    fn from(rgb: (f64, f64, f64)) -> Self {
        let (r, g, b) = rgb;
        let r = (r * 255.0) as u8;
        let g = (g * 255.0) as u8;
        let b = (b * 255.0) as u8;
        Self::new(r, g, b)
    }
}

impl From<[f64; 3]> for RGB24 {
    fn from(rgb: [f64; 3]) -> Self {
        let r = (rgb[0] * 255.0) as u8;
        let g = (rgb[1] * 255.0) as u8;
        let b = (rgb[2] * 255.0) as u8;
        Self::new(r, g, b)
    }
}

impl From<u32> for RGB24 {
    fn from(rgb_: u32) -> Self {
        let r = ((rgb_ >> 16) & 0xff) as u8;
        let g = ((rgb_ >> 8) & 0xff) as u8;
        let b = (rgb_ & 0xff) as u8;
        Self::new(r, g, b)
    }
}

impl From<u8> for RGB24 {
    fn from(rgb: u8) -> Self {
        Self::new(rgb, rgb, rgb)
    }
}

impl From<(u8, u8, u8)> for RGB24 {
    fn from(rgb: (u8, u8, u8)) -> Self {
        let (r, g, b) = rgb;
        Self::new(r, g, b)
    }
}

impl From<[u8; 3]> for RGB24 {
    fn from(rgb: [u8; 3]) -> Self {
        Self::new(rgb[0], rgb[1], rgb[2])
    }
}

impl From<RGB> for RGB24 {
    fn from(color: RGB) -> Self {
        color.to_rgb24()
    }
}

impl From<RGBf> for RGB24 {
    fn from(color: RGBf) -> Self {
        color.to_rgb24()
    }
}

impl From<RGBA> for RGB24 {
    fn from(color: RGBA) -> Self {
        color.to_rgb24()
    }
}

impl From<RGBAf> for RGB24 {
    fn from(color: RGBAf) -> Self {
        color.to_rgb24()
    }
}

impl From<RGBA32> for RGB24 {
    fn from(color: RGBA32) -> Self {
        color.to_rgb24()
    }
}

impl RGB24 {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self::new(0, 0, 0)
    }

    pub fn white() -> Self {
        Self::new(255, 255, 255)
    }

    pub fn red() -> Self {
        Self::new(255, 0, 0)
    }

    pub fn green() -> Self {
        Self::new(0, 255, 0)
    }

    pub fn blue() -> Self {
        Self::new(0, 0, 255)
    }

    pub fn r(self) -> u8 {
        self.r
    }

    pub fn g(self) -> u8 {
        self.g
    }

    pub fn b(self) -> u8 {
        self.b
    }
}

impl RGB24 {
    pub fn sum(self) -> i32 {
        self.r as i32 + self.g as i32 + self.b as i32
    }

    pub fn gray(self) -> u8 {
        (self.sum() / 3) as u8
    }

    pub fn min_element(self) -> u8 {
        self.r.min(self.g).min(self.b)
    }

    pub fn max_element(self) -> u8 {
        self.r.max(self.g).max(self.b)
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        ruby_assert!(min.r <= max.r);
        ruby_assert!(min.g <= max.g);
        ruby_assert!(min.b <= max.b);

        self.min(max).max(min)
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.r.min(rhs.r), self.g.min(rhs.g), self.b.min(rhs.b))
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.r.max(rhs.r), self.g.max(rhs.g), self.b.max(rhs.b))
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::black(), Self::white())
    }
}

impl RGB24 {
    pub fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    pub fn to_tuple(self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }

    pub fn to_rgb(self) -> RGB {
        RGB::new(
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
        )
    }

    pub fn to_rgbf(self) -> RGBf {
        RGBf::new(
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
        )
    }

    pub fn to_rgba(self) -> RGBA {
        RGBA::new(
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
            1.0,
        )
    }

    pub fn to_rgba_alpha(self, alpha: f64) -> RGBA {
        RGBA::new(
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
            alpha,
        )
    }

    pub fn to_rgbaf(self) -> RGBAf {
        RGBAf::new(
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            1.0,
        )
    }

    pub fn to_rgbaf_alpha(self, alpha: f32) -> RGBAf {
        RGBAf::new(
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            alpha,
        )
    }

    pub fn to_rgba32(self) -> RGBA32 {
        RGBA32::new(self.r, self.g, self.b, 255)
    }

    pub fn to_rgba32_alpha(self, alpha: u8) -> RGBA32 {
        RGBA32::new(self.r, self.g, self.b, alpha)
    }
}
