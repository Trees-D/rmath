#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

use super::{RGBAf, RGBf, RGB, RGB24, RGBA};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RGBA32 {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Display for RGBA32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RGBA32(r: {}, g: {}, b: {}, a: {})",
            self.r, self.g, self.b, self.a
        )
    }
}

impl Default for RGBA32 {
    fn default() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }
    }
}

impl Index<usize> for RGBA32 {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            3 => &self.a,
            _ => panic!("`rmath::color::RGBA32::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for RGBA32 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            3 => &mut self.a,
            _ => panic!("`rmath::color::RGBA32::index_mut`: index out of bounds."),
        }
    }
}

impl From<f64> for RGBA32 {
    fn from(rgb: f64) -> Self {
        let rgb = (rgb * 255.0) as u8;
        Self::new(rgb, rgb, rgb, rgb)
    }
}

impl From<(f64, f64, f64)> for RGBA32 {
    fn from(rgb: (f64, f64, f64)) -> Self {
        let (r, g, b) = rgb;
        let r = (r * 255.0) as u8;
        let g = (g * 255.0) as u8;
        let b = (b * 255.0) as u8;
        Self::new(r, g, b, 255)
    }
}

impl From<(f64, f64, f64, f64)> for RGBA32 {
    fn from(rgba: (f64, f64, f64, f64)) -> Self {
        let (r, g, b, a) = rgba;
        let r = (r * 255.0) as u8;
        let g = (g * 255.0) as u8;
        let b = (b * 255.0) as u8;
        let a = (a * 255.0) as u8;
        Self::new(r, g, b, a)
    }
}

impl From<[f64; 3]> for RGBA32 {
    fn from(rgb: [f64; 3]) -> Self {
        let r = (rgb[0] * 255.0) as u8;
        let g = (rgb[1] * 255.0) as u8;
        let b = (rgb[2] * 255.0) as u8;
        Self::new(r, g, b, 255)
    }
}

impl From<[f64; 4]> for RGBA32 {
    fn from(rgba: [f64; 4]) -> Self {
        let r = (rgba[0] * 255.0) as u8;
        let g = (rgba[1] * 255.0) as u8;
        let b = (rgba[2] * 255.0) as u8;
        let a = (rgba[3] * 255.0) as u8;
        Self::new(r, g, b, a)
    }
}

impl From<u32> for RGBA32 {
    fn from(rgba: u32) -> Self {
        let r = ((rgba >> 24) & 0xff) as u8;
        let g = ((rgba >> 16) & 0xff) as u8;
        let b = ((rgba >> 8) & 0xff) as u8;
        let a = (rgba & 0xff) as u8;
        Self::new(r, g, b, a)
    }
}

impl From<u8> for RGBA32 {
    fn from(rgb: u8) -> Self {
        Self::new(rgb, rgb, rgb, 255)
    }
}

impl From<(u8, u8, u8)> for RGBA32 {
    fn from(rgb: (u8, u8, u8)) -> Self {
        let (r, g, b) = rgb;
        Self::new(r, g, b, 255)
    }
}

impl From<(u8, u8, u8, u8)> for RGBA32 {
    fn from(rgba: (u8, u8, u8, u8)) -> Self {
        let (r, g, b, a) = rgba;
        Self::new(r, g, b, a)
    }
}

impl From<[u8; 3]> for RGBA32 {
    fn from(rgb: [u8; 3]) -> Self {
        Self::new(rgb[0], rgb[1], rgb[2], 255)
    }
}

impl From<[u8; 4]> for RGBA32 {
    fn from(rgba: [u8; 4]) -> Self {
        Self::new(rgba[0], rgba[1], rgba[2], rgba[3])
    }
}

impl From<RGB> for RGBA32 {
    fn from(color: RGB) -> Self {
        color.to_rgba32()
    }
}

impl From<(RGB, u8)> for RGBA32 {
    fn from(color: (RGB, u8)) -> Self {
        color.0.to_rgba32_alpha(color.1)
    }
}

impl From<RGBf> for RGBA32 {
    fn from(color: RGBf) -> Self {
        color.to_rgba32()
    }
}

impl From<(RGBf, u8)> for RGBA32 {
    fn from(color: (RGBf, u8)) -> Self {
        color.0.to_rgba32_alpha(color.1)
    }
}

impl From<RGBA> for RGBA32 {
    fn from(color: RGBA) -> Self {
        color.to_rgba32()
    }
}

impl From<RGBAf> for RGBA32 {
    fn from(color: RGBAf) -> Self {
        color.to_rgba32()
    }
}

impl From<RGB24> for RGBA32 {
    fn from(color: RGB24) -> Self {
        color.to_rgba32()
    }
}

impl From<(RGB24, u8)> for RGBA32 {
    fn from(color: (RGB24, u8)) -> Self {
        color.0.to_rgba32_alpha(color.1)
    }
}

impl RGBA32 {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn black() -> Self {
        Self::new(0, 0, 0, 255)
    }

    pub fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }

    pub fn red() -> Self {
        Self::new(255, 0, 0, 255)
    }

    pub fn green() -> Self {
        Self::new(0, 1, 0, 255)
    }

    pub fn blue() -> Self {
        Self::new(0, 0, 255, 255)
    }

    pub fn black_alpha(alpha: u8) -> Self {
        Self::new(0, 0, 0, alpha)
    }

    pub fn white_alpha(alpha: u8) -> Self {
        Self::new(255, 255, 255, alpha)
    }

    pub fn red_alpha(alpha: u8) -> Self {
        Self::new(255, 0, 0, alpha)
    }

    pub fn green_alpha(alpha: u8) -> Self {
        Self::new(0, 1, 0, alpha)
    }

    pub fn blue_alpha(alpha: u8) -> Self {
        Self::new(0, 0, 255, alpha)
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

    pub fn a(self) -> u8 {
        self.a
    }
}

impl RGBA32 {
    pub fn sum(self) -> i32 {
        self.r as i32 + self.g as i32 + self.b as i32
    }

    pub fn gray(self) -> u8 {
        (self.sum() / 3) as u8
    }

    pub fn min_element(self) -> u8 {
        self.r.min(self.g).min(self.b).min(self.a)
    }

    pub fn max_element(self) -> u8 {
        self.r.max(self.g).max(self.b).max(self.a)
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        ruby_assert!(min.r <= max.r);
        ruby_assert!(min.g <= max.g);
        ruby_assert!(min.b <= max.b);
        ruby_assert!(min.a <= max.a);

        self.min(max).max(min)
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(
            self.r.min(rhs.r),
            self.g.min(rhs.g),
            self.b.min(rhs.b),
            self.a.min(rhs.a),
        )
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(
            self.r.max(rhs.r),
            self.g.max(rhs.g),
            self.b.max(rhs.b),
            self.a.max(rhs.a),
        )
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::black_alpha(0), Self::white_alpha(255))
    }
}

impl RGBA32 {
    pub fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub fn to_tuple(self) -> (u8, u8, u8, u8) {
        (self.r, self.g, self.b, self.a)
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
            self.a as f64 / 255.0,
        )
    }

    pub fn to_rgbaf(self) -> RGBAf {
        RGBAf::new(
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        )
    }

    pub fn to_rgb24(self) -> RGB24 {
        RGB24::new(self.r, self.g, self.b)
    }
}
