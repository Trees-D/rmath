#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{RGBAf, RGBf, RGB, RGB24, RGBA32};

fn convert_f64_to_u8(v: f64) -> u8 {
    let v = (v * 255.0 + 0.5) as i32;
    v.max(0).min(255) as u8
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RGBA {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

impl Display for RGBA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RGBA(r: {}, g: {}, b: {}, a: {})",
            self.r, self.g, self.b, self.a
        )
    }
}

impl Default for RGBA {
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }
}

impl Add<RGBA> for RGBA {
    type Output = RGBA;

    fn add(self, rhs: RGBA) -> Self::Output {
        RGBA::new(
            self.r + rhs.r,
            self.g + rhs.g,
            self.b + rhs.b,
            self.a + rhs.a,
        )
    }
}

impl Add<f64> for RGBA {
    type Output = RGBA;

    fn add(self, rhs: f64) -> Self::Output {
        RGBA::new(self.r + rhs, self.g + rhs, self.b + rhs, self.a + rhs)
    }
}

impl Add<RGBA> for f64 {
    type Output = RGBA;

    fn add(self, rhs: RGBA) -> Self::Output {
        RGBA::new(self + rhs.r, self + rhs.g, self + rhs.b, self + rhs.a)
    }
}

impl AddAssign<RGBA> for RGBA {
    fn add_assign(&mut self, rhs: RGBA) {
        *self = *self + rhs;
    }
}

impl AddAssign<f64> for RGBA {
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}

impl Sub<RGBA> for RGBA {
    type Output = RGBA;

    fn sub(self, rhs: RGBA) -> Self::Output {
        RGBA::new(
            self.r - rhs.r,
            self.g - rhs.g,
            self.b - rhs.b,
            self.a - rhs.a,
        )
    }
}

impl Sub<f64> for RGBA {
    type Output = RGBA;

    fn sub(self, rhs: f64) -> Self::Output {
        RGBA::new(self.r - rhs, self.g - rhs, self.b - rhs, self.a - rhs)
    }
}

impl Sub<RGBA> for f64 {
    type Output = RGBA;

    fn sub(self, rhs: RGBA) -> Self::Output {
        RGBA::new(self - rhs.r, self - rhs.g, self - rhs.b, self - rhs.a)
    }
}

impl SubAssign<RGBA> for RGBA {
    fn sub_assign(&mut self, rhs: RGBA) {
        *self = *self - rhs;
    }
}

impl SubAssign<f64> for RGBA {
    fn sub_assign(&mut self, rhs: f64) {
        *self = *self - rhs;
    }
}

impl Mul<RGBA> for RGBA {
    type Output = RGBA;

    fn mul(self, rhs: RGBA) -> Self::Output {
        RGBA::new(
            self.r * rhs.r,
            self.g * rhs.g,
            self.b * rhs.b,
            self.a * rhs.a,
        )
    }
}

impl Mul<f64> for RGBA {
    type Output = RGBA;

    fn mul(self, rhs: f64) -> Self::Output {
        RGBA::new(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
    }
}

impl Mul<RGBA> for f64 {
    type Output = RGBA;

    fn mul(self, rhs: RGBA) -> Self::Output {
        RGBA::new(self * rhs.r, self * rhs.g, self * rhs.b, self * rhs.a)
    }
}

impl MulAssign<RGBA> for RGBA {
    fn mul_assign(&mut self, rhs: RGBA) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for RGBA {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl Div<RGBA> for RGBA {
    type Output = RGBA;

    fn div(self, rhs: RGBA) -> Self::Output {
        RGBA::new(
            self.r / rhs.r,
            self.g / rhs.g,
            self.b / rhs.b,
            self.a / rhs.a,
        )
    }
}

impl Div<f64> for RGBA {
    type Output = RGBA;

    fn div(self, rhs: f64) -> Self::Output {
        RGBA::new(self.r / rhs, self.g / rhs, self.b / rhs, self.a / rhs)
    }
}

impl Div<RGBA> for f64 {
    type Output = RGBA;

    fn div(self, rhs: RGBA) -> Self::Output {
        RGBA::new(self / rhs.r, self / rhs.g, self / rhs.b, self / rhs.a)
    }
}

impl DivAssign<RGBA> for RGBA {
    fn div_assign(&mut self, rhs: RGBA) {
        *self = *self / rhs;
    }
}

impl DivAssign<f64> for RGBA {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl Neg for RGBA {
    type Output = RGBA;

    fn neg(self) -> Self::Output {
        RGBA::new(-self.r, -self.g, -self.b, -self.a)
    }
}

impl Index<usize> for RGBA {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            3 => &self.a,
            _ => panic!("`rmath::color::RGBA::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for RGBA {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            3 => &mut self.a,
            _ => panic!("`rmath::color::RGBA::index_mut`: index out of bounds."),
        }
    }
}

impl From<f64> for RGBA {
    fn from(rgb: f64) -> Self {
        Self::new(rgb, rgb, rgb, 1.0)
    }
}

impl From<(f64, f64, f64)> for RGBA {
    fn from(rgb: (f64, f64, f64)) -> Self {
        Self::new(rgb.0, rgb.1, rgb.2, 1.0)
    }
}

impl From<(f64, f64, f64, f64)> for RGBA {
    fn from(rgba: (f64, f64, f64, f64)) -> Self {
        Self::new(rgba.0, rgba.1, rgba.2, rgba.3)
    }
}

impl From<[f64; 3]> for RGBA {
    fn from(rgb: [f64; 3]) -> Self {
        Self::new(rgb[0], rgb[1], rgb[2], 1.0)
    }
}

impl From<[f64; 4]> for RGBA {
    fn from(rgba: [f64; 4]) -> Self {
        Self::new(rgba[0], rgba[1], rgba[2], rgba[3])
    }
}

impl From<u32> for RGBA {
    fn from(rgba: u32) -> Self {
        let r = ((rgba >> 24) & 0xff) as f64 / 255.0;
        let g = ((rgba >> 16) & 0xff) as f64 / 255.0;
        let b = ((rgba >> 8) & 0xff) as f64 / 255.0;
        let a = (rgba & 0xff) as f64 / 255.0;
        Self::new(r, g, b, a)
    }
}

impl From<u8> for RGBA {
    fn from(rgba: u8) -> Self {
        let rgba = rgba as f64 / 255.0;
        Self::new(rgba, rgba, rgba, rgba)
    }
}

impl From<(u8, u8, u8)> for RGBA {
    fn from(rgb: (u8, u8, u8)) -> Self {
        let (r, g, b) = rgb;
        let r = r as f64 / 255.0;
        let g = g as f64 / 255.0;
        let b = b as f64 / 255.0;
        Self::new(r, g, b, 1.0)
    }
}

impl From<(u8, u8, u8, u8)> for RGBA {
    fn from(rgba: (u8, u8, u8, u8)) -> Self {
        let (r, g, b, a) = rgba;
        let r = r as f64 / 255.0;
        let g = g as f64 / 255.0;
        let b = b as f64 / 255.0;
        let a = a as f64 / 255.0;
        Self::new(r, g, b, a)
    }
}

impl From<[u8; 3]> for RGBA {
    fn from(rgb: [u8; 3]) -> Self {
        let r = rgb[0] as f64 / 255.0;
        let g = rgb[1] as f64 / 255.0;
        let b = rgb[2] as f64 / 255.0;
        Self::new(r, g, b, 1.0)
    }
}

impl From<[u8; 4]> for RGBA {
    fn from(rgba: [u8; 4]) -> Self {
        let r = rgba[0] as f64 / 255.0;
        let g = rgba[1] as f64 / 255.0;
        let b = rgba[2] as f64 / 255.0;
        let a = rgba[3] as f64 / 255.0;
        Self::new(r, g, b, a)
    }
}

impl From<RGB> for RGBA {
    fn from(color: RGB) -> Self {
        color.to_rgba()
    }
}

impl From<(RGB, f64)> for RGBA {
    fn from(color: (RGB, f64)) -> Self {
        color.0.to_rgba_alpha(color.1)
    }
}

impl From<RGBf> for RGBA {
    fn from(color: RGBf) -> Self {
        color.to_rgba()
    }
}

impl From<(RGBf, f64)> for RGBA {
    fn from(color: (RGBf, f64)) -> Self {
        color.0.to_rgba_alpha(color.1)
    }
}

impl From<RGBAf> for RGBA {
    fn from(color: RGBAf) -> Self {
        color.to_rgba()
    }
}

impl From<RGB24> for RGBA {
    fn from(color: RGB24) -> Self {
        color.to_rgba()
    }
}

impl From<(RGB24, f64)> for RGBA {
    fn from(color: (RGB24, f64)) -> Self {
        color.0.to_rgba_alpha(color.1)
    }
}

impl From<RGBA32> for RGBA {
    fn from(color: RGBA32) -> Self {
        color.to_rgba()
    }
}

impl RGBA {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    pub fn red() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    pub fn green() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0)
    }

    pub fn blue() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }

    pub fn black_alpha(alpha: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, alpha)
    }

    pub fn white_alpha(alpha: f64) -> Self {
        Self::new(1.0, 1.0, 1.0, alpha)
    }

    pub fn red_alpha(alpha: f64) -> Self {
        Self::new(1.0, 0.0, 0.0, alpha)
    }

    pub fn green_alpha(alpha: f64) -> Self {
        Self::new(0.0, 1.0, 0.0, alpha)
    }

    pub fn blue_alpha(alpha: f64) -> Self {
        Self::new(0.0, 0.0, 1.0, alpha)
    }

    pub fn r(self) -> f64 {
        self.r
    }

    pub fn g(self) -> f64 {
        self.g
    }

    pub fn b(self) -> f64 {
        self.b
    }

    pub fn a(self) -> f64 {
        self.a
    }
}

impl RGBA {
    pub fn sum(self) -> f64 {
        self.r + self.g + self.b
    }

    pub fn gray(self) -> f64 {
        self.sum() / 3.0
    }

    pub fn luma1(self) -> f64 {
        0.299 * self.r + 0.587 * self.g + 0.144 * self.b
    }

    pub fn luma2(self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn min_element(self) -> f64 {
        self.r.min(self.g).min(self.b).min(self.a)
    }

    pub fn max_element(self) -> f64 {
        self.r.max(self.g).max(self.b).max(self.a)
    }

    pub fn is_finite(self) -> bool {
        self.r.is_finite() && self.g.is_finite() && self.b.is_finite() && self.a.is_finite()
    }

    pub fn is_nan(self) -> bool {
        self.r.is_nan() || self.g.is_nan() || self.b.is_nan() || self.a.is_nan()
    }

    pub fn is_infinite(self) -> bool {
        self.r.is_infinite() || self.g.is_infinite() || self.b.is_infinite() || self.a.is_infinite()
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

    pub fn abs(self) -> Self {
        Self::new(self.r.abs(), self.g.abs(), self.b.abs(), self.a.abs())
    }

    pub fn round(self) -> Self {
        Self::new(
            self.r.round(),
            self.g.round(),
            self.b.round(),
            self.a.round(),
        )
    }

    pub fn floor(self) -> Self {
        Self::new(
            self.r.floor(),
            self.g.floor(),
            self.b.floor(),
            self.a.floor(),
        )
    }

    pub fn ceil(self) -> Self {
        Self::new(self.r.ceil(), self.g.ceil(), self.b.ceil(), self.a.ceil())
    }

    pub fn trunc(self) -> Self {
        Self::new(
            self.r.trunc(),
            self.g.trunc(),
            self.b.trunc(),
            self.a.trunc(),
        )
    }

    pub fn fract(self) -> Self {
        Self::new(
            self.r.fract(),
            self.g.fract(),
            self.b.fract(),
            self.a.fract(),
        )
    }

    pub fn sqrt(self) -> Self {
        Self::new(self.r.sqrt(), self.g.sqrt(), self.b.sqrt(), self.a.sqrt())
    }

    pub fn exp(self) -> Self {
        Self::new(self.r.exp(), self.g.exp(), self.b.exp(), self.a.exp())
    }

    pub fn exp2(self) -> Self {
        Self::new(self.r.exp2(), self.g.exp2(), self.b.exp2(), self.a.exp2())
    }

    pub fn ln(self) -> Self {
        Self::new(self.r.ln(), self.g.ln(), self.b.ln(), self.a.ln())
    }

    pub fn log(self, base: f64) -> Self {
        Self::new(
            self.r.log(base),
            self.g.log(base),
            self.b.log(base),
            self.a.log(base),
        )
    }

    pub fn log2(self) -> Self {
        Self::new(self.r.log2(), self.g.log2(), self.b.log2(), self.a.log2())
    }

    pub fn log10(self) -> Self {
        Self::new(
            self.r.log10(),
            self.g.log10(),
            self.b.log10(),
            self.a.log10(),
        )
    }

    pub fn cbrt(self) -> Self {
        Self::new(self.r.cbrt(), self.g.cbrt(), self.b.cbrt(), self.a.cbrt())
    }

    pub fn powf(self, n: f64) -> Self {
        Self::new(
            self.r.powf(n),
            self.g.powf(n),
            self.b.powf(n),
            self.a.powf(n),
        )
    }

    pub fn sin(self) -> Self {
        Self::new(self.r.sin(), self.g.sin(), self.b.sin(), self.a.sin())
    }

    pub fn cos(self) -> Self {
        Self::new(self.r.cos(), self.g.cos(), self.b.cos(), self.a.cos())
    }

    pub fn tan(self) -> Self {
        Self::new(self.r.tan(), self.g.tan(), self.b.tan(), self.a.tan())
    }

    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    pub fn recip(self) -> Self {
        Self::new(
            self.r.recip(),
            self.g.recip(),
            self.b.recip(),
            self.a.recip(),
        )
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::black_alpha(0.0), Self::white_alpha(1.0))
    }

    pub fn lerp(self, rhs: Self, s: f64) -> Self {
        (rhs - self) * s + self
    }

    pub fn gamma_correct(self) -> Self {
        let inv = 1.0 / 2.2;
        Self::new(self.r.powf(inv), self.g.powf(inv), self.b.powf(inv), self.a)
    }
}

impl RGBA {
    pub fn to_array(self) -> [f64; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub fn to_tuple(self) -> (f64, f64, f64, f64) {
        (self.r, self.g, self.b, self.a)
    }

    pub fn to_rgbf(self) -> RGBf {
        RGBf::new(self.r as f32, self.g as f32, self.b as f32)
    }

    pub fn to_rgb(self) -> RGB {
        RGB::new(self.r, self.g, self.b)
    }

    pub fn to_rgbaf(self) -> RGBAf {
        RGBAf::new(self.r as f32, self.g as f32, self.b as f32, self.a as f32)
    }

    pub fn to_rgb24(self) -> RGB24 {
        RGB24::new(
            convert_f64_to_u8(self.r),
            convert_f64_to_u8(self.g),
            convert_f64_to_u8(self.b),
        )
    }

    pub fn to_rgba32(self) -> RGBA32 {
        RGBA32::new(
            convert_f64_to_u8(self.r),
            convert_f64_to_u8(self.g),
            convert_f64_to_u8(self.b),
            convert_f64_to_u8(self.a),
        )
    }
}
