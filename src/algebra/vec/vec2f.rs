#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{Vec3f, Vec4f};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vec2f {
    x: f32,
    y: f32,
}

pub fn vec2f(x: f32, y: f32) -> Vec2f {
    Vec2f::new(x, y)
}

impl Display for Vec2f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec2f(x: {}, y: {})", self.x, self.y)
    }
}

impl Default for Vec2f {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Add<Vec2f> for Vec2f {
    type Output = Vec2f;

    fn add(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Add<f32> for Vec2f {
    type Output = Vec2f;

    fn add(self, rhs: f32) -> Self::Output {
        Vec2f::new(self.x + rhs, self.y + rhs)
    }
}

impl Add<Vec2f> for f32 {
    type Output = Vec2f;

    fn add(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self + rhs.x, self + rhs.y)
    }
}

impl AddAssign<Vec2f> for Vec2f {
    fn add_assign(&mut self, rhs: Vec2f) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl AddAssign<f32> for Vec2f {
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
    }
}

impl Sub<Vec2f> for Vec2f {
    type Output = Vec2f;

    fn sub(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Sub<f32> for Vec2f {
    type Output = Vec2f;

    fn sub(self, rhs: f32) -> Self::Output {
        Vec2f::new(self.x - rhs, self.y - rhs)
    }
}

impl Sub<Vec2f> for f32 {
    type Output = Vec2f;

    fn sub(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self - rhs.x, self - rhs.y)
    }
}

impl SubAssign<Vec2f> for Vec2f {
    fn sub_assign(&mut self, rhs: Vec2f) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl SubAssign<f32> for Vec2f {
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
    }
}

impl Mul<Vec2f> for Vec2f {
    type Output = Vec2f;

    fn mul(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Mul<f32> for Vec2f {
    type Output = Vec2f;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec2f::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2f> for f32 {
    type Output = Vec2f;

    fn mul(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self * rhs.x, self * rhs.y)
    }
}

impl MulAssign<Vec2f> for Vec2f {
    fn mul_assign(&mut self, rhs: Vec2f) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl MulAssign<f32> for Vec2f {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<Vec2f> for Vec2f {
    type Output = Vec2f;

    fn div(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Div<f32> for Vec2f {
    type Output = Vec2f;

    fn div(self, rhs: f32) -> Self::Output {
        Vec2f::new(self.x / rhs, self.y / rhs)
    }
}

impl Div<Vec2f> for f32 {
    type Output = Vec2f;

    fn div(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self / rhs.x, self / rhs.y)
    }
}

impl DivAssign<Vec2f> for Vec2f {
    fn div_assign(&mut self, rhs: Vec2f) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl DivAssign<f32> for Vec2f {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl Neg for Vec2f {
    type Output = Vec2f;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl Index<usize> for Vec2f {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("`rmath::algebra::Vec2f::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Vec2f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("`rmath::algebra::Vec2f::index_mut`: index out of bounds."),
        }
    }
}

impl From<f32> for Vec2f {
    fn from(v: f32) -> Self {
        Self::new(v, v)
    }
}

impl From<(f32, f32)> for Vec2f {
    fn from(v: (f32, f32)) -> Self {
        let (x, y) = v;
        Self::new(x, y)
    }
}

impl From<Vec3f> for Vec2f {
    fn from(v: Vec3f) -> Self {
        v.xy()
    }
}

impl From<Vec4f> for Vec2f {
    fn from(v: Vec4f) -> Self {
        v.xy()
    }
}

impl Vec2f {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn one() -> Self {
        Self::new(1.0, 1.0)
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Vec2f {
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor())
    }

    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil())
    }

    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round())
    }

    pub fn trunc(self) -> Self {
        Self::new(self.x.trunc(), self.y.trunc())
    }

    pub fn fract(self) -> Self {
        Self::new(self.x.fract(), self.y.fract())
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum())
    }

    pub fn powf(self, n: f32) -> Self {
        Self::new(self.x.powf(n), self.y.powf(n))
    }

    pub fn sqrt(self) -> Self {
        Self::new(self.x.sqrt(), self.y.sqrt())
    }

    pub fn exp(self) -> Self {
        Self::new(self.x.exp(), self.y.exp())
    }

    pub fn exp2(self) -> Self {
        Self::new(self.x.exp2(), self.y.exp2())
    }

    pub fn ln(self) -> Self {
        Self::new(self.x.ln(), self.y.ln())
    }

    pub fn log(self, base: f32) -> Self {
        Self::new(self.x.log(base), self.y.log(base))
    }

    pub fn log2(self) -> Self {
        Self::new(self.x.log2(), self.y.log2())
    }

    pub fn log10(self) -> Self {
        Self::new(self.x.log10(), self.y.log10())
    }

    pub fn cbrt(self) -> Self {
        Self::new(self.x.cbrt(), self.y.cbrt())
    }

    pub fn sin(self) -> Self {
        Self::new(self.x.sin(), self.y.sin())
    }

    pub fn cos(self) -> Self {
        Self::new(self.x.cos(), self.y.cos())
    }

    pub fn tan(self) -> Self {
        Self::new(self.x.tan(), self.y.tan())
    }

    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self + (rhs - self) * s
    }

    pub fn lerp_vec(self, rhs: Self, s: Self) -> Self {
        self + (rhs - self) * s
    }

    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    pub fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.y.is_infinite()
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip())
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }

    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        ruby_assert!(min.x <= max.x);
        ruby_assert!(min.y <= max.y);

        self.min(max).max(min)
    }

    pub fn saturate(self) -> Self {
        self.clamp(Self::zero(), Self::one())
    }

    pub fn min_element(self) -> f32 {
        self.x.min(self.y)
    }

    pub fn max_element(self) -> f32 {
        self.x.max(self.y)
    }
}

impl Vec2f {
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    pub fn cross(self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length_recip(self) -> f32 {
        self.length().recip()
    }

    pub fn distance(self, rhs: Self) -> f32 {
        (rhs - self).length()
    }

    pub fn distance_squared(self, rhs: Self) -> f32 {
        (rhs - self).length_squared()
    }

    pub fn normalize(self) -> Self {
        let normalized = self * self.length_recip();
        ruby_assert!(normalized.is_finite());
        normalized
    }

    pub fn try_normalize(self) -> Option<Self> {
        let recip = self.length_recip();
        if recip.is_finite() && recip > 0.0 {
            Some(self * recip)
        } else {
            None
        }
    }

    pub fn normalize_or_zero(self) -> Self {
        let recip = self.length_recip();
        if recip.is_finite() && recip > 0.0 {
            self * recip
        } else {
            Self::zero()
        }
    }

    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0f32).abs() <= f32::EPSILON
    }

    pub fn angle_between(self, rhs: Self) -> f32 {
        let angle = self
            .dot(rhs)
            .div(self.length_squared().mul(rhs.length_squared()).sqrt())
            .acos();
        if self.cross(rhs) < 0.0 {
            -angle
        } else {
            angle
        }
    }
}

impl Vec2f {
    pub fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn to_tuple(self) -> (f32, f32) {
        (self.x, self.y)
    }
}

impl Vec2f {
    pub fn x(self) -> f32 {
        self.x
    }

    pub fn y(self) -> f32 {
        self.y
    }

    pub fn xx(self) -> Self {
        Self::new(self.x, self.x)
    }

    pub fn xy(self) -> Self {
        Self::new(self.x, self.y)
    }

    pub fn yx(self) -> Self {
        Self::new(self.y, self.x)
    }

    pub fn yy(self) -> Self {
        Self::new(self.y, self.y)
    }

    pub fn xxx(self) -> Vec3f {
        Vec3f::new(self.x, self.x, self.x)
    }

    pub fn xxy(self) -> Vec3f {
        Vec3f::new(self.x, self.x, self.y)
    }

    pub fn xyx(self) -> Vec3f {
        Vec3f::new(self.x, self.y, self.x)
    }

    pub fn xyy(self) -> Vec3f {
        Vec3f::new(self.x, self.y, self.y)
    }

    pub fn yxx(self) -> Vec3f {
        Vec3f::new(self.y, self.x, self.x)
    }

    pub fn yxy(self) -> Vec3f {
        Vec3f::new(self.y, self.x, self.y)
    }

    pub fn yyx(self) -> Vec3f {
        Vec3f::new(self.y, self.y, self.x)
    }

    pub fn yyy(self) -> Vec3f {
        Vec3f::new(self.y, self.y, self.y)
    }

    pub fn xxxx(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.x, self.x)
    }

    pub fn xxxy(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.x, self.y)
    }

    pub fn xxyx(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.y, self.x)
    }

    pub fn xxyy(self) -> Vec4f {
        Vec4f::new(self.x, self.x, self.y, self.y)
    }

    pub fn yxxx(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.x, self.x)
    }

    pub fn yxxy(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.x, self.y)
    }

    pub fn yxyx(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.y, self.x)
    }

    pub fn yxyy(self) -> Vec4f {
        Vec4f::new(self.y, self.x, self.y, self.y)
    }
}
