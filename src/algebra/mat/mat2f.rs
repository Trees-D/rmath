#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::algebra::{vec2f, Mat2d, Mat3f, Mat4f, Vec2f};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mat2f {
    x_axis: Vec2f,
    y_axis: Vec2f,
}

pub fn mat2f(m00: f32, m01: f32, m10: f32, m11: f32) -> Mat2f {
    Mat2f::from_array([m00, m01, m10, m11])
}

impl Display for Mat2f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mat2f({}, {} | {}, {})",
            self.x_axis.x(),
            self.x_axis.y(),
            self.y_axis.x(),
            self.y_axis.y()
        )
    }
}

impl Default for Mat2f {
    fn default() -> Self {
        Self {
            x_axis: Vec2f::default(),
            y_axis: Vec2f::default(),
        }
    }
}

impl Add<Mat2f> for Mat2f {
    type Output = Mat2f;

    fn add(self, rhs: Mat2f) -> Self::Output {
        Mat2f::new(self.x_axis + rhs.x_axis, self.y_axis + rhs.y_axis)
    }
}

impl Add<f32> for Mat2f {
    type Output = Mat2f;

    fn add(self, rhs: f32) -> Self::Output {
        Mat2f::new(self.x_axis + rhs, self.y_axis + rhs)
    }
}

impl Add<Mat2f> for f32 {
    type Output = Mat2f;

    fn add(self, rhs: Mat2f) -> Self::Output {
        Mat2f::new(self + rhs.x_axis, self + rhs.y_axis)
    }
}

impl AddAssign<Mat2f> for Mat2f {
    fn add_assign(&mut self, rhs: Mat2f) {
        *self = *self + rhs;
    }
}

impl AddAssign<f32> for Mat2f {
    fn add_assign(&mut self, rhs: f32) {
        *self = *self + rhs;
    }
}

impl Sub<Mat2f> for Mat2f {
    type Output = Mat2f;

    fn sub(self, rhs: Mat2f) -> Self::Output {
        Mat2f::new(self.x_axis - rhs.x_axis, self.y_axis - rhs.y_axis)
    }
}

impl Sub<f32> for Mat2f {
    type Output = Mat2f;

    fn sub(self, rhs: f32) -> Self::Output {
        Mat2f::new(self.x_axis - rhs, self.y_axis - rhs)
    }
}

impl Sub<Mat2f> for f32 {
    type Output = Mat2f;

    fn sub(self, rhs: Mat2f) -> Self::Output {
        Mat2f::new(self - rhs.x_axis, self - rhs.y_axis)
    }
}

impl SubAssign<Mat2f> for Mat2f {
    fn sub_assign(&mut self, rhs: Mat2f) {
        *self = *self - rhs;
    }
}

impl SubAssign<f32> for Mat2f {
    fn sub_assign(&mut self, rhs: f32) {
        *self = *self - rhs;
    }
}

impl Mul<Mat2f> for Mat2f {
    type Output = Mat2f;

    fn mul(self, rhs: Mat2f) -> Self::Output {
        let m00 = self.x_axis.dot(vec2f(rhs[0][0], rhs[1][0]));
        let m01 = self.x_axis.dot(vec2f(rhs[0][1], rhs[1][1]));
        let m10 = self.y_axis.dot(vec2f(rhs[0][0], rhs[1][0]));
        let m11 = self.y_axis.dot(vec2f(rhs[0][1], rhs[1][1]));
        Mat2f::new(vec2f(m00, m01), vec2f(m10, m11))
    }
}

impl Mul<Vec2f> for Mat2f {
    type Output = Vec2f;

    fn mul(self, rhs: Vec2f) -> Self::Output {
        let v0 = self.x_axis.dot(rhs);
        let v1 = self.y_axis.dot(rhs);
        vec2f(v0, v1)
    }
}

impl Mul<f32> for Mat2f {
    type Output = Mat2f;

    fn mul(self, rhs: f32) -> Self::Output {
        Mat2f::new(self.x_axis * rhs, self.y_axis * rhs)
    }
}

impl Mul<Mat2f> for f32 {
    type Output = Mat2f;

    fn mul(self, rhs: Mat2f) -> Self::Output {
        Mat2f::new(self * rhs.x_axis, self * rhs.y_axis)
    }
}

impl MulAssign<Mat2f> for Mat2f {
    fn mul_assign(&mut self, rhs: Mat2f) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Mat2f {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<f32> for Mat2f {
    type Output = Mat2f;

    fn div(self, rhs: f32) -> Self::Output {
        Mat2f::new(self.x_axis / rhs, self.y_axis / rhs)
    }
}

impl DivAssign<f32> for Mat2f {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Index<usize> for Mat2f {
    type Output = Vec2f;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x_axis,
            1 => &self.y_axis,
            _ => panic!("`rmath::algebra::Mat2f::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Mat2f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            _ => panic!("`rmath::algebra::Mat2f::index_mut`: index out of bounds."),
        }
    }
}

impl Mat2f {
    pub fn new(x_axis: Vec2f, y_axis: Vec2f) -> Self {
        Self { x_axis, y_axis }
    }

    pub fn zero() -> Self {
        Self::new(vec2f(0.0, 0.0), vec2f(0.0, 0.0))
    }

    pub fn identity() -> Self {
        Self::new(vec2f(1.0, 0.0), vec2f(0.0, 1.0))
    }
}

impl Mat2f {
    pub fn col(&self, index: usize) -> Vec2f {
        match index {
            0 => vec2f(self[0][0], self[1][0]),
            1 => vec2f(self[0][1], self[1][1]),
            _ => panic!("`rmath::algebra::Mat2f::col`: index out of bounds."),
        }
    }

    pub fn row(&self, index: usize) -> Vec2f {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => panic!("`rmath::algebra::Mat2f::row`: index out of bounds."),
        }
    }

    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    pub fn is_infinite(&self) -> bool {
        self.x_axis.is_infinite() || self.y_axis.is_infinite()
    }

    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    pub fn transpose(&self) -> Self {
        let (m00, m01) = self.x_axis.to_tuple();
        let (m10, m11) = self.y_axis.to_tuple();
        Self::new(vec2f(m00, m10), vec2f(m01, m11))
    }

    pub fn determinant(&self) -> f32 {
        let (m00, m01) = self.x_axis.to_tuple();
        let (m10, m11) = self.y_axis.to_tuple();
        m00 * m11 - m01 * m10
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();
        ruby_assert!(!(det.abs() < f32::EPSILON));

        let inv_det = det.recip();
        let a00 = self[1][1];
        let a01 = -self[1][0];
        let a10 = -self[0][1];
        let a11 = self[0][0];

        Self::new(vec2f(a00, a10), vec2f(a01, a11)).mul(inv_det)
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }

        let inv_det = det.recip();
        let a00 = self[1][1];
        let a01 = -self[1][0];
        let a10 = -self[0][1];
        let a11 = self[0][0];

        Some(Self::new(vec2f(a00, a10), vec2f(a01, a11)).mul(inv_det))
    }
}

impl From<Mat3f> for Mat2f {
    fn from(m: Mat3f) -> Self {
        mat2f(m[0][0], m[0][1], m[1][0], m[1][1])
    }
}

impl From<Mat4f> for Mat2f {
    fn from(m: Mat4f) -> Self {
        mat2f(m[0][0], m[0][1], m[1][0], m[1][1])
    }
}

impl Mat2f {
    pub fn from_array(m: [f32; 4]) -> Self {
        Self::new(vec2f(m[0], m[1]), vec2f(m[2], m[3]))
    }

    pub fn from_array_2d(m: [[f32; 2]; 2]) -> Self {
        Self::new(vec2f(m[0][0], m[0][1]), vec2f(m[1][0], m[1][1]))
    }

    pub fn from_diagonal(diagonal: Vec2f) -> Self {
        Self::new(vec2f(diagonal.x(), 0.0), vec2f(0.0, diagonal.y()))
    }
}

impl Mat2f {
    pub fn to_array(self) -> [f32; 4] {
        [self[0][0], self[0][1], self[1][0], self[1][1]]
    }

    pub fn to_array_2d(self) -> [[f32; 2]; 2] {
        [[self[0][0], self[0][1]], [self[1][0], self[1][1]]]
    }

    pub fn to_mat3f(self) -> Mat3f {
        Mat3f::from(self)
    }

    pub fn to_mat4f(self) -> Mat4f {
        Mat4f::from(self)
    }

    pub fn to_mat2d(self) -> Mat2d {
        Mat2d::new(self.x_axis.to_vec2d(), self.y_axis.to_vec2d())
    }
}
