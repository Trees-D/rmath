#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::algebra::{vec2d, Mat2f, Mat3d, Mat4d, Vec2d};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mat2d {
    x_axis: Vec2d,
    y_axis: Vec2d,
}

pub fn mat2d(m00: f64, m01: f64, m10: f64, m11: f64) -> Mat2d {
    Mat2d::from_array([m00, m01, m10, m11])
}

impl Display for Mat2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mat2d({}, {} | {}, {})",
            self.x_axis.x(),
            self.x_axis.y(),
            self.y_axis.x(),
            self.y_axis.y()
        )
    }
}

impl Default for Mat2d {
    fn default() -> Self {
        Self {
            x_axis: Vec2d::default(),
            y_axis: Vec2d::default(),
        }
    }
}

impl Add<Mat2d> for Mat2d {
    type Output = Mat2d;

    fn add(self, rhs: Mat2d) -> Self::Output {
        Mat2d::new(self.x_axis + rhs.x_axis, self.y_axis + rhs.y_axis)
    }
}

impl Add<f64> for Mat2d {
    type Output = Mat2d;

    fn add(self, rhs: f64) -> Self::Output {
        Mat2d::new(self.x_axis + rhs, self.y_axis + rhs)
    }
}

impl Add<Mat2d> for f64 {
    type Output = Mat2d;

    fn add(self, rhs: Mat2d) -> Self::Output {
        Mat2d::new(self + rhs.x_axis, self + rhs.y_axis)
    }
}

impl AddAssign<Mat2d> for Mat2d {
    fn add_assign(&mut self, rhs: Mat2d) {
        *self = *self + rhs;
    }
}

impl AddAssign<f64> for Mat2d {
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}

impl Sub<Mat2d> for Mat2d {
    type Output = Mat2d;

    fn sub(self, rhs: Mat2d) -> Self::Output {
        Mat2d::new(self.x_axis - rhs.x_axis, self.y_axis - rhs.y_axis)
    }
}

impl Sub<f64> for Mat2d {
    type Output = Mat2d;

    fn sub(self, rhs: f64) -> Self::Output {
        Mat2d::new(self.x_axis - rhs, self.y_axis - rhs)
    }
}

impl Sub<Mat2d> for f64 {
    type Output = Mat2d;

    fn sub(self, rhs: Mat2d) -> Self::Output {
        Mat2d::new(self - rhs.x_axis, self - rhs.y_axis)
    }
}

impl SubAssign<Mat2d> for Mat2d {
    fn sub_assign(&mut self, rhs: Mat2d) {
        *self = *self - rhs;
    }
}

impl SubAssign<f64> for Mat2d {
    fn sub_assign(&mut self, rhs: f64) {
        *self = *self - rhs;
    }
}

impl Mul<Mat2d> for Mat2d {
    type Output = Mat2d;

    fn mul(self, rhs: Mat2d) -> Self::Output {
        let m00 = self.x_axis.dot(vec2d(rhs[0][0], rhs[1][0]));
        let m01 = self.x_axis.dot(vec2d(rhs[0][1], rhs[1][1]));
        let m10 = self.y_axis.dot(vec2d(rhs[0][0], rhs[1][0]));
        let m11 = self.y_axis.dot(vec2d(rhs[0][1], rhs[1][1]));
        Mat2d::new(vec2d(m00, m01), vec2d(m10, m11))
    }
}

impl Mul<Vec2d> for Mat2d {
    type Output = Vec2d;

    fn mul(self, rhs: Vec2d) -> Self::Output {
        let v0 = self.x_axis.dot(rhs);
        let v1 = self.y_axis.dot(rhs);
        vec2d(v0, v1)
    }
}

impl Mul<f64> for Mat2d {
    type Output = Mat2d;

    fn mul(self, rhs: f64) -> Self::Output {
        Mat2d::new(self.x_axis * rhs, self.y_axis * rhs)
    }
}

impl Mul<Mat2d> for f64 {
    type Output = Mat2d;

    fn mul(self, rhs: Mat2d) -> Self::Output {
        Mat2d::new(self * rhs.x_axis, self * rhs.y_axis)
    }
}

impl MulAssign<Mat2d> for Mat2d {
    fn mul_assign(&mut self, rhs: Mat2d) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for Mat2d {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl Div<f64> for Mat2d {
    type Output = Mat2d;

    fn div(self, rhs: f64) -> Self::Output {
        Mat2d::new(self.x_axis / rhs, self.y_axis / rhs)
    }
}

impl DivAssign<f64> for Mat2d {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl Index<usize> for Mat2d {
    type Output = Vec2d;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x_axis,
            1 => &self.y_axis,
            _ => panic!("`rmath::algebra::Mat2d::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Mat2d {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            _ => panic!("`rmath::algebra::Mat2d::index_mut`: index out of bounds."),
        }
    }
}

impl Mat2d {
    pub fn new(x_axis: Vec2d, y_axis: Vec2d) -> Self {
        Self { x_axis, y_axis }
    }

    pub fn zero() -> Self {
        Self::new(vec2d(0.0, 0.0), vec2d(0.0, 0.0))
    }

    pub fn identity() -> Self {
        Self::new(vec2d(1.0, 0.0), vec2d(0.0, 1.0))
    }
}

impl Mat2d {
    pub fn col(&self, index: usize) -> Vec2d {
        match index {
            0 => vec2d(self[0][0], self[1][0]),
            1 => vec2d(self[0][1], self[1][1]),
            _ => panic!("`rmath::algebra::Mat2d::col`: index out of bounds."),
        }
    }

    pub fn row(&self, index: usize) -> Vec2d {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => panic!("`rmath::algebra::Mat2d::row`: index out of bounds."),
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
        Self::new(vec2d(m00, m10), vec2d(m01, m11))
    }

    pub fn determinant(&self) -> f64 {
        let (m00, m01) = self.x_axis.to_tuple();
        let (m10, m11) = self.y_axis.to_tuple();
        m00 * m11 - m01 * m10
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();
        ruby_assert!(!(det.abs() < f64::EPSILON));

        let inv_det = det.recip();
        let a00 = self[1][1];
        let a01 = -self[1][0];
        let a10 = -self[0][1];
        let a11 = self[0][0];

        Self::new(vec2d(a00, a10), vec2d(a01, a11)).mul(inv_det)
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f64::EPSILON {
            return None;
        }

        let inv_det = det.recip();
        let a00 = self[1][1];
        let a01 = -self[1][0];
        let a10 = -self[0][1];
        let a11 = self[0][0];

        Some(Self::new(vec2d(a00, a10), vec2d(a01, a11)).mul(inv_det))
    }
}

impl From<Mat3d> for Mat2d {
    fn from(m: Mat3d) -> Self {
        mat2d(m[0][0], m[0][1], m[1][0], m[1][1])
    }
}

impl From<Mat4d> for Mat2d {
    fn from(m: Mat4d) -> Self {
        mat2d(m[0][0], m[0][1], m[1][0], m[1][1])
    }
}

impl Mat2d {
    pub fn from_array(m: [f64; 4]) -> Self {
        Self::new(vec2d(m[0], m[1]), vec2d(m[2], m[3]))
    }

    pub fn from_array_2d(m: [[f64; 2]; 2]) -> Self {
        Self::new(vec2d(m[0][0], m[0][1]), vec2d(m[1][0], m[1][1]))
    }

    pub fn from_diagonal(diagonal: Vec2d) -> Self {
        Self::new(vec2d(diagonal.x(), 0.0), vec2d(0.0, diagonal.y()))
    }
}

impl Mat2d {
    pub fn to_array(self) -> [f64; 4] {
        [self[0][0], self[0][1], self[1][0], self[1][1]]
    }

    pub fn to_array_2d(self) -> [[f64; 2]; 2] {
        [[self[0][0], self[0][1]], [self[1][0], self[1][1]]]
    }

    pub fn to_mat3d(self) -> Mat3d {
        Mat3d::from(self)
    }

    pub fn to_mat4d(self) -> Mat4d {
        Mat4d::from(self)
    }

    pub fn to_mat2f(self) -> Mat2f {
        Mat2f::new(self.x_axis.to_vec2f(), self.y_axis.to_vec2f())
    }
}
