#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::algebra::{vec4f, Mat2f, Mat3f, Mat4d, Vec3f, Vec4f};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mat4f {
    x_axis: Vec4f,
    y_axis: Vec4f,
    z_axis: Vec4f,
    w_axis: Vec4f,
}

pub fn mat4f(
    m00: f32,
    m01: f32,
    m02: f32,
    m03: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m13: f32,
    m20: f32,
    m21: f32,
    m22: f32,
    m23: f32,
    m30: f32,
    m31: f32,
    m32: f32,
    m33: f32,
) -> Mat4f {
    Mat4f::from_array([
        m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33,
    ])
}

impl Display for Mat4f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (m00, m01, m02, m03) = self.x_axis.to_tuple();
        let (m10, m11, m12, m13) = self.y_axis.to_tuple();
        let (m20, m21, m22, m23) = self.z_axis.to_tuple();
        let (m30, m31, m32, m33) = self.w_axis.to_tuple();
        write!(
            f,
            "Mat4f({}, {}, {}, {} | {}, {}, {}, {} | {}, {}, {}, {} | {}, {}, {}, {})",
            m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33
        )
    }
}

impl Default for Mat4f {
    fn default() -> Self {
        Self {
            x_axis: Vec4f::default(),
            y_axis: Vec4f::default(),
            z_axis: Vec4f::default(),
            w_axis: Vec4f::default(),
        }
    }
}

impl Add<Mat4f> for Mat4f {
    type Output = Mat4f;

    fn add(self, rhs: Mat4f) -> Self::Output {
        Mat4f::new(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
            self.w_axis + rhs.w_axis,
        )
    }
}

impl Add<f32> for Mat4f {
    type Output = Mat4f;

    fn add(self, rhs: f32) -> Self::Output {
        Mat4f::new(
            self.x_axis + rhs,
            self.y_axis + rhs,
            self.z_axis + rhs,
            self.w_axis + rhs,
        )
    }
}

impl Add<Mat4f> for f32 {
    type Output = Mat4f;

    fn add(self, rhs: Mat4f) -> Self::Output {
        Mat4f::new(
            self + rhs.x_axis,
            self + rhs.y_axis,
            self + rhs.z_axis,
            self + rhs.w_axis,
        )
    }
}

impl AddAssign<Mat4f> for Mat4f {
    fn add_assign(&mut self, rhs: Mat4f) {
        *self = *self + rhs;
    }
}

impl AddAssign<f32> for Mat4f {
    fn add_assign(&mut self, rhs: f32) {
        *self = *self + rhs;
    }
}

impl Sub<Mat4f> for Mat4f {
    type Output = Mat4f;

    fn sub(self, rhs: Mat4f) -> Self::Output {
        Mat4f::new(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
            self.w_axis - rhs.w_axis,
        )
    }
}

impl Sub<f32> for Mat4f {
    type Output = Mat4f;

    fn sub(self, rhs: f32) -> Self::Output {
        Mat4f::new(
            self.x_axis - rhs,
            self.y_axis - rhs,
            self.z_axis - rhs,
            self.w_axis - rhs,
        )
    }
}

impl Sub<Mat4f> for f32 {
    type Output = Mat4f;

    fn sub(self, rhs: Mat4f) -> Self::Output {
        Mat4f::new(
            self - rhs.x_axis,
            self - rhs.y_axis,
            self - rhs.z_axis,
            self - rhs.w_axis,
        )
    }
}

impl SubAssign<Mat4f> for Mat4f {
    fn sub_assign(&mut self, rhs: Mat4f) {
        *self = *self - rhs;
    }
}

impl SubAssign<f32> for Mat4f {
    fn sub_assign(&mut self, rhs: f32) {
        *self = *self - rhs;
    }
}

impl Mul<Mat4f> for Mat4f {
    type Output = Mat4f;

    fn mul(self, rhs: Mat4f) -> Self::Output {
        let m00 = self
            .x_axis
            .dot(vec4f(rhs[0][0], rhs[1][0], rhs[2][0], rhs[3][0]));
        let m01 = self
            .x_axis
            .dot(vec4f(rhs[0][1], rhs[1][1], rhs[2][1], rhs[3][1]));
        let m02 = self
            .x_axis
            .dot(vec4f(rhs[0][2], rhs[1][2], rhs[2][2], rhs[3][2]));
        let m03 = self
            .x_axis
            .dot(vec4f(rhs[0][3], rhs[1][3], rhs[2][3], rhs[3][3]));

        let m10 = self
            .y_axis
            .dot(vec4f(rhs[0][0], rhs[1][0], rhs[2][0], rhs[3][0]));
        let m11 = self
            .y_axis
            .dot(vec4f(rhs[0][1], rhs[1][1], rhs[2][1], rhs[3][1]));
        let m12 = self
            .y_axis
            .dot(vec4f(rhs[0][2], rhs[1][2], rhs[2][2], rhs[3][2]));
        let m13 = self
            .y_axis
            .dot(vec4f(rhs[0][3], rhs[1][3], rhs[2][3], rhs[3][3]));

        let m20 = self
            .z_axis
            .dot(vec4f(rhs[0][0], rhs[1][0], rhs[2][0], rhs[3][0]));
        let m21 = self
            .z_axis
            .dot(vec4f(rhs[0][1], rhs[1][1], rhs[2][1], rhs[3][1]));
        let m22 = self
            .z_axis
            .dot(vec4f(rhs[0][2], rhs[1][2], rhs[2][2], rhs[3][2]));
        let m23 = self
            .z_axis
            .dot(vec4f(rhs[0][3], rhs[1][3], rhs[2][3], rhs[3][3]));

        let m30 = self
            .w_axis
            .dot(vec4f(rhs[0][0], rhs[1][0], rhs[2][0], rhs[3][0]));
        let m31 = self
            .w_axis
            .dot(vec4f(rhs[0][1], rhs[1][1], rhs[2][1], rhs[3][1]));
        let m32 = self
            .w_axis
            .dot(vec4f(rhs[0][2], rhs[1][2], rhs[2][2], rhs[3][2]));
        let m33 = self
            .w_axis
            .dot(vec4f(rhs[0][3], rhs[1][3], rhs[2][3], rhs[3][3]));

        Mat4f::new(
            vec4f(m00, m01, m02, m03),
            vec4f(m10, m11, m12, m13),
            vec4f(m20, m21, m22, m23),
            vec4f(m30, m31, m32, m33),
        )
    }
}

impl Mul<Vec4f> for Mat4f {
    type Output = Vec4f;

    fn mul(self, rhs: Vec4f) -> Self::Output {
        let v0 = self.x_axis.dot(rhs);
        let v1 = self.y_axis.dot(rhs);
        let v2 = self.z_axis.dot(rhs);
        let v3 = self.w_axis.dot(rhs);
        vec4f(v0, v1, v2, v3)
    }
}

impl Mul<f32> for Mat4f {
    type Output = Mat4f;

    fn mul(self, rhs: f32) -> Self::Output {
        Mat4f::new(
            self.x_axis * rhs,
            self.y_axis * rhs,
            self.z_axis * rhs,
            self.w_axis * rhs,
        )
    }
}

impl Mul<Mat4f> for f32 {
    type Output = Mat4f;

    fn mul(self, rhs: Mat4f) -> Self::Output {
        Mat4f::new(
            self * rhs.x_axis,
            self * rhs.y_axis,
            self * rhs.z_axis,
            self * rhs.w_axis,
        )
    }
}

impl MulAssign<Mat4f> for Mat4f {
    fn mul_assign(&mut self, rhs: Mat4f) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Mat4f {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<f32> for Mat4f {
    type Output = Mat4f;

    fn div(self, rhs: f32) -> Self::Output {
        Mat4f::new(
            self.x_axis / rhs,
            self.y_axis / rhs,
            self.z_axis / rhs,
            self.w_axis / rhs,
        )
    }
}

impl DivAssign<f32> for Mat4f {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Index<usize> for Mat4f {
    type Output = Vec4f;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x_axis,
            1 => &self.y_axis,
            2 => &self.z_axis,
            3 => &self.w_axis,
            _ => panic!("`rmath::algebra::Mat4f::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Mat4f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            3 => &mut self.w_axis,
            _ => panic!("`rmath::algebra::Mat4f::index_mut`: index out of bounds."),
        }
    }
}

impl Mat4f {
    pub fn new(x_axis: Vec4f, y_axis: Vec4f, z_axis: Vec4f, w_axis: Vec4f) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    pub fn zero() -> Self {
        Self::new(
            vec4f(0.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn identity() -> Self {
        Self::new(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }
}

impl Mat4f {
    pub fn col(&self, index: usize) -> Vec4f {
        match index {
            0 => vec4f(self[0][0], self[1][0], self[2][0], self[3][0]),
            1 => vec4f(self[0][1], self[1][1], self[2][1], self[3][1]),
            2 => vec4f(self[0][2], self[1][2], self[2][2], self[3][2]),
            3 => vec4f(self[0][3], self[1][3], self[2][3], self[3][3]),
            _ => panic!("`rmath::algebra::Mat4f::col`: index out of bounds."),
        }
    }

    pub fn row(&self, index: usize) -> Vec4f {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            3 => self.w_axis,
            _ => panic!("`rmath::algebra::Mat4f::row`: index out of bounds."),
        }
    }

    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan() || self.z_axis.is_nan() || self.w_axis.is_nan()
    }

    pub fn is_infinite(&self) -> bool {
        self.x_axis.is_infinite()
            || self.y_axis.is_infinite()
            || self.z_axis.is_infinite()
            || self.z_axis.is_infinite()
    }

    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite()
            && self.y_axis.is_finite()
            && self.z_axis.is_finite()
            && self.w_axis.is_finite()
    }

    pub fn transpose(&self) -> Self {
        let (m00, m01, m02, m03) = self.x_axis.to_tuple();
        let (m10, m11, m12, m13) = self.y_axis.to_tuple();
        let (m20, m21, m22, m23) = self.z_axis.to_tuple();
        let (m30, m31, m32, m33) = self.w_axis.to_tuple();
        Self::new(
            vec4f(m00, m10, m20, m30),
            vec4f(m01, m11, m21, m31),
            vec4f(m02, m12, m22, m32),
            vec4f(m03, m13, m23, m33),
        )
    }

    pub fn determinant(&self) -> f32 {
        let (m00, m01, m02, m03) = self.x_axis.to_tuple();
        let (m10, m11, m12, m13) = self.y_axis.to_tuple();
        let (m20, m21, m22, m23) = self.z_axis.to_tuple();
        let (m30, m31, m32, m33) = self.w_axis.to_tuple();

        let c23 = m22 * m33 - m23 * m32;
        let c13 = m21 * m33 - m23 * m31;
        let c12 = m21 * m32 - m22 * m31;
        let c03 = m20 * m33 - m23 * m30;
        let c02 = m20 * m32 - m22 * m30;
        let c01 = m20 * m31 - m21 * m30;

        let a00 = m11 * c23 - m12 * c13 + m13 * c12;
        let a01 = m10 * c23 - m12 * c03 + m13 * c02;
        let a02 = m10 * c13 - m11 * c03 + m13 * c01;
        let a03 = m10 * c12 - m11 * c02 + m12 * c01;

        m00 * a00 - m01 * a01 + m02 * a02 - m03 * a03
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();
        ruby_assert!(!(det.abs() < f32::EPSILON));

        let (m00, m01, m02, m03) = self.x_axis.to_tuple();
        let (m10, m11, m12, m13) = self.y_axis.to_tuple();
        let (m20, m21, m22, m23) = self.z_axis.to_tuple();
        let (m30, m31, m32, m33) = self.w_axis.to_tuple();

        let c1201 = m10 * m21 - m11 * m20;
        let c1202 = m10 * m22 - m12 * m20;
        let c1203 = m10 * m23 - m13 * m20;
        let c1212 = m11 * m22 - m12 * m21;
        let c1213 = m11 * m23 - m13 * m21;
        let c1223 = m12 * m23 - m13 * m22;

        let c1301 = m10 * m31 - m11 * m30;
        let c1302 = m10 * m32 - m12 * m30;
        let c1303 = m10 * m33 - m13 * m30;
        let c1312 = m11 * m32 - m12 * m31;
        let c1313 = m11 * m33 - m13 * m31;
        let c1323 = m12 * m33 - m13 * m32;

        let c2301 = m20 * m31 - m21 * m30;
        let c2302 = m20 * m32 - m22 * m30;
        let c2303 = m20 * m33 - m23 * m30;
        let c2312 = m21 * m32 - m22 * m31;
        let c2313 = m21 * m33 - m22 * m31;
        let c2323 = m22 * m33 - m23 * m32;

        let a00 = m11 * c2323 - m12 * c2313 + m13 * c2312;
        let a01 = -(m10 * c2323 - m12 * c2303 + m13 * c2302);
        let a02 = m10 * c2313 - m11 * c2303 + m13 * c2301;
        let a03 = -(m10 * c2312 - m11 * c2302 + m12 * c2301);

        let a10 = -(m01 * c2323 - m02 * c2313 + m03 * c2312);
        let a11 = m00 * c2323 - m02 * c2303 + m03 * c2302;
        let a12 = -(m00 * c2313 - m01 * c2303 + m03 * c2301);
        let a13 = m00 * c2312 - m01 * c2302 + m02 * c2301;

        let a20 = m01 * c1323 - m02 * c1313 + m03 * c1312;
        let a21 = -(m00 * c1323 - m02 * c1303 + m03 * c1302);
        let a22 = m00 * c1313 - m01 * c1303 + m03 * c1301;
        let a23 = -(m00 * c1312 - m01 * c1302 + m02 * c1301);

        let a30 = -(m01 * c1223 - m02 * c1213 + m03 * c1212);
        let a31 = m00 * c1223 - m02 * c1203 + m03 * c1202;
        let a32 = -(m00 * c1213 - m01 * c1203 + m03 * c1201);
        let a33 = m00 * c1212 - m01 * c1202 + m02 * c1201;

        Self::new(
            vec4f(a00, a10, a20, a30),
            vec4f(a01, a11, a21, a31),
            vec4f(a02, a12, a22, a32),
            vec4f(a03, a13, a23, a33),
        )
        .mul(det.recip())
    }

    pub fn tray_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }

        let (m00, m01, m02, m03) = self.x_axis.to_tuple();
        let (m10, m11, m12, m13) = self.y_axis.to_tuple();
        let (m20, m21, m22, m23) = self.z_axis.to_tuple();
        let (m30, m31, m32, m33) = self.w_axis.to_tuple();

        let c1201 = m10 * m21 - m11 * m20;
        let c1202 = m10 * m22 - m12 * m20;
        let c1203 = m10 * m23 - m13 * m20;
        let c1212 = m11 * m22 - m12 * m21;
        let c1213 = m11 * m23 - m13 * m21;
        let c1223 = m12 * m23 - m13 * m22;

        let c1301 = m10 * m31 - m11 * m30;
        let c1302 = m10 * m32 - m12 * m30;
        let c1303 = m10 * m33 - m13 * m30;
        let c1312 = m11 * m32 - m12 * m31;
        let c1313 = m11 * m33 - m13 * m31;
        let c1323 = m12 * m33 - m13 * m32;

        let c2301 = m20 * m31 - m21 * m30;
        let c2302 = m20 * m32 - m22 * m30;
        let c2303 = m20 * m33 - m23 * m30;
        let c2312 = m21 * m32 - m22 * m31;
        let c2313 = m21 * m33 - m22 * m31;
        let c2323 = m22 * m33 - m23 * m32;

        let a00 = m11 * c2323 - m12 * c2313 + m13 * c2312;
        let a01 = -(m10 * c2323 - m12 * c2303 + m13 * c2302);
        let a02 = m10 * c2313 - m11 * c2303 + m13 * c2301;
        let a03 = -(m10 * c2312 - m11 * c2302 + m12 * c2301);

        let a10 = -(m01 * c2323 - m02 * c2313 + m03 * c2312);
        let a11 = m00 * c2323 - m02 * c2303 + m03 * c2302;
        let a12 = -(m00 * c2313 - m01 * c2303 + m03 * c2301);
        let a13 = m00 * c2312 - m01 * c2302 + m02 * c2301;

        let a20 = m01 * c1323 - m02 * c1313 + m03 * c1312;
        let a21 = -(m00 * c1323 - m02 * c1303 + m03 * c1302);
        let a22 = m00 * c1313 - m01 * c1303 + m03 * c1301;
        let a23 = -(m00 * c1312 - m01 * c1302 + m02 * c1301);

        let a30 = -(m01 * c1223 - m02 * c1213 + m03 * c1212);
        let a31 = m00 * c1223 - m02 * c1203 + m03 * c1202;
        let a32 = -(m00 * c1213 - m01 * c1203 + m03 * c1201);
        let a33 = m00 * c1212 - m01 * c1202 + m02 * c1201;

        Some(
            Self::new(
                vec4f(a00, a10, a20, a30),
                vec4f(a01, a11, a21, a31),
                vec4f(a02, a12, a22, a32),
                vec4f(a03, a13, a23, a33),
            )
            .mul(det.recip()),
        )
    }
}

impl From<Mat2f> for Mat4f {
    fn from(m: Mat2f) -> Self {
        mat4f(
            m[0][0], m[0][1], 0.0, 0.0, m[1][0], m[1][1], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        )
    }
}

impl From<Mat3f> for Mat4f {
    fn from(m: Mat3f) -> Self {
        mat4f(
            m[0][0], m[0][1], m[0][2], 0.0, m[1][0], m[1][1], m[1][2], 0.0, m[2][0], m[2][1],
            m[2][2], 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }
}

impl Mat4f {
    pub fn from_array(m: [f32; 16]) -> Self {
        Self::new(
            vec4f(m[0], m[1], m[2], m[3]),
            vec4f(m[4], m[5], m[6], m[7]),
            vec4f(m[8], m[9], m[10], m[11]),
            vec4f(m[12], m[13], m[14], m[15]),
        )
    }

    pub fn from_array_2d(m: [[f32; 4]; 4]) -> Self {
        Self::new(
            vec4f(m[0][0], m[0][1], m[0][2], m[0][3]),
            vec4f(m[1][0], m[1][1], m[1][2], m[1][3]),
            vec4f(m[2][0], m[2][1], m[2][2], m[2][3]),
            vec4f(m[3][0], m[3][1], m[3][2], m[3][3]),
        )
    }

    pub fn from_diagonal(diagonal: Vec4f) -> Self {
        Self::new(
            vec4f(diagonal.x(), 0.0, 0.0, 0.0),
            vec4f(0.0, diagonal.y(), 0.0, 0.0),
            vec4f(0.0, 0.0, diagonal.z(), 0.0),
            vec4f(0.0, 0.0, 0.0, diagonal.w()),
        )
    }
}

impl Mat4f {
    pub fn to_array(self) -> [f32; 16] {
        [
            self[0][0], self[0][1], self[0][2], self[0][3], self[1][0], self[1][1], self[1][2],
            self[1][3], self[2][0], self[2][1], self[2][2], self[2][3], self[3][0], self[3][1],
            self[3][2], self[3][3],
        ]
    }

    pub fn to_array_2d(self) -> [[f32; 4]; 4] {
        [
            [self[0][0], self[0][1], self[0][2], self[0][3]],
            [self[1][0], self[1][1], self[1][2], self[1][3]],
            [self[2][0], self[2][1], self[2][2], self[2][3]],
            [self[3][0], self[3][1], self[3][2], self[3][3]],
        ]
    }

    pub fn to_mat2f(self) -> Mat2f {
        Mat2f::from(self)
    }

    pub fn to_mat3f(self) -> Mat3f {
        Mat3f::from(self)
    }

    pub fn to_mat4d(self) -> Mat4d {
        Mat4d::new(
            self.x_axis.to_vec4d(),
            self.y_axis.to_vec4d(),
            self.z_axis.to_vec4d(),
            self.w_axis.to_vec4d(),
        )
    }
}

impl Mat4f {
    pub fn translation(translate: Vec3f) -> Self {
        Self::new(
            vec4f(1.0, 0.0, 0.0, translate.x()),
            vec4f(0.0, 1.0, 0.0, translate.y()),
            vec4f(0.0, 0.0, 1.0, translate.z()),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn rotation_x(angle: f32) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        Self::new(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, cos, -sin, 0.0),
            vec4f(0.0, sin, cos, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn rotation_y(angle: f32) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        Self::new(
            vec4f(cos, 0.0, sin, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(-sin, 0.0, cos, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn rotation_z(angle: f32) -> Self {
        let (sin, cos) = angle.to_radians().sin_cos();
        Self::new(
            vec4f(cos, -sin, 0.0, 0.0),
            vec4f(sin, cos, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn rotation_axis(axis: Vec3f, angle: f32) -> Self {
        let axis = axis.normalize();
        let (sin, cos) = angle.to_radians().sin_cos();
        let rotation = cos * Self::identity()
            + (1.0 - cos)
                * Self::new(
                    axis.x().mul(axis).to_homogeneous_coord_vector(),
                    axis.y().mul(axis).to_homogeneous_coord_vector(),
                    axis.z().mul(axis).to_homogeneous_coord_vector(),
                    Vec4f::zero(),
                )
            + sin
                * Self::new(
                    vec4f(0.0, -axis.z(), axis.y(), 0.0),
                    vec4f(axis.z(), 0.0, -axis.x(), 0.0),
                    vec4f(-axis.y(), axis.x(), 0.0, 0.0),
                    Vec4f::zero(),
                );
        rotation
    }

    pub fn scale(scale: Vec3f) -> Self {
        Self::new(
            vec4f(scale.x(), 0.0, 0.0, 0.0),
            vec4f(0.0, scale.y(), 0.0, 0.0),
            vec4f(0.0, 0.0, scale.z(), 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn look_at(eye: Vec3f, target: Vec3f, up: Vec3f) -> Self {
        let g = (target - eye).normalize();
        let u = up.normalize();
        let s = g.cross(u).normalize();
        let u = s.cross(g);
        let g = -g;

        Self::new(
            vec4f(s.x(), s.y(), s.z(), s.dot(-eye)),
            vec4f(u.x(), u.y(), u.z(), u.dot(-eye)),
            vec4f(g.x(), g.y(), g.z(), g.dot(-eye)),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn perspective(fovy: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        ruby_assert!(aspect.abs() > f32::EPSILON);

        let (n, f) = (z_near, z_far);
        let tan_half_theta = fovy.to_radians().mul(0.5).tan();

        let perspective = Self::new(
            vec4f(tan_half_theta.mul(aspect).recip(), 0.0, 0.0, 0.0),
            vec4f(0.0, tan_half_theta.recip(), 0.0, 0.0),
            vec4f(0.0, 0.0, -(n + f) / (f - n), -2.0 * n * f / (f - n)),
            vec4f(0.0, 0.0, -1.0, 0.0),
        );

        perspective
    }
}
