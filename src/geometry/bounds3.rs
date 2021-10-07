#![allow(dead_code)]

use std::{
    fmt::Display,
    mem::swap,
    ops::{Index, IndexMut},
};

use crate::{
    algebra::{vec3d, Mat4d, Transform, Transformable, Vec2d, Vec3d},
    geometry::Ray,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Bounds3 {
    min: Vec3d,
    max: Vec3d,
}

impl Display for Bounds3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bounds3(min: {}, max: {})", self.min, self.max)
    }
}

impl Default for Bounds3 {
    fn default() -> Self {
        Self {
            min: Vec3d::default(),
            max: Vec3d::default(),
        }
    }
}

impl Index<usize> for Bounds3 {
    type Output = Vec3d;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.min,
            1 => &self.max,
            _ => panic!("`rmath::geometry::Bounds3::index`: index out of bounds."),
        }
    }
}

impl IndexMut<usize> for Bounds3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.min,
            1 => &mut self.max,
            _ => panic!("`rmath::geometry::Bounds3::index`: index out of bounds."),
        }
    }
}

impl Bounds3 {
    pub fn new(p1: Vec3d, p2: Vec3d) -> Self {
        Self {
            min: vec3d(p1.x().min(p2.x()), p1.y().min(p2.y()), p1.z().min(p2.z())),
            max: vec3d(p1.x().max(p2.x()), p1.y().max(p2.y()), p1.z().max(p2.z())),
        }
    }

    pub fn corner(&self, index: usize) -> Vec3d {
        vec3d(
            (*self)[index & 1].x(),
            (*self)[((index & 2) != 0) as usize].y(),
            (*self)[((index & 4) != 0) as usize].z(),
        )
    }

    pub fn union_point(&self, point: Vec3d) -> Self {
        Self::new(self.min.min(point), self.max.max(point))
    }

    pub fn union_bounds(&self, bounds: Bounds3) -> Self {
        Self::new(self.min.min(bounds.min), self.max.max(bounds.max))
    }

    pub fn is_overlapped(&self, bounds: Bounds3) -> bool {
        let x = self.max.x() >= bounds.min.x() && self.min.x() <= bounds.max.x();
        let y = self.max.y() >= bounds.min.y() && self.min.y() <= bounds.max.y();
        let z = self.max.z() >= bounds.min.z() && self.min.z() <= bounds.max.z();
        x && y && z
    }

    pub fn is_inside(&self, point: Vec3d) -> bool {
        (point.x() >= self.min.x() && point.x() <= self.max.x())
            && (point.y() >= self.min.y() && point.y() <= self.max.y())
            && (point.z() >= self.min.z() && point.z() <= self.max.z())
    }

    pub fn expand(&self, delta: f64) -> Self {
        Self::new(
            self.min - vec3d(delta, delta, delta),
            self.max + vec3d(delta, delta, delta),
        )
    }

    pub fn diagonal(&self) -> Vec3d {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f64 {
        let diag = self.diagonal();
        2.0 * diag.xyz().dot(diag.zxy())
    }

    pub fn volume(&self) -> f64 {
        let diag = self.diagonal();
        diag.x() * diag.y() * diag.z()
    }

    pub fn lerp(&self, t: Vec3d) -> Vec3d {
        (1.0 - t) * self.min + t * self.max
    }

    pub fn offset(&self, point: Vec3d) -> Vec3d {
        let mut o = point - self.min;
        for i in 0..3 {
            if self.max[i] > self.min[i] {
                o[i] /= self.max[i] - self.min[i];
            }
        }
        o
    }

    pub fn intersect(&self, ray: Ray, range: Vec2d) -> bool {
        let (mut tmin, mut tmax) = range.to_tuple();
        let inv_dir = ray.direction().recip();
        for i in 0..3 {
            let inv = inv_dir[i];
            let mut t0 = (self.min[i] - ray.origin()[i]) * inv;
            let mut t1 = (self.max[i] - ray.origin()[i]) * inv;
            if inv < 0.0 {
                swap(&mut t0, &mut t1);
            }
            tmin = t0.max(tmin);
            tmax = t1.min(tmax);
            if tmax <= tmin {
                return false;
            }
        }
        return true;
    }

    pub fn min(&self) -> Vec3d {
        self.min
    }

    pub fn max(&self) -> Vec3d {
        self.max
    }
}

impl Transformable for Bounds3 {
    fn apply(&self, transform: Transform) -> Self {
        let p = transform.transform_point3d((*self)[0]);
        Self::new(p, p)
            .union_point(transform.transform_point3d((*self)[1]))
            .union_point(transform.transform_point3d((*self)[2]))
            .union_point(transform.transform_point3d((*self)[3]))
            .union_point(transform.transform_point3d((*self)[4]))
            .union_point(transform.transform_point3d((*self)[5]))
            .union_point(transform.transform_point3d((*self)[6]))
            .union_point(transform.transform_point3d((*self)[7]))
    }

    fn apply_matrix4d(&self, matrix: Mat4d) -> Self {
        let t = Transform::from_matrix4d(matrix);
        self.apply(t)
    }
}
