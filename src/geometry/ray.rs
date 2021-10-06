#![allow(dead_code)]

use std::fmt::Display;

use crate::algebra::{Mat4d, Transform, Transformable, Vec3d};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ray {
    origin: Vec3d,
    direction: Vec3d,
}

pub fn ray(origin: Vec3d, direction: Vec3d) -> Ray {
    Ray::new(origin, direction)
}

impl Display for Ray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ray(origin: ({}, {}, {}), direction: ({}, {}, {}))",
            self.origin.x(),
            self.origin.y(),
            self.origin.z(),
            self.direction.x(),
            self.direction.y(),
            self.direction.z()
        )
    }
}

impl Default for Ray {
    fn default() -> Self {
        Self {
            origin: Vec3d::default(),
            direction: Vec3d::default(),
        }
    }
}

impl Ray {
    pub fn new(origin: Vec3d, direction: Vec3d) -> Self {
        let direction = direction.normalize();
        Self { origin, direction }
    }

    pub fn at(&self, t: f64) -> Vec3d {
        self.origin + t * self.direction
    }
}

impl Ray {
    pub fn origin(&self) -> Vec3d {
        self.origin
    }

    pub fn set_origin(mut self, origin: Vec3d) -> Self {
        self.origin = origin;
        self
    }

    pub fn direction(&self) -> Vec3d {
        self.direction
    }

    pub fn set_direction(mut self, direction: Vec3d) -> Self {
        self.direction = direction;
        self
    }
}

impl Transformable for Ray {
    fn apply(&self, transform: Transform) -> Self {
        let origin = transform.transform_point3d(self.origin);
        let direction = transform.transform_vector3d(self.direction);
        Self::new(origin, direction)
    }

    fn apply_matrix4d(&self, matrix: Mat4d) -> Self {
        let t = Transform::from_matrix4d(matrix);
        let origin = t.transform_point3d(self.origin);
        let direction = t.transform_vector3d(self.direction);
        Self::new(origin, direction)
    }
}
