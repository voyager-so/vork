//! Shape operations shared by import/install and numerical conformance tests.
use anyhow::{Result, ensure};
use glam::{Mat3, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
    pub opacity: f32,
}

impl Shape {
    /// Tight bounds of the rotated ellipsoid at the renderer's support radius.
    pub fn bounds(&self, radius: f32) -> Result<(Vec3, Vec3)> {
        let r = Mat3::from_quat(self.rotation).as_dmat3();
        let scale=self.scale.as_dvec3();
        let axes = [r.x_axis*scale.x, r.y_axis*scale.y, r.z_axis*scale.z];
        let square = axes[0]*axes[0] + axes[1]*axes[1] + axes[2]*axes[2];
        let extent = glam::DVec3::from_array(square.to_array().map(f64::sqrt))*f64::from(radius);
        let lo=(self.position.as_dvec3()-extent).as_vec3(); let hi=(self.position.as_dvec3()+extent).as_vec3();
        ensure!(lo.is_finite() && hi.is_finite(), "Gaussian splat bounds exceed finite precision; rescale the capture");
        Ok((lo,hi))
    }

    pub fn normal(&self) -> Vec3 {
        self.rotation * if self.scale.x <= self.scale.y && self.scale.x <= self.scale.z {Vec3::X}
            else if self.scale.y <= self.scale.z {Vec3::Y} else {Vec3::Z}
    }

}
