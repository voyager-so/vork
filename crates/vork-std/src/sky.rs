//! Author-side solar disc declaration for std/sky's scattering atmosphere.
use anyhow::{Result, ensure};
use glam::{Quat, Vec3};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Inputs {
    name: String,
    sun_elevation: f32,
    sun_rotation: f32,
    altitude: f32,
    air_density: f32,
    aerosol_density: f32,
    ozone_density: f32,
    sun_size: f32,
    sun_intensity: f32,
}
impl Default for Inputs {
    fn default() -> Self {
        Self {
            name: "sky".into(),
            sun_elevation: 0.261799,
            sun_rotation: 0.,
            altitude: 100.,
            air_density: 1.,
            aerosol_density: 1.,
            ozone_density: 1.,
            sun_size: 0.00951204,
            sun_intensity: 1.,
        }
    }
}
pub fn evaluate(argument: &str) -> Result<String> {
    let i: Inputs = serde_json::from_str(argument)?;
    ensure!(
        !i.name.is_empty()
            && [
                i.sun_elevation,
                i.sun_rotation,
                i.altitude,
                i.air_density,
                i.aerosol_density,
                i.ozone_density,
                i.sun_size,
                i.sun_intensity
            ]
            .iter()
            .all(|v| v.is_finite()),
        "sky inputs must be finite and named"
    );
    ensure!(
        (0. ..=100000.).contains(&i.altitude)
            && [
                i.air_density,
                i.aerosol_density,
                i.ozone_density,
                i.sun_intensity
            ]
            .iter()
            .all(|v| (0. ..=1000.).contains(v))
            && (0. ..=std::f32::consts::FRAC_PI_2).contains(&i.sun_size),
        "sky inputs outside std/sky ranges"
    );
    let direction = Vec3::new(
        i.sun_elevation.cos() * i.sun_rotation.sin(),
        i.sun_elevation.sin(),
        -i.sun_elevation.cos() * i.sun_rotation.cos(),
    );
    let origin = Vec3::new(0., 6360000. + i.altitude.max(1.), 0.);
    let b = origin.dot(direction);
    let h = origin.length() - 6360000.;
    let transmission = if b < 0. && b * b - h * (origin.length() + 6360000.) > 0. {
        Vec3::ZERO
    } else {
        let ht = origin.length() - 6460000.;
        let distance = (-b + (b * b - ht * (origin.length() + 6460000.)).max(0.).sqrt()).max(0.);
        let mut optical = Vec3::ZERO;
        for k in 0..12 {
            let a = k as f32 / 12.;
            let b = (k + 1) as f32 / 12.;
            let p = origin + direction * (distance * (a * a + b * b) * 0.5);
            let height = (p.length() - 6360000.).max(0.);
            let rho = Vec3::new(
                (-height / 8000.).exp() * i.air_density,
                (-height / 1200.).exp() * i.aerosol_density,
                (1. - (height - 25000.).abs() / 15000.).max(0.) * i.ozone_density,
            );
            optical += (Vec3::new(5.802e-6, 13.558e-6, 33.100e-6) * rho.x
                + Vec3::splat(4.440e-6) * rho.y
                + Vec3::new(0.650e-6, 1.881e-6, 0.085e-6) * rho.z)
                * distance
                * (b * b - a * a);
        }
        Vec3::new((-optical.x).exp(), (-optical.y).exp(), (-optical.z).exp())
    };
    let energy = Vec3::new(147.4, 185.04, 191.198)
        * transmission
        * i.sun_intensity
        * (i.sun_size * 0.25).cos().powi(2);
    let intensity = if i.sun_size > 0. {
        energy.max_element()
    } else {
        0.
    };
    let color = if intensity > 0. {
        (energy / intensity).to_array()
    } else {
        [0.; 3]
    };
    Ok(serde_json::to_string(&serde_json::json!({"materials":{i.name.clone():{"program":"std/sky","inputs":{"sun_elevation":i.sun_elevation,"sun_rotation":i.sun_rotation,"altitude":i.altitude,"air_density":i.air_density,"aerosol_density":i.aerosol_density,"ozone_density":i.ozone_density}}},"nodes":[{"type":"light","name":i.name,"shape":"environment","material":i.name},{"type":"light","name":format!("{}_sun",i.name),"shape":"sun","rotation":Quat::from_rotation_arc(Vec3::NEG_Z,-direction).to_array(),"angle":i.sun_size,"color":color,"energy":intensity}]}))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solar_declarations_follow_azimuth_and_preserve_solar_irradiance() -> Result<()> {
        for rotation in [0., std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -0.7] {
            let result:serde_json::Value=serde_json::from_str(&evaluate(&serde_json::json!({"sun_elevation":0.5,"sun_rotation":rotation,"air_density":0,"aerosol_density":0,"ozone_density":0}).to_string())?)?;
            let sun = &result["nodes"][1];
            let q: [f32; 4] = serde_json::from_value(sun["rotation"].clone())?;
            let expected = Vec3::new(
                0.5f32.cos() * rotation.sin(),
                0.5f32.sin(),
                -0.5f32.cos() * rotation.cos(),
            );
            assert!((-(Quat::from_array(q) * Vec3::NEG_Z) - expected).length() < 1e-6);
            let intensity = sun["energy"].as_f64().unwrap();
            let angle = sun["angle"].as_f64().unwrap();
            let rgb: Vec<f32> = serde_json::from_value(sun["color"].clone())?;
            for k in 0..3 {
                assert!(
                    (rgb[k] as f64 * intensity / (angle * 0.25).cos().powi(2)
                        - [147.4, 185.04, 191.198][k])
                        .abs()
                        < 0.0001
                );
            }
        }
        Ok(())
    }
}
