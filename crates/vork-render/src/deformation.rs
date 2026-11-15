//! CPU position evaluation shared by renderer subdivision and scene read-back.
//! Influence normalization also owns the renderer's GPU upload contract.
use crate::animation::Influence;
use anyhow::{Result, ensure};
use glam::{Mat4, Vec3};

pub(crate) fn normalize(
    influences: &mut [Influence],
    count: usize,
    sets: usize,
    joints: usize,
) -> Result<()> {
    ensure!(
        joints > 0 && sets > 0 && influences.len() == count * sets,
        "invalid deformation influences"
    );
    for vertex in influences.chunks_mut(sets) {
        let total: f32 = vertex.iter().flat_map(|v| v.WEIGHTS_0).sum();
        ensure!(
            total.is_finite() && total > 0.,
            "skin weights require a finite positive sum"
        );
        for influence in vertex {
            ensure!(
                influence.JOINTS_0.iter().all(|j| (*j as usize) < joints)
                    && influence
                        .WEIGHTS_0
                        .iter()
                        .all(|w| w.is_finite() && *w >= 0.),
                "invalid skin influence"
            );
            for weight in &mut influence.WEIGHTS_0 {
                *weight /= total;
            }
        }
    }
    Ok(())
}

/// Morph first, then skin with normalized, vertex-major influence sets.
/// This is the existing subdivision cage evaluator; callers validate array shapes.
pub(crate) fn positions(
    source: &[[f32; 3]],
    influences: &[Influence],
    sets: usize,
    morphs: &[[f32; 3]],
    matrices: &[Mat4],
    weights: &[f32],
) -> Vec<[f32; 3]> {
    let count = source.len();
    debug_assert_eq!(influences.len(), count * sets);
    debug_assert_eq!(morphs.len(), count * weights.len());
    source
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut p = Vec3::from(*p);
            for (target, &weight) in weights.iter().enumerate() {
                if weight != 0. {
                    p += Vec3::from(morphs[target * count + i]) * weight;
                }
            }
            let mut out = Vec3::ZERO;
            for inf in &influences[i * sets..(i + 1) * sets] {
                for (&joint, &weight) in inf.JOINTS_0.iter().zip(&inf.WEIGHTS_0) {
                    if weight != 0. {
                        out += matrices[joint as usize].transform_point3(p) * weight;
                    }
                }
            }
            out.to_array()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_influence_sets_share_normalization_and_morph_precedes_skin() -> Result<()> {
        let mut influences = vec![
            Influence {
                JOINTS_0: [0; 4],
                WEIGHTS_0: [2., 0., 0., 0.],
            },
            Influence {
                JOINTS_0: [1; 4],
                WEIGHTS_0: [6., 0., 0., 0.],
            },
        ];
        normalize(&mut influences, 1, 2, 2)?;
        let matrices = [
            Mat4::from_scale(Vec3::splat(2.)),
            Mat4::from_translation(Vec3::new(8., 0., 0.)),
        ];
        let result = positions(
            &[[1., 0., 0.]],
            &influences,
            2,
            &[[2., 0., 0.]],
            &matrices,
            &[0.5],
        );
        assert_eq!(result, vec![[8.5, 0., 0.]]);
        for weights in [[0.; 4], [-1., 2., 0., 0.], [f32::NAN, 0., 0., 0.]] {
            let mut bad = [Influence {
                JOINTS_0: [0; 4],
                WEIGHTS_0: weights,
            }];
            assert!(normalize(&mut bad, 1, 1, 1).is_err());
        }
        Ok(())
    }
}
