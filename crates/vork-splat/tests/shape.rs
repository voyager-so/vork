use glam::{Quat,Vec3};
use vork_splat::shape::Shape;
#[test]
fn rotated_extremely_flat_shapes_have_conservative_finite_bounds_and_axis_normals() {
    for scale in [Vec3::new(1e-8,2.,3.),Vec3::new(3.,1e-8,2.),Vec3::new(2.,3.,1e-8),Vec3::new(1e20,2e20,3e20)] {
        let rotation=Quat::from_rotation_y(0.7)*Quat::from_rotation_z(-1.1);
        let shape=Shape {position:Vec3::new(2.,-4.,7.),scale,rotation,opacity:0.7};
        let (lo,hi)=shape.bounds(3.).unwrap();
        for i in 0..1024 {
            let z=2.*(i as f32+0.5)/1024.-1.;let angle=i as f32*2.3999632;
            let radial=(1.-z*z).sqrt();let axis=Vec3::new(radial*angle.cos(),radial*angle.sin(),z);
            let point=shape.position+rotation*(axis*scale*3.);
            let error=scale.max_element()*2e-6;
            assert!(point.cmpge(lo-Vec3::splat(error)).all() && point.cmple(hi+Vec3::splat(error)).all());
        }
        let normal=rotation.conjugate()*shape.normal();
        let normal_scale=(normal*scale).length();
        if scale.min_element()<1. {assert!((normal_scale-scale.min_element()).abs()<2e-6);}
    }
}
