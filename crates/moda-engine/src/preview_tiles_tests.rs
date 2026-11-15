use super::{MotionCache, MotionKey, PREVIEW_TILE_SIZE};
use crate::frame::CameraParams;

fn camera() -> CameraParams {
    CameraParams {
        camera_x: -17.25,
        camera_y: 43.125,
        zoom: 0.75,
        viewport_w: 641.2,
        viewport_h: 401.3,
        dpr: 1.25,
        clear_rgb: [0.95, 0.95, 0.96],
    }
}

#[test]
fn fixed_tiles_are_off_by_default_and_switching_drops_the_lattice() {
    let mut cache = MotionCache::default();
    let camera = camera();
    let key = MotionKey::new(1, &camera);
    assert!(!cache.preview_tiles_enabled());
    assert!(cache.begin_anchor(key, &camera));
    let original = cache.next_job(&key, &[camera]).unwrap();
    assert_eq!(original.present.width_px, key.width_px);
    cache.set_preview_tiles_enabled(true);
    assert!(!cache.anchored_for(&key));
    assert!(cache.begin_anchor(key, &camera));
    let job = cache.next_job(&key, &[camera]).unwrap();
    assert_eq!(job.present.width_px, PREVIEW_TILE_SIZE);
    assert_eq!(job.present.height_px, PREVIEW_TILE_SIZE);
    assert_ne!(original.generation, job.generation);
    cache.set_preview_tiles_enabled(true);
    assert!(cache.anchored_for(&key));
    cache.set_preview_tiles_enabled(false);
    assert!(!cache.anchored_for(&key));
}

#[test]
fn fixed_grid_preserves_fractional_dpr_scale_and_negative_coordinates() {
    let mut cache = MotionCache::default();
    cache.set_preview_tiles_enabled(true);
    let camera = camera();
    let key = MotionKey::new(1, &camera);
    assert!(cache.begin_anchor(key, &camera));
    let anchor = *cache.anchor().unwrap();
    let (scale_x, scale_y) = key.scale();
    assert_eq!((anchor.camera_x * scale_x).round(), -512.0);
    assert_eq!((anchor.camera_y * scale_y).round(), 0.0);
    let job = cache.job((-2, 1));
    assert_eq!(job.present.width_px, 512);
    assert_eq!(job.present.height_px, 512);
    let tile_scale_x = f64::from(job.present.width_px) * camera.zoom / job.present.view.viewport_w;
    let tile_scale_y = f64::from(job.present.height_px) * camera.zoom / job.present.view.viewport_h;
    assert!((scale_x - tile_scale_x).abs() < 1e-12);
    assert!((scale_y - tile_scale_y).abs() < 1e-12);
    assert_eq!(cache.tile_origin((-2, 1)), (-1024, 512));
    assert_eq!(cache.tiles_at((0, 0)), ((0, 1), (0, 0)));
    assert_eq!(cache.tiles_at((-1, -1)), ((-1, 1), (-1, 0)));
    let region = cache.tile_region(job.tile);
    assert!(region[0] < job.present.view.camera_x);
    assert!(region[1] < job.present.view.camera_y);
}

#[test]
fn changed_content_resources_view_and_clear_reject_old_identity() {
    let mut cache = MotionCache::default();
    cache.set_preview_tiles_enabled(true);
    let camera = camera();
    let key = MotionKey::new(1, &camera);
    assert!(cache.begin_anchor(key, &camera));
    let changes = [
        MotionKey { content: 2, ..key },
        MotionKey { zoom: 1.0, ..key },
        MotionKey { dpr: 2.0, ..key },
        MotionKey {
            viewport_w: 700.0,
            ..key
        },
        MotionKey {
            width_px: 900,
            ..key
        },
        MotionKey {
            clear_rgb: [0.0; 3],
            ..key
        },
    ];
    for changed in changes {
        assert!(!cache.anchored_for(&changed));
        assert!(cache.offset(&changed, &camera).is_none());
        assert!(cache.next_job(&changed, &[camera]).is_none());
    }
    let job = cache.next_job(&key, &[camera]).unwrap();
    cache.clear();
    assert!(cache.begin_anchor(key, &camera));
    assert_ne!(job.generation, cache.generation);
}

#[test]
fn malformed_or_uncoverable_views_refuse_fixed_tiles() {
    let mut cache = MotionCache::default();
    cache.set_preview_tiles_enabled(true);
    let camera = camera();
    for malformed in [
        CameraParams {
            zoom: 0.0,
            ..camera
        },
        CameraParams {
            dpr: f64::NAN,
            ..camera
        },
        CameraParams {
            viewport_w: 0.0,
            ..camera
        },
        CameraParams {
            camera_x: f64::INFINITY,
            ..camera
        },
        CameraParams {
            viewport_w: 16384.0,
            viewport_h: 16384.0,
            ..camera
        },
    ] {
        assert!(!cache.begin_anchor(MotionKey::new(1, &malformed), &malformed));
        assert!(cache.anchor().is_none());
    }
}
