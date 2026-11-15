//! Arc → cubic conversion now lives in `moda-scene` (`path_geom::arc`) so the
//! boolean path solver can reuse it: moda-engine depends on moda-scene, so the
//! reverse import would be a dependency cycle. This module is a pure re-export
//! — the implementation moved byte-identical.
pub use moda_scene::path_geom::arc::*;
