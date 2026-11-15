import vork;
import vork_sampling;

public enum SensorFit { Auto, Horizontal, Vertical };
public enum CameraProjection { Persp, Ortho };
public struct Inputs {
    CameraProjection projection = CameraProjection.Persp;
    SensorFit fit = SensorFit.Auto;
    float lens = 50.;
    float sensor_width = 36.;
    float sensor_height = 24.;
    float ortho_scale = 6.;
    float2 shift = float2(0.);
    float clip_start = .1;
    float clip_end = 1000.;
    bool use_dof = false;
    float focus_distance = 10.;
    float aperture_fstop = 2.8;
    uint aperture_blades = 0;
    float aperture_rotation = 0.;
    float aperture_ratio = 1.;
};

// Blender view plane. A positive focus argument is the evaluated focus-object depth.
public CameraCalibration camera_calibration(float aspect, float focus, float time, Inputs i) {
    let span = int(i.fit) == int(SensorFit.Horizontal) ? aspect : int(i.fit) == int(SensorFit.Vertical) ? 1. : (aspect > 1. ? aspect : 1.);
    let sensor = int(i.fit) == int(SensorFit.Vertical) ? i.sensor_height : i.sensor_width;
    let height = sensor / span;
    let ortho = int(i.projection) == int(CameraProjection.Ortho);
    CameraCalibration c;
    c.screen = float4(ortho ? 2.*span/(i.ortho_scale*aspect) : 2.*i.lens/(height*aspect),
        ortho ? 2.*span/i.ortho_scale : 2.*i.lens/height,
        -2.*i.shift.x*span/aspect, -2.*i.shift.y*span);
    c.clip = float4(i.clip_start, i.clip_end, ortho ? 1. : 0., 1.);
    c.aperture = float4(i.use_dof && !ortho ? i.lens*.001/(2.*i.aperture_fstop) : 0.,
        focus >= 0. ? focus : i.focus_distance, float(i.aperture_blades), i.aperture_rotation);
    c.shape = float4(i.aperture_ratio, 0., 0., 0.);
    let valid = i.lens>0. && i.sensor_width>0. && i.sensor_height>0. && i.ortho_scale>0.
        && (!i.use_dof || (!ortho && c.aperture.y>i.lens*.001 && i.aperture_fstop>0.
        && (i.aperture_blades==0 || (i.aperture_blades>=3 && i.aperture_blades<=16))
        && i.aperture_ratio>=.01 && i.aperture_ratio<=100.));
    if (!valid) c.clip.w=-1.;
    return c;
}

public CameraRay camera(CameraContext ctx, Inputs i) {
    let c = camera_calibration(ctx.film_size.x/ctx.film_size.y, ctx.focus_distance, ctx.time, i);
    let film = (ctx.uv*float2(2.,-2.)+float2(-1.,1.)-c.screen.zw)/c.screen.xy;
    let lens = aperture_point(polar_disk(ctx.lens_sample),float4(c.aperture.zw,c.shape.x,0.))*c.aperture.x;
    CameraRay r;
    r.origin = ctx.origin;
    if (c.clip.z != 0.) {
        r.origin += film.x*ctx.right + film.y*ctx.up;
        r.direction = ctx.forward;
    } else {
        r.origin += lens.x*ctx.right + lens.y*ctx.up;
        let aim = film-lens/c.aperture.y;
        r.direction = normalize(ctx.forward+aim.x*ctx.right+aim.y*ctx.up);
    }
    let cosine = dot(r.direction, ctx.forward);
    r.t_min = i.clip_start / cosine;
    r.t_max = i.clip_end / cosine;
    return r;
}
