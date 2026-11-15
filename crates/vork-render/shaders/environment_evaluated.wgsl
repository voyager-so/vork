// Retained source statistics and row-major proposal. No radiance leaves the GPU.
struct Row { sh:array<vec4<f32>,4>, summary:vec4<f32> }
@group(0) @binding(0) var source:texture_2d_array<u32>;
@group(0) @binding(1) var proposal:texture_2d<f32>;
@group(0) @binding(2) var<storage,read_write> rows:array<Row>;
@group(0) @binding(3) var<storage,read_write> distribution:array<vec2<f32>>;
@group(0) @binding(4) var<storage,read_write> statistics:array<vec4<f32>,$STATISTICS_VECTORS>;
const PI:f32=$PI;
@compute @workgroup_size(64) fn peaks(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(source);let y=id.x;if y>=size.y {return;}
    var peak=0.;var invalid=0.;var same=true;
    let reference=textureLoad(source,vec2<i32>(0),1,0).rgb;
    for(var x=0u;x<size.x;x++) {for(var layer=0;layer<2;layer++) {
        let bits=textureLoad(source,vec2<i32>(i32(x),i32(y)),layer,0).rgb;
        let value=bitcast<vec3<f32>>(bits);
        if any((bits & vec3<u32>(0x7f800000u))==vec3<u32>(0x7f800000u)) || any(value<vec3<f32>(0.)) {invalid=1.;}
        if layer==1 {peak=max(peak,max(value.r,max(value.g,value.b)));same=same && all(bits==reference);}
    }}
    rows[y].summary=vec4(peak,0.,invalid,select(0.,1.,same));
}
@compute @workgroup_size(1) fn peak_summary() {
    var peak=0.;var invalid=0.;var same=true;
    for(var y=0u;y<textureDimensions(source).y;y++) {peak=max(peak,rows[y].summary.x);invalid=max(invalid,rows[y].summary.z);same=same && rows[y].summary.w!=0.;}
    statistics[6]=vec4(bitcast<vec3<f32>>(textureLoad(source,vec2<i32>(0),1,0).rgb),select(0.,1.,same));
    let scale=max(1.,peak/$HALF_SCALE);
    statistics[4]=vec4(scale,0.,invalid,1./scale);
}
@compute @workgroup_size(64) fn moments_and_rows(@builtin(global_invocation_id) id:vec3<u32>) {
    let source_size=textureDimensions(source);let size=textureDimensions(proposal);let y=id.x;
    let scale=statistics[4].x;
    if y<source_size.y {
        let theta=PI*(f32(y)+.5)/f32(source_size.y);
        let omega=2.*PI/f32(source_size.x)*2.*sin(theta)*sin(PI*.5/f32(source_size.y));
        var sh:array<vec3<f32>,4>;
        for(var x=0u;x<source_size.x;x++) {
            let phi=2.*PI*((f32(x)+.5)/f32(source_size.x)-.5);
            let direction=vec3(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));
            let rgb=bitcast<vec4<f32>>(textureLoad(source,vec2<i32>(i32(x),i32(y)),1,0)).rgb/scale;
            sh[0]+=rgb*(omega/(4.*PI));
            for(var axis=0u;axis<3u;axis++) {sh[axis+1u]+=rgb*direction[axis]*(omega/(2.*PI));}
        }
        for(var c=0u;c<4u;c++) {rows[y].sh[c]=vec4(sh[c],0.);}
    }
    if y<size.y {
        let omega=2.*PI/f32(size.x)*2.*sin(PI*(f32(y)+.5)/f32(size.y))*sin(PI*.5/f32(size.y));
        var total=0.;
        for(var x=0u;x<size.x;x++) {
            var rgb=vec3<f32>(0.);
            for(var dy=-1;dy<=1;dy++) {for(var dx=-1;dx<=1;dx++) {
                let p=vec2((i32(x)+dx+i32(size.x))%i32(size.x),clamp(i32(y)+dy,0,i32(size.y)-1));
                let value=textureLoad(proposal,p,0);
                let weight=f32((2-abs(dx))*(2-abs(dy)))/16.;
                rgb+=(value.rgb/scale)*(value.a*weight);
            }}
            let luminance=dot(rgb,vec3<f32>($LUMINANCE));total+=luminance*omega;
            distribution[y*size.x+x]=vec2(total,luminance);
        }
        rows[y].summary.y=total;
    }
}
@compute @workgroup_size(1) fn summarize() {
    var total=0.;let size=textureDimensions(proposal);
    for(var y=0u;y<size.y;y++) {total+=rows[y].summary.y;rows[y].summary.y=total;}
    for(var c=0u;c<4u;c++) {
        var value=vec3<f32>(0.);for(var y=0u;y<textureDimensions(source).y;y++) {value+=rows[y].sh[c].xyz;}
        statistics[c]=vec4(value,0.);
    }
    var factor=1.;
    for(var c=0u;c<3u;c++) {
        let amplitude=length(vec3(statistics[1][c],statistics[2][c],statistics[3][c]));
        factor=min(factor,statistics[0][c]/max(amplitude*$DERING,$EPSILON));
    }
    for(var c=1u;c<4u;c++) {statistics[c]=statistics[c]*factor;}
    statistics[5]=vec4(statistics[0].rgb*statistics[4].x,0.);
    if any((bitcast<vec3<u32>>(statistics[5].rgb) & vec3<u32>(0x7f800000u))==vec3<u32>(0x7f800000u)) {statistics[4].z=1.;}
    statistics[4].y=total;
}
@compute @workgroup_size(64) fn normalize_rows(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(proposal);let y=id.x;if y>=size.y {return;}
    let total=statistics[4].y;var prefix=0.;if y>0u {prefix=rows[y-1u].summary.y;}
    for(var x=0u;x<size.x;x++) {
        let index=y*size.x+x;
        if total>0. {distribution[index]=vec2((prefix+distribution[index].x)/total,distribution[index].y/total);}
        else {
            let row=(1.-cos(PI*f32(y)/f32(size.y)))*.5;let next=(1.-cos(PI*f32(y+1u)/f32(size.y)))*.5;
            distribution[index]=vec2(row+(next-row)*f32(x+1u)/f32(size.x),1./(4.*PI));
        }
    }
    if y==size.y-1u {distribution[size.x*size.y-1u].x=1.;}
}
