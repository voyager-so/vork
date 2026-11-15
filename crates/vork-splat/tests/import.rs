use std::io::{Cursor, Write};
use vork_splat::{Filter, inspect_ply, inspect_spz, read_ply, read_spz};

fn properties(degree: u8) -> Vec<(String, f32)> {
    let mut p = [("x",1.25),("y",-2.5),("z",3.75),
        ("scale_0",0.),("scale_1",1.),("scale_2",-1.),("opacity",0.),
        ("rot_0",2.),("rot_1",0.),("rot_2",0.),("rot_3",0.),
        ("f_dc_0",0.2),("f_dc_1",0.3),("f_dc_2",0.4)]
        .into_iter().map(|(k,v)|(k.to_owned(),v)).collect::<Vec<_>>();
    let count = ((usize::from(degree)+1).pow(2)-1)*3;
    p.extend((0..count).map(|i|(format!("f_rest_{i}"),i as f32/100.)));
    p
}

fn ply(p: &[(String,f32)], encoding: &str, count: u32) -> Vec<u8> {
    let mut header = format!("ply\nformat {encoding} 1.0\ncomment Vertical axis: 1 -2 3\ncomment SplatRenderMode: mip\nelement vertex {count}\n");
    for (key,_) in p {header.push_str(&format!("property float {key}\n"));}
    header.push_str("end_header\n");
    let mut bytes=header.into_bytes();
    if encoding == "ascii" {
        for (_,v) in p {bytes.extend_from_slice(format!("{v} ").as_bytes());}
        bytes.push(b'\n');
    } else {
        for (_,v) in p {bytes.extend_from_slice(&if encoding == "binary_little_endian" {v.to_le_bytes()} else {v.to_be_bytes()});}
    }
    bytes
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut writer=flate2::write::GzEncoder::new(Vec::new(),flate2::Compression::default());
    writer.write_all(bytes).unwrap();writer.finish().unwrap()
}

fn spz(version: u32, degree: u8) -> Vec<u8> {
    let mut b = b"NGSP".to_vec();
    b.extend(version.to_le_bytes());b.extend(1u32.to_le_bytes());
    b.extend([degree,12,1,0]);
    for v in [5120i32,-10240,15360] {b.extend_from_slice(&v.to_le_bytes()[..3]);}
    b.push(128); // alpha, independent of RGB
    b.extend([100,150,200]); // DC
    b.extend([160,176,144]); // exp(0), exp(1), exp(-1)
    if version==2 {b.extend([128,128,128]);} else {b.extend((3u32<<30).to_le_bytes());}
    for i in 0..((usize::from(degree)+1).pow(2)-1) {b.extend([128+i as u8,128,128-i as u8]);}
    b
}

#[test]
fn ply_encodings_and_sh_degrees_preserve_attributes_in_arbitrary_order() {
    for degree in 0..=3 {
        let mut p=properties(degree);p.reverse();
        for encoding in ["ascii","binary_little_endian","binary_big_endian"] {
            let b=ply(&p,encoding,1);
            let header=inspect_ply(Cursor::new(&b)).unwrap();
            assert_eq!(header.attribute_bytes(), (11+(u64::from(degree)+1).pow(2)*3)*4);
            let c=read_ply(Cursor::new(b),header.attribute_bytes()).unwrap();
            assert_eq!(c.positions,[[1.25,-2.5,3.75]]);
            assert_eq!(c.rotations,[[0.,0.,0.,1.]]);
            assert_eq!(c.opacities,[0.5]);
            assert_eq!(c.scales,[[1.,1.0f32.exp(),(-1.0f32).exp()]]);
            assert_eq!(c.metadata.filter,Filter::Mip);
            assert_eq!(c.metadata.vertical_axis,Some([1.,-2.,3.]));
            assert_eq!(c.harmonics[0],[[0.2,0.3,0.4]]);
            let rest=c.harmonics.len()-1;
            for i in 0..rest {assert_eq!(c.harmonics[i+1][0], [i as f32/100.,(i+rest) as f32/100.,(i+2*rest) as f32/100.]);}
        }
    }
}

#[test]
fn spz_quantization_is_decoded_without_discarding_bands() {
    for version in [2,3] {
        for degree in 0..=3 {
            let b=gzip(&spz(version,degree));
            let header=inspect_spz(Cursor::new(&b)).unwrap();
            let c=read_spz(Cursor::new(b),header.attribute_bytes()).unwrap();
            assert_eq!(c.positions,[[1.25,-2.5,3.75]]);
            assert_eq!(c.metadata.degree,degree);
            assert_eq!(c.metadata.filter,Filter::Mip);
            assert_eq!(c.opacities,[128./255.]);
            assert_eq!(c.scales,[[1.,1.0f32.exp(),(-1.0f32).exp()]]);
            for i in 0..c.harmonics.len()-1 {assert_eq!(c.harmonics[i+1][0], [i as f32/128.,0.,-(i as f32)/128.]);}
            assert!((c.rotations[0].iter().map(|v|v*v).sum::<f32>()-1.).abs()<1e-6);
        }
    }
}

#[test]
fn smallest_three_rotation_decodes_index_order_and_sign() {
    let mut b=spz(3,0);
    // largest = y; x=+1/sqrt(2), z=-1/(sqrt(2)*511), w=0.
    let packed=(1u32<<30)|(511<<20)|(513<<10);
    b[32..36].copy_from_slice(&packed.to_le_bytes());
    let c=read_spz(Cursor::new(gzip(&b)),4096).unwrap();
    let q=c.rotations[0];
    assert!((q[0]-std::f32::consts::FRAC_1_SQRT_2).abs()<1e-6);
    assert!(q[1]>0.7 && q[2]<0. && q[3]==0.);
}

#[test]
fn memory_admission_happens_before_reading_a_payload() {
    let b=ply(&properties(3),"binary_little_endian",2_000_000);
    let header=inspect_ply(Cursor::new(&b)).unwrap();
    assert_eq!(header.attribute_bytes(),472_000_000);
    let e=read_ply(Cursor::new(b),1000).unwrap_err().to_string();
    assert!(e.contains("472000000") && e.contains("1000") && e.contains("memory"),"{e}");
    let mut b=spz(3,3);b[8..12].copy_from_slice(&4_000_000u32.to_le_bytes());b.truncate(16);
    let e=read_spz(Cursor::new(gzip(&b)),1000).unwrap_err().to_string();
    assert!(e.contains("944000000"),"{e}");
}

#[test]
fn malformed_and_unsupported_input_refuses() {
    for encoding in ["ascii","binary_little_endian","binary_big_endian"] {
        for value in [f32::NAN,f32::INFINITY] {
            let mut p=properties(0);p[0].1=value;
            assert!(read_ply(Cursor::new(ply(&p,encoding,1)),4096).is_err());
        }
        let mut p=properties(1);p.pop();
        assert!(read_ply(Cursor::new(ply(&p,encoding,1)),4096).is_err());
        let mut p=properties(0);p.push(p[0].clone());
        assert!(read_ply(Cursor::new(ply(&p,encoding,1)),4096).is_err());
        let mut p=properties(0);p[7].1=0.;
        assert!(read_ply(Cursor::new(ply(&p,encoding,1)),4096).is_err());
    }
    let b=ply(&properties(3),"binary_little_endian",1);
    for n in 0..b.len() {assert!(read_ply(Cursor::new(&b[..n]),4096).is_err(),"truncation {n}");}
    let packed=gzip(&spz(3,3));
    for n in 0..packed.len() {assert!(read_spz(Cursor::new(&packed[..n]),4096).is_err(),"gzip truncation {n}");}
    for (index,value) in [(4,4),(12,4),(13,32),(14,2),(15,1)] {
        let mut b=spz(3,3);b[index]=value;
        assert!(read_spz(Cursor::new(gzip(&b)),4096).is_err());
    }
    let mut b=spz(3,3);b.push(0);
    assert!(read_spz(Cursor::new(gzip(&b)),4096).is_err());
    let mut b=packed.clone();b.push(0);
    assert!(read_spz(Cursor::new(b),4096).is_err());
    let mut b=packed;b[20]^=64;
    assert!(read_spz(Cursor::new(b),4096).is_err());
}

#[test]
fn uncompressed_spz_versions_name_the_supported_export() {
    let bytes=spz(4,0);
    for error in [inspect_spz(Cursor::new(&bytes)).unwrap_err(),read_spz(Cursor::new(&bytes),1024).unwrap_err()] {
        let message=error.to_string();
        assert!(message.contains("unsupported SPZ version 4") && message.contains("version 2 or 3") && !message.contains('\n'));
    }
}
