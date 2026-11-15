//! Scene-sized named light-link bitsets in the shared immutable material word table.
use super::*;
use ::vork_scene::v2;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
pub(crate) struct Selection {
    /// Word address of a length-prefixed bitset; zero is the empty set.
    pub mask: u32,
    pub all: bool,
}
const ALL: Selection = Selection { mask: 0, all: true };
pub(crate) struct Links {
    names: BTreeMap<String, usize>,
    pub groups: Vec<u32>,
    illuminates: Vec<Selection>,
    shadowed_by: Vec<Selection>,
    words: Vec<u32>,
    sets: BTreeMap<Vec<u32>, u32>,
    base: u32,
}
impl Links {
    fn intern(&mut self, bits: Vec<u32>) -> Result<u32> {
        if bits.iter().all(|&word| word == 0) {return Ok(0);}
        if let Some(&address) = self.sets.get(&bits) {return Ok(address);}
        let address = u32::try_from(self.words.len())?;
        self.words.push(u32::try_from(bits.len())?);
        self.words.extend_from_slice(&bits);
        self.sets.insert(bits, address);
        Ok(address)
    }
    fn bits(&self, address: u32) -> &[u32] {
        if address == 0 {return &[];}
        let at = (address - self.base) as usize;
        &self.words[at+1..at+1+self.words[at] as usize]
    }
    fn union(&mut self, parent: u32, names: &[String]) -> Result<u32> {
        let mut bits = vec![0; self.names.len().div_ceil(32)];
        let inherited = self.bits(parent);
        bits[..inherited.len()].copy_from_slice(inherited);
        for name in names {
            let index = self.names[name];
            bits[index/32] |= 1 << (index%32);
        }
        self.intern(bits)
    }
    fn selection(&mut self, names: Option<&[String]>) -> Result<Selection> {
        Ok(Selection {mask: self.union(0, names.unwrap_or_default())?, all: names.is_none()})
    }
    #[cfg(test)]
    fn intersects(&self, a: u32, b: u32) -> bool {
        self.bits(a).iter().zip(self.bits(b)).any(|(a,b)| a&b != 0)
    }
    pub fn illuminates(&self, node: usize) -> Selection {self.illuminates.get(node).copied().unwrap_or(ALL)}
    pub fn shadowed_by(&self, node: usize) -> Selection {self.shadowed_by.get(node).copied().unwrap_or(ALL)}
    pub fn new(nodes: &[EvalNode]) -> Result<Self> {
        let mut names = BTreeSet::new();
        for node in nodes.iter().map(|n| &n.node) {
            names.extend(node.light_link_tags.iter().cloned());
            if let v2::NodeKind::Asset { nodes, .. } = &node.kind {
                for node in nodes.values() {names.extend(node.light_link_tags.iter().flatten().cloned());}
            }
        }
        let names = names.into_iter().enumerate().map(|(i,name)|(name,i)).collect();
        let mut result = Self {names, groups:vec![0;nodes.len()], illuminates:vec![ALL;nodes.len()], shadowed_by:vec![ALL;nodes.len()], words:vec![0], sets:BTreeMap::new(), base:0};
        for (i,node) in nodes.iter().enumerate() {
            let parent = node.parent.map_or(0,|p|result.groups[p]);
            result.groups[i] = result.union(parent, &node.node.light_link_tags)?;
            if let v2::NodeKind::Light {illuminates,shadowed_by,..} = &node.node.kind {
                result.illuminates[i] = result.selection(illuminates.as_deref())?;
                result.shadowed_by[i] = result.selection(shadowed_by.as_deref())?;
            }
        }
        Ok(result)
    }
    pub fn object_groups(&mut self, root: usize, source: &v2::Node, path: &[(usize,String)]) -> Result<u32> {
        let mut groups = self.groups[root];
        if let v2::NodeKind::Asset {nodes,..} = &source.kind {
            for (index,name) in path {
                if let Some(own) = buffers::imported_override(nodes,*index,name).and_then(|n|n.light_link_tags.as_ref()) {
                    groups = self.union(groups,own)?;
                }
            }
        }
        Ok(groups)
    }
    /// Relocate every handle once, after all imported memberships are interned.
    pub fn finish(&mut self, words: &mut Vec<u32>, record_words: u32, objects: &mut [Object]) -> Result<()> {
        self.base = record_words.checked_add(u32::try_from(words.len())?).context("light-link word address exceeds u32")?;
        self.base.checked_add(u32::try_from(self.words.len())?).context("light-link word table exceeds u32")?;
        let relocate = |address: &mut u32| {if *address != 0 {*address += self.base;}};
        for group in &mut self.groups {relocate(group);}
        for selection in self.illuminates.iter_mut().chain(&mut self.shadowed_by) {relocate(&mut selection.mask);}
        for object in objects {relocate(&mut object.metadata[0]);}
        words.extend_from_slice(&self.words);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_link_boundaries_and_relocation() -> Result<()> {
        for count in [0usize,1,31,32,33,63,64,65,127,128,129,65537,1048577] {
            let tags=(0..count).map(|i|format!("g{i:08}")).collect::<Vec<_>>();
            let selected=tags.last().cloned().into_iter().collect::<Vec<_>>();
            let nodes:Vec<v2::Node>=vork_scene::v2::from_test_value(serde_json::json!([
                {"type":"node","light_link_tags":tags},
                {"type":"node"},
                {"type":"light","shape":"point","illuminates":selected,"shadowed_by":[]},
                {"type":"node"}
            ]))?;
            let nodes:Vec<_>=nodes.into_iter().enumerate().map(|(i,node)|EvalNode {
                camera:None,node,parent:(i==1).then_some(0),world:Mat4::IDENTITY,dynamic:false,embedded:None
            }).collect();
            let mut links=Links::new(&nodes)?;
            assert_eq!(links.groups[0],links.groups[1],"inherited set is interned");
            assert!(links.illuminates(0).all);
            assert!(!links.illuminates(2).all);
            assert!(!links.shadowed_by(2).all);
            assert_eq!(links.shadowed_by(2).mask,0);
            let mut objects=[Object::new(Mat4::IDENTITY)];
            objects[0].metadata[0]=links.groups[1];
            let mut words=vec![0xdeadbeef;17];
            links.finish(&mut words,64,&mut objects)?;
            assert_eq!(objects[0].metadata[0],links.groups[1]);
            let read=|address:u32| -> &[u32] {
                if address==0 {return &[];}
                let at=address as usize-64;
                &words[at+1..at+1+words[at] as usize]
            };
            let bits=read(links.groups[1]);
            assert_eq!(bits.len(),count.div_ceil(32));
            for i in 0..count {assert_ne!(bits[i/32]&(1<<(i%32)),0);}
            let selection=read(links.illuminates(2).mask);
            assert_eq!(selection.iter().map(|w|w.count_ones()).sum::<u32>(),u32::from(count>0));
            if count>0 {assert_eq!(selection[(count-1)/32],1<<((count-1)%32));}
            assert_eq!(links.intersects(links.groups[1],links.illuminates(2).mask),count>0);
            assert!(!links.intersects(links.groups[3],links.illuminates(2).mask));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn link_word_boundaries_on_gpu() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut words = vec![0u32];
        let mut cases = Vec::new();
        for count in [0usize, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 65537, 1048577] {
            let width = count.div_ceil(32);
            let a = words.len() as u32;
            words.push(width as u32);
            words.extend(std::iter::repeat_n(u32::MAX, width));
            let b = words.len() as u32;
            words.push(width as u32);
            words.extend(std::iter::repeat_n(0, width));
            if count != 0 { words[b as usize + 1 + (count-1)/32] = 1 << ((count-1)%32); }
            cases.extend([[a,b,u32::from(count!=0),0], [a,0,0,0], [0,b,0,0]]);
            let empty = words.len() as u32;
            words.push(width as u32);
            words.extend(std::iter::repeat_n(0, width));
            cases.push([a,empty,0,0]);
        }
        cases.resize(cases.len().next_multiple_of(64), [0,0,0,0]);
        let function = include_str!("../../shaders/vork_material_table.slang")
            .split("export bool links_intersection_words").nth(1).unwrap().split("#ifdef").next().unwrap();
        let source = format!(r#"
[[vk::binding(0,0)]] StructuredBuffer<uint> material_words;
[[vk::binding(1,0)]] StructuredBuffer<uint4> cases;
[[vk::binding(2,0)]] RWStructuredBuffer<uint> results;
export bool links_intersection_words{function}
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    results[id.x] = uint(links_intersection_words(cases[id.x].x,cases[id.x].y));
}}
"#);
        let entries = [true,true,false].into_iter().enumerate().map(|(i,read_only)|
            crate::pipelines::buffer_entry(i as u32,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only})).collect::<Vec<_>>();
        let pipeline = crate::medium::test_pipeline(&r,&source,&Default::default(),&entries,(64,1,1))?;
        let table = init(&r.device,"link words",bytemuck::cast_slice(&words),wgpu::BufferUsages::STORAGE);
        let inputs = init(&r.device,"link cases",bytemuck::cast_slice(&cases),wgpu::BufferUsages::STORAGE);
        let size = cases.len() as u64 * 4;
        let output = r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let readback = r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,table.as_entire_binding()),entry(1,inputs.as_entire_binding()),entry(2,output.as_entire_binding())]});
        let mut encoder = r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(cases.len() as u32/64,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&readback,0,size);
        crate::command_finish::Segments::submit(&r.queue,encoder);
        let (tx,rx)=std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move |v| {let _=tx.send(v);});
        r.wait(None)?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);
        let actual:&[u32]=bytemuck::cast_slice(&mapped);
        for (i,(result,case)) in actual.iter().zip(&cases).enumerate() {assert_eq!(*result,case[2],"link case {i}");}
        drop(mapped);readback.unmap();
        r.errors.check()?;
        Ok(())
    }

    #[test]
    fn arbitrary_width_memberships_and_wildcards() -> Result<()> {
        let nodes:Vec<v2::Node> = vork_scene::v2::from_test_value(serde_json::json!([
            {"type":"node", "light_link_tags":(0..130).map(|i|format!("g{i}")).collect::<Vec<_>>()},
            {"type":"light","shape":"point","illuminates":["g129"]},
            {"type":"node"},
            {"type":"node","light_link_tags":["g1"]}
        ]))?;
        let nodes:Vec<_> = nodes.into_iter().enumerate().map(|(i,node)|EvalNode { camera: None,node,parent:(i==2).then_some(0),world:Mat4::IDENTITY,dynamic:false,embedded:None}).collect();
        let mut links = Links::new(&nodes)?;
        assert!(links.intersects(links.groups[2],links.illuminates(1).mask));
        assert!(!links.intersects(links.groups[3],links.illuminates(1).mask));
        assert!(links.illuminates(0).all);
        assert!(!links.illuminates(1).all);
        let mut words=vec![0;17];
        links.finish(&mut words,64,&mut [])?;
        assert!(links.intersects(links.groups[2],links.illuminates(1).mask));
        assert!(!links.intersects(links.groups[3],links.illuminates(1).mask));
        Ok(())
    }
}
