// Priority-ordered active interiors: a small inline list and reusable device spills.
import vork_constants;
import closure;
import vork_material_table;
import vork_spectrum;
import vork_prelude;
__exported import dielectric_interface;
struct PathBoundary {
    uint object;
    uint medium; // Shared material-record handle; participating coefficients reuse it.
    float ior;
#ifndef VORK_UNIT_BOUNDARY_IOR
    float reference_ior;
    float specular_weight;
#endif
    float dispersion_scale;
    float dispersion_abbe;
    float3 color;
    float depth;
};
PathBoundary path_air() {PathBoundary air={};air.object=0xffffffffu;air.medium=0xffffffffu;air.ior=1.;
#ifndef VORK_UNIT_BOUNDARY_IOR
    air.reference_ior=1.;air.specular_weight=1.;
#endif
    air.color=float3(1.);return air;}
PathBoundary path_boundary(uint object, Closure c, uint medium=0xffffffffu) {
    PathBoundary b;b.object=object;b.medium=medium;
#ifdef VORK_UNIT_BOUNDARY_IOR
    b.ior=c.specular_ior;
#else
    b.ior=dielectric_bulk_ior(c.specular_ior,c.specular_weight,1.);
    b.reference_ior=c.specular_ior;b.specular_weight=c.specular_weight;
#endif
    b.dispersion_scale=c.transmission_dispersion_scale;b.dispersion_abbe=c.transmission_dispersion_abbe_number;
    b.color=c.transmission_color;b.depth=c.transmission_depth;return b;
}
float boundary_ior(PathBoundary b) {return spectral_ior(b.ior,vork_path_wavelength,b.dispersion_scale,b.dispersion_abbe);}
float boundary_material_ior(Closure c,PathBoundary incident) {
    return dielectric_material_ior(c,path_segment(incident),vork_path_wavelength);
}
Closure boundary_closure(Closure c,PathBoundary incident,PathBoundary transmitted,bool solid) {
    let optical_incident=boundary_ior(incident);
    let optical_transmitted=solid ? boundary_ior(transmitted) : boundary_material_ior(c,incident);
#ifdef VORK_UNIT_BOUNDARY_IOR
    let reference=transmitted.ior;
#else
    let reference=transmitted.reference_ior;
#endif
    return dielectric_interface(c,optical_incident,solid ? reference : c.specular_ior,optical_transmitted/optical_incident);
}
DielectricSegment path_segment(PathBoundary b) {DielectricSegment s;s.color=b.color;s.depth=b.depth;s.ior=boundary_ior(b);s.reference_ior=b.ior;return s;}
DielectricInterface path_interface(PathBoundary incident,PathBoundary transmitted) {
    DielectricInterface media;media.incident=path_segment(incident);media.transmitted=path_segment(transmitted);return media;
}
float3 boundary_attenuation(PathBoundary b,float distance) {return segment_attenuation(path_segment(b),distance);}
// Priority is a material fact, addressed by the handle already in each medium;
// do not grow every live descriptor for a value used only at crossings.
interface IBoundaryPriority {static int priority(uint medium);}
struct MaterialBoundaryPriority : IBoundaryPriority {
    static int priority(uint medium) {
        if(medium==0xffffffffu) return 0;
        return asint(material_records[medium].extra.x);
    }
}
struct EqualBoundaryPriority : IBoundaryPriority {static int priority(uint medium) {return 0;}}
int boundary_priority(PathBoundary b) {return MaterialBoundaryPriority.priority(b.medium);}
// The inline list is small and independent of scene size. Additional active
// interiors use device storage, allocated only when a ray actually needs them.
// Bitmap allocation permits immediate reuse without a lock or ABA-prone free
// list. A slot belongs to exactly one invocation until that invocation frees it.
static const uint boundary_node_words=REG_PATH_BOUNDARY_NODE_WORDS;
void boundary_memory_fence() {
#ifdef VORK_METAL
    // Slang's DeviceMemoryBarrier lowers to a threadgroup execution barrier on
    // Metal. Allocation is divergent: order device accesses without waiting
    // for other invocations (Metal 3.2+, already required by supported Macs).
    __intrinsic_asm "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)";
#else
    DeviceMemoryBarrier();
#endif
}
uint boundary_heap_groups() {return shader_errors[REG_SHADER_DIAGNOSTICS_WORDS];}
uint boundary_node_at(uint node) {return 2u+boundary_heap_groups()+(node-1u)*boundary_node_words;}
uint boundary_allocate() {
    let groups=boundary_heap_groups();if(groups==0u) return 0u;
    uint first;InterlockedAdd(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+1],1u,first);first%=groups;
    for(uint i=0u;i<groups;i++) {
        let group=(first+i)%groups;uint used;
        InterlockedOr(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+2u+group],0u,used);
        while(used!=0xffffffffu) {
            let bit=firstbitlow(~used);let mask=1u<<bit;uint previous;
            InterlockedCompareExchange(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+2u+group],used,used|mask,previous);
            if(previous==used) {boundary_memory_fence();return group*32u+bit+1u;}
            used=previous;
        }
    }
    return 0u;
}
void boundary_free(uint node) {
    // Finish all accesses before another invocation can claim this slot.
    boundary_memory_fence();
    let slot=node-1u;
    InterlockedAnd(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+2u+slot/32u],~(1u<<(slot%32u)));
}
struct BoundaryEntry {PathBoundary boundary;uint tag;float exit_distance;};
struct PathOccupancy<let capacity:uint> {
    BoundaryEntry active[capacity>0u ? capacity : 1u];
    uint count;uint spill;
    uint node(uint index) {
        var result=spill;
        for(uint i=count-1u;i>index;i--) result=shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+boundary_node_at(result)];
        return result;
    }
    BoundaryEntry entry(uint index) {
        if(capacity<REG_PATH_REGISTER_BOUNDARIES || index<capacity) {
            BoundaryEntry result={};
            [unroll] for(uint i=0u;i<capacity;i++) {if(i==index) result=active[i];}
            return result;
        }
        let at=boundary_node_at(node(index));BoundaryEntry e;
        e.boundary.object=shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+1u];e.boundary.medium=shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+2u];
        e.boundary.ior=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+3u]);
        e.boundary.dispersion_scale=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+4u]);e.boundary.dispersion_abbe=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+5u]);
        e.boundary.color=asfloat(uint3(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+6u],shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+7u],shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+8u]));
        e.boundary.depth=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+9u]);e.tag=shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+10u];e.exit_distance=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+11u]);
#ifndef VORK_UNIT_BOUNDARY_IOR
        e.boundary.reference_ior=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+12u]);e.boundary.specular_weight=asfloat(shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+13u]);
#endif
        return e;
    }
    [mutating] void set_entry(uint index,BoundaryEntry e) {
        if(capacity<REG_PATH_REGISTER_BOUNDARIES || index<capacity) {
            [unroll] for(uint i=0u;i<capacity;i++) {if(i==index) active[i]=e;}
            return;
        }
        let at=boundary_node_at(node(index));
        shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+1u]=e.boundary.object;shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+2u]=e.boundary.medium;shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+3u]=asuint(e.boundary.ior);
        shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+4u]=asuint(e.boundary.dispersion_scale);shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+5u]=asuint(e.boundary.dispersion_abbe);
        shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+6u]=asuint(e.boundary.color.x);shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+7u]=asuint(e.boundary.color.y);shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+8u]=asuint(e.boundary.color.z);
        shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+9u]=asuint(e.boundary.depth);shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+10u]=e.tag;shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+11u]=asuint(e.exit_distance);
#ifndef VORK_UNIT_BOUNDARY_IOR
        shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+12u]=asuint(e.boundary.reference_ior);shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+at+13u]=asuint(e.boundary.specular_weight);
#endif
    }
    PathBoundary at(uint index) {return entry(index).boundary;}
    [mutating] void set(uint index,PathBoundary b) {var e=entry(index);e.boundary=b;set_entry(index,e);}
    uint tag(uint index) {return entry(index).tag;}
    [mutating] void set_tag(uint index,uint value) {var e=entry(index);e.tag=value;set_entry(index,e);resolve_iors();}
    // Existing priority/entry order is authoritative. Cache each bulk d-line
    // index before dispersion; provisional exits cannot surround the next owner.
    [mutating] void resolve_iors() {
#ifndef VORK_UNIT_BOUNDARY_IOR
        var surrounding=path_air();
        for(uint i=0u;i<count;i++) {
            var e=entry(i);if(e.tag==1u || e.tag==3u) continue;
            e.boundary.ior=dielectric_bulk_ior(e.boundary.reference_ior,e.boundary.specular_weight,surrounding.ior);
            if(e.boundary.specular_weight==0.) {
                e.boundary.dispersion_scale=surrounding.dispersion_scale;
                e.boundary.dispersion_abbe=surrounding.dispersion_abbe;
            }
            surrounding=e.boundary;set_entry(i,e);
        }
#endif
    }
    // Reserve before shifting entries: every set() addresses owned storage.
    [mutating] bool append(BoundaryEntry e) {
        if(count>=capacity) {
            if(capacity<REG_PATH_REGISTER_BOUNDARIES) return false;
            let allocated=boundary_allocate();if(allocated==0u) return false;
            shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+boundary_node_at(allocated)]=spill;spill=allocated;
        }
        count++;set_entry(count-1u,e);return true;
    }
    uint find(uint object) {
        for(uint i=0u;i<count;i++) {if(at(i).object==object) return i;}
        return count;
    }
    PathBoundary current() {if(count>0u) return at(count-1u);return path_air();}
    PathBoundary plain_transmitted(PathBoundary b,bool entering,out bool neutral) {
        neutral=true;
        if(entering && find(b.object)==count) {
            neutral=false;
#ifndef VORK_UNIT_BOUNDARY_IOR
            let surrounding=current();b.ior=dielectric_bulk_ior(b.reference_ior,b.specular_weight,surrounding.ior);
            if(b.specular_weight==0.) {b.dispersion_scale=surrounding.dispersion_scale;b.dispersion_abbe=surrounding.dispersion_abbe;}
#endif
            return b;
        }
        if(!entering && count>0u && at(count-1u).object==b.object) {
            neutral=false;if(count>1u) return at(count-2u);return path_air();
        }
        return current();
    }
    float ior() {return boundary_ior(current());}
    float3 attenuation(float distance) {return boundary_attenuation(current(),distance);}
    [mutating] bool cross(PathBoundary b,bool entering) {return cross<MaterialBoundaryPriority>(b,entering);}
    [mutating] bool cross<P:IBoundaryPriority>(PathBoundary b,bool entering) {
        let slot=find(b.object);
        if(entering) {
            if(slot<count) return true;
            BoundaryEntry e={};e.boundary=b;
            if(!append(e)) return false;
            var insert=count-1u;
            while(insert>0u) {
                let previous=entry(insert-1u);
                if(P.priority(previous.boundary.medium)<=P.priority(b.medium)) break;
                set_entry(insert,previous);insert--;
            }
            set_entry(insert,e);
            resolve_iors();
        } else if(slot<count) remove(slot);
        return true;
    }
    [mutating] void remove(uint slot) {
        for(uint i=slot+1u;i<count;i++) set_entry(i-1u,entry(i));
        if(capacity>=REG_PATH_REGISTER_BOUNDARIES && count>capacity) {
            let released=spill;spill=shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+boundary_node_at(released)];boundary_free(released);
        }
        count--;
        resolve_iors();
    }
    [mutating] void clear() {
        while(capacity>=REG_PATH_REGISTER_BOUNDARIES && spill!=0u) {let released=spill;spill=shader_errors[REG_SHADER_DIAGNOSTICS_WORDS+boundary_node_at(released)];boundary_free(released);}
        count=0u;
    }
};
