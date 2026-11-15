// The decoupled slots buffer layout (passes/decoupled.rs slot_words), shared by
// the merge and the shade: dense slots, overflow list heads, overflow pool.
import vork_constants;
// A visibility key is object + 1 (zero marks a pixel no triangle covers) and the
// draw's primitive: every draw has its own objects (install), so the object names the draw.
public uint2 visibility_key(uint object, uint primitive) {return uint2(object+1u,primitive);}
// Visibility sample `sample`'s hashed-alpha index and count: the exposure's
// strata (`temporal`, frame.temporal.xy) times its `samples` (frame.sampling.w).
public uint2 visibility_coverage(uint sample, uint samples, float2 temporal) {
    return uint2(uint(temporal.x)*samples+sample,uint(temporal.y)*samples);
}

public struct SlotLayout {
    public uint pixels;
    public uint first_pixel;
    public uint masks;
    public uint stride;
    public __init(uint pixels, uint samples, uint packet_pixels=0u, uint first_pixel=0u) {
        this.first_pixel=first_pixel;
        this.pixels=packet_pixels==0u ? pixels : packet_pixels;
        masks=(samples+31u)/32u;
        stride=REG_VISIBILITY_KEY_WORDS+masks;
    }
    // A slot record index shared by reflection and diffusion: one
    // per dense slot, then one per overflow pool entry.
    public uint record(uint at,bool direct=false) {
        // Direct visibility visits only dense slot zero; its retained records
        // need no space for the other dense slots or an overflow pool.
        if(direct) {return at/(REG_VISIBILITY_SLOTS*stride);}
        let dense=pixels*REG_VISIBILITY_SLOTS;
        let index=select(at<dense*stride,at/stride,dense+(at-entry(0u)-1u)/(1u+stride));
        return index;
    }
    public uint address(uint index) {
        let dense=pixels*REG_VISIBILITY_SLOTS;
        return index<dense ? index*stride : entry(index-dense)+1u;
    }
    public uint diffusion(uint at) {return record(at)*REG_VISIBILITY_DIFFUSION_WORDS;}
    public uint reflection(uint at,bool direct) {return record(at,direct);}
    // Every consumer follows the same dense/pool list and empty-slot rule.
    public uint first(StructuredBuffer<uint> slots, uint pixel) {
        let at=dense(pixel,0u);
        if (slots[at]!=0u) {return at;}
        let e=slots[head(pixel)];return select(e==0u,0xffffffffu,entry(e-1u)+1u);
    }
    public uint next(StructuredBuffer<uint> slots, uint pixel, uint at) {
        let end=dense(pixel,REG_VISIBILITY_SLOTS);
        if (at<end-stride) {let following=at+stride;if (slots[following]!=0u) {return following;}}
        var e=0u;
        if (at<end) {e=slots[head(pixel)];} else {e=slots[at-1u];}
        return select(e==0u,0xffffffffu,entry(e-1u)+1u);
    }
    // Dense slot `slot` of `pixel`: its key, then its masks.
    public uint local(uint pixel) {return pixel-first_pixel;}
    public uint dense(uint pixel, uint slot) {return ((local(pixel))*REG_VISIBILITY_SLOTS+slot)*stride;}
    // The word holding `pixel`'s first overflow entry + 1 (0: none).
    public uint head(uint pixel) {return pixels*REG_VISIBILITY_SLOTS*stride+local(pixel);}
    // Pool entry `index`: the next entry + 1, then a key and its masks.
    public uint entry(uint index) {return pixels*(REG_VISIBILITY_SLOTS*stride+1u)+index*(1u+stride);}
}
// The transparent fragment lists (decoupled_visibility.slang transparent_fragments_fs):
// a header (allocations, the frame's demand once full, capacity, width, height),
// then `capacity` records (the next record + 1, the key, and the batch's samples
// it covers in their mask word), then per pass of a merge batch, the record + 1
// starting each pixel's list.
public struct FragmentLayout {
    public uint capacity;
    public uint width;
    public uint height;
    public __init(uint capacity, uint width, uint height) {this.capacity=capacity;this.width=width;this.height=height;}
    public uint depth(uint sample,uint2 pixel,uint samples) {return head(min(samples,REG_MERGE_BATCH)+sample%REG_MERGE_BATCH,pixel);}
    public uint record(uint index) {return REG_FRAGMENT_HEADER_WORDS+index*REG_FRAGMENT_RECORD_WORDS;}
    public uint head(uint layer, uint2 pixel) {
        return REG_FRAGMENT_HEADER_WORDS+capacity*REG_FRAGMENT_RECORD_WORDS+(layer*height+pixel.y)*width+pixel.x;
    }
}
