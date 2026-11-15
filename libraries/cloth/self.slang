void cloth_self_refit(uint lane) {
 for(uint f=lane;f<SELF_LEAVES;f+=64) {
  float3 lo=float3(1e30),hi=float3(-1e30);
  if(f<CLOTH_FACES) {
   uint face=self_order_load(f);uint3 ids=uint3(triangles_load(3*face),triangles_load(3*face+1),triangles_load(3*face+2));
   float3 a=snapshot_load(ids.x),b=snapshot_load(ids.y),c=snapshot_load(ids.z);lo=min(min(a,min(b,c)),min(old_load(ids.x),min(old_load(ids.y),old_load(ids.z))));hi=max(max(a,max(b,c)),max(old_load(ids.x),max(old_load(ids.y),old_load(ids.z))));
  }
  self_lo_store(SELF_LEAVES+f,lo);self_hi_store(SELF_LEAVES+f,hi);
 }
 DeviceMemoryBarrierWithGroupSync();
 for(uint first=SELF_LEAVES/2;first>0;first/=2) {
  for(uint node=first+lane;node<2*first;node+=64) {
   self_lo_store(node,min(self_lo_load(2*node),self_lo_load(2*node+1)));
   self_hi_store(node,max(self_hi_load(2*node),self_hi_load(2*node+1)));
  }
  DeviceMemoryBarrierWithGroupSync();
 }
}
float3 cloth_self_normal(float3 delta,float3 old_delta,float3 face_normal,bool interior,out float distance) {
 distance=length(delta);float nl=length(face_normal);
 float3 n=distance>1e-6*(SELF_MARGIN/3)?delta/distance:cloth_unit(old_delta);
 // A plane crossing is a face contact only in the face interior. Beside
 // an edge, crossing the infinite supporting plane is not penetration.
 if(interior&&nl>0) {
  float3 fn=face_normal/nl;
  if(dot(old_delta,fn)*dot(delta,fn)<0){n=dot(old_delta,fn)>0?fn:-fn;distance=dot(delta,n);}
 }
 return n;
}
void cloth_segments(float3 p,float3 q,float3 a,float3 b,out float s,out float t) {
 float3 u=q-p,v=b-a,r=p-a;float x=dot(u,u),y=dot(v,v),uv=dot(u,v),ur=dot(u,r),vr=dot(v,r);
 float edge_epsilon2=1e-12*(SELF_MARGIN/3)*(SELF_MARGIN/3);
 if(x<=edge_epsilon2){s=0;t=y>edge_epsilon2?saturate(vr/y):0;return;}
 if(y<=edge_epsilon2){t=0;s=saturate(-ur/x);return;}
 float den=x*y-uv*uv;s=den>1e-6*x*y?saturate((uv*vr-ur*y)/den):0;t=(uv*s+vr)/y;
 if(t<0){t=0;s=saturate(-ur/x);}else if(t>1){t=1;s=saturate((uv-ur)/x);}
}
void cloth_register(uint i,uint kind,uint4 key,uint4 ids,float4 weights,float4 plane,float3 motion) {
 float own=(i==ids.x?weights.x:0)+(i==ids.y?weights.y:0)+(i==ids.z?weights.z:0)+(i==ids.w?weights.w:0);
 if(abs(own)<1e-6)return;
 uint count=uint(contact_count_load(i));
 for(uint k=0;k<min(count,CONTACT_CAPACITY);k++) {
  uint slot=i*CONTACT_CAPACITY+k;
  if(uint(contact_kind_load(slot))==kind&&all(uint4(contact_key_load(slot))==key))return;
 }
 if(count>=CONTACT_CAPACITY){contact_count_store(i,float(CONTACT_CAPACITY+1));contact_status_atomic_max(0,CONTACT_CAPACITY+1);return;}
 uint slot=i*CONTACT_CAPACITY+count;
 contact_key_store(slot,float4(key));contact_ids_store(slot,float4(ids));contact_weights_store(slot,weights);contact_plane_store(slot,plane);contact_motion_store(slot,motion);contact_tangent_lambda_store(slot,float3(0));contact_kind_store(slot,float(kind));contact_lambda_store(slot,0);
 uint owner=0xffffffffu;for(uint j=0;j<4;j++)if(abs(weights[j])>1e-6&&mass_load(ids[j])>0)owner=min(owner,ids[j]);contact_owner_store(slot,float(owner));
 contact_count_store(i,float(count+1));
}
void cloth_contact_add(uint i,uint kind,uint4 ids,float4 weights,float3 normal,float margin) {
 cloth_register(i,kind,ids,ids,weights,float4(normal,margin),float3(0));
}
// Rest-close primitives are part of the fabric, not collision partners.
// Test closest rest points (not current barycentrics) so filtering is invariant
// under deformation and agrees for every vertex that gathers the contact.
bool cloth_rest_vertex_face(uint vertex,uint3 face,float margin) {
 float3 bary;
 float3 p=rest_position_load(vertex);
 return length(p-cloth_closest(p,rest_position_load(face.x),rest_position_load(face.y),rest_position_load(face.z),bary))<margin;
}
bool cloth_rest_edges(uint4 pair,float margin) {
 float s,t;
 float3 a=rest_position_load(pair.x),b=rest_position_load(pair.y),c=rest_position_load(pair.z),d=rest_position_load(pair.w);
 cloth_segments(a,b,c,d,s,t);
 return length(lerp(a,b,s)-lerp(c,d,t))<margin;
}
bool cloth_self_boxes(float3 lo,float3 hi,float3 other_lo,float3 other_hi,float margin) {return all(hi+margin>=other_lo)&&all(other_hi+margin>=lo);}
void cloth_self_generate(uint i) {
 if(any((asuint(p_load(i))&uint3(0x7f800000u))==uint3(0x7f800000u)))return;
 contact_count_store(i,0);if(mass_load(i)==0)return;cloth_body_generate(i);
 float3 p=snapshot_load(i),lo=min(p,old_load(i)),hi=max(p,old_load(i));
 for(uint j=neighbour_offsets_load(i);j<neighbour_offsets_load(i+1);j++) {
  uint k=neighbours_load(j);lo=min(lo,min(snapshot_load(k),old_load(k)));hi=max(hi,max(snapshot_load(k),old_load(k)));
 }
 float margin=2*SELF_MARGIN;lo-=margin;hi+=margin;
 uint stack[32];uint top=1;stack[0]=1;
 while(top>0) {
  uint node=stack[--top];if(any(self_hi_load(node)<lo)||any(self_lo_load(node)>hi))continue;
  if(node<SELF_LEAVES){stack[top++]=2*node;stack[top++]=2*node+1;continue;}
  uint leaf=node-SELF_LEAVES;if(leaf>=CLOTH_FACES)continue;
  uint face=self_order_load(leaf);uint3 ids=uint3(triangles_load(3*face),triangles_load(3*face+1),triangles_load(3*face+2));
  float3 a=snapshot_load(ids.x),b=snapshot_load(ids.y),c=snapshot_load(ids.z),bary;
  float3 glo=min(a,min(b,c)),ghi=max(a,max(b,c));
  if(!any(ids==uint3(i))&&cloth_self_boxes(p,p,glo,ghi,2*margin)) {
   float3 near=cloth_closest(p,a,b,c,bary),delta=p-near;float d;
   float3 old_delta=old_load(i)-bary.x*old_load(ids.x)-bary.y*old_load(ids.y)-bary.z*old_load(ids.z);
   float3 normal=cloth_self_normal(delta,old_delta,cross(b-a,c-a),all(bary>float3(0)),d);
   if(d<margin&&length(delta)<2*margin&&!cloth_rest_vertex_face(i,ids,margin))cloth_contact_add(i,0,uint4(i,ids),float4(1,-bary),normal,margin);
  }
  for(uint j=incident_offsets_load(i);j<incident_offsets_load(i+1);j++) {
   uint f=incidents_load(j);uint3 own=uint3(triangles_load(3*f),triangles_load(3*f+1),triangles_load(3*f+2));
   float3 x=snapshot_load(own.x),y=snapshot_load(own.y),z=snapshot_load(own.z),flo=min(x,min(y,z)),fhi=max(x,max(y,z));
   if(!cloth_self_boxes(flo,fhi,glo,ghi,2*margin))continue;
   for(uint k=0;k<3;k++) {
    uint v=ids[k];if(any(own==uint3(v)))continue;float3 q=snapshot_load(v),weights;if(!cloth_self_boxes(q,q,flo,fhi,2*margin))continue;
    float3 r=cloth_closest(q,x,y,z,weights),diff=q-r;
    float3 old_diff=old_load(v)-weights.x*old_load(own.x)-weights.y*old_load(own.y)-weights.z*old_load(own.z);
    float distance;float3 normal=cloth_self_normal(diff,old_diff,cross(y-x,z-x),all(weights>float3(0)),distance);
    if(distance<margin&&length(diff)<2*margin&&!cloth_rest_vertex_face(v,own,margin))cloth_contact_add(i,0,uint4(v,own),float4(1,-weights),normal,margin);
   }
  }
  for(uint j=neighbour_offsets_load(i);j<neighbour_offsets_load(i+1);j++) {
   uint other=neighbours_load(j);float3 q=snapshot_load(other);
   for(uint k=0;k<3;k++) {
    uint u=ids[k],v=ids[(k+1)%3];if(i==u||i==v||other==u||other==v)continue;
    if(!cloth_self_boxes(min(p,q),max(p,q),min(snapshot_load(u),snapshot_load(v)),max(snapshot_load(u),snapshot_load(v)),margin))continue;
    uint4 pair=uint4(min(i,other),max(i,other),min(u,v),max(u,v));
    if(pair.x>pair.z||(pair.x==pair.z&&pair.y>pair.w))pair=pair.zwxy;
    float s,t;cloth_segments(snapshot_load(pair.x),snapshot_load(pair.y),snapshot_load(pair.z),snapshot_load(pair.w),s,t);
    float3 diff=lerp(snapshot_load(pair.x),snapshot_load(pair.y),s)-lerp(snapshot_load(pair.z),snapshot_load(pair.w),t);float distance=length(diff);
    if(distance>=margin||cloth_rest_edges(pair,margin))continue;
    float3 old_diff=lerp(old_load(pair.x),old_load(pair.y),s)-lerp(old_load(pair.z),old_load(pair.w),t);
    float3 n=distance>1e-6*(SELF_MARGIN/3)?diff/distance:cloth_unit(old_diff);
    if(dot(diff,old_diff)<0){n=cloth_unit(old_diff);distance=dot(diff,n);}
    cloth_contact_add(i,1,pair,float4(1-s,s,-(1-t),-t),n,margin);
   }
  }
 }
}
void cloth_contacts_gauss_seidel() {
 for(uint i=0;i<CLOTH_VERTICES;i++) {
  uint count=uint(contact_count_load(i));
  if(count>CONTACT_CAPACITY)return;
  for(uint k=0;k<count;k++) {
   uint slot=i*CONTACT_CAPACITY+k;if(uint(contact_owner_load(slot))!=i)continue;
   uint4 ids=uint4(contact_ids_load(slot));float4 w=contact_weights_load(slot),plane=contact_plane_load(slot);
   float4 mass=float4(mass_load(ids.x),mass_load(ids.y),mass_load(ids.z),mass_load(ids.w));float den=dot(mass,w*w);if(den<=0)continue;
   float3 delta=w.x*cloth_x[ids.x]+w.y*cloth_x[ids.y]+w.z*cloth_x[ids.z]+w.w*cloth_x[ids.w];
   float C=dot(delta,plane.xyz)-plane.w,old=contact_lambda_load(slot),dl=max(-old,-C/den),next=old+dl;
   contact_lambda_store(slot,next);float3 impulse=dl*plane.xyz;
   if(uint(contact_kind_load(slot))>=2) {
    float3 previous=contact_tangent_lambda_load(slot);
    float3 before=w.x*old_load(ids.x)+w.y*old_load(ids.y)+w.z*old_load(ids.z)+w.w*old_load(ids.w);
    float3 travel=delta-before-contact_motion_load(slot);travel-=dot(travel,plane.xyz)*plane.xyz;
    float3 target=previous-travel/den;float limit=.6*next;float magnitude=length(target);
    // Below 2^-51 the squared norm is under 2^-102: still a normal float, but
    // within 2^24 of the subnormal boundary (2^-126), where a square keeps
    // fewer than a float's digits before it reaches zero. Normalise by the
    // largest component first.
    if((magnitude<4.4408921e-16||!isfinite(magnitude))&&any(target!=float3(0))){float scale=max(abs(target.x),max(abs(target.y),abs(target.z)));magnitude=scale*length(target/scale);}
    if(magnitude>0)target*=min(1.,limit/magnitude);
    contact_tangent_lambda_store(slot,target);impulse+=target-previous;
   }
   for(uint j=0;j<4;j++)if(abs(w[j])>1e-6&&mass[j]>0)cloth_x[ids[j]]+=mass[j]*w[j]*impulse;
  }
 }
}
