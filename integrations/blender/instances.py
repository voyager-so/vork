"""Materialize evaluated particle occurrences for the glTF snapshot export.

Blender's glTF selection filter treats particle templates as ordinary children,
so templates in hidden collections are dropped even when their emitter renders.
Collection instancers are already handled by glTF with inherited selection.
"""
from contextlib import contextmanager

SOURCE = '_vork_instance_source'


@contextmanager
def particle_occurrences(scene, *, animated=False):
    import bpy
    from .render_geometry import direct_render_objects
    from .visibility import FLAGS
    ray_flags = (*FLAGS, 'visible_volume_scatter')
    visible = direct_render_objects(bpy.context.view_layer.layer_collection)
    emitters = {obj for obj in bpy.data.objects if any(
        modifier.type == 'PARTICLE_SYSTEM' and modifier.show_render
        and modifier.particle_system.settings.render_type in {'OBJECT', 'COLLECTION'}
        for modifier in obj.modifiers)}
    if not emitters:
        yield 0
        return
    objects, meshes, modifiers, hidden_emitters = [], {}, [], []
    try:
        deps = bpy.context.evaluated_depsgraph_get()
        active_emitters = emitters & visible
        active_emitters.update(i.object.original for i in deps.object_instances
                              if i.is_instance and i.parent and i.parent.original in visible
                              and i.object.original in emitters)
        if animated and active_emitters:
            raise ValueError('Particle instance animation cannot be exported; export one evaluated frame')
        # Copy iterator-owned transforms before changing the dependency graph.
        occurrences = [(i.object.original, i.matrix_world.copy(),
                        {flag: getattr(i.object, flag) and getattr(i.parent, flag) for flag in ray_flags}) for i in deps.object_instances
                       if i.is_instance and i.parent and i.parent.original in visible
                       and i.particle_system is not None
                       and i.particle_system.settings.render_type in {'OBJECT', 'COLLECTION'}
                       and i.object.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'} and i.show_self]
        if not occurrences:
            yield 0
            return
        for source, _, _ in occurrences:
            key = source.as_pointer()
            if key not in meshes:
                meshes[key] = bpy.data.meshes.new_from_object(source.evaluated_get(deps), preserve_all_data_layers=True, depsgraph=deps)
        for source, transform, visibility in occurrences:
            key = source.as_pointer()
            obj = bpy.data.objects.new(source.name + ' occurrence', meshes[key])
            objects.append(obj)
            obj[SOURCE] = source.name
            for flag, value in visibility.items():
                setattr(obj, flag, value)
            scene.collection.objects.link(obj)
            obj.matrix_world = transform
        for emitter in emitters:
            for modifier in emitter.modifiers:
                if modifier.type == 'PARTICLE_SYSTEM':
                    modifiers.append((modifier, modifier.show_viewport, modifier.show_render))
                    modifier.show_viewport = modifier.show_render = False
            if not emitter.show_instancer_for_render:
                hidden_emitters.append((emitter, emitter.data, emitter.hide_render))
                # Collection occurrences inherit visibility from their instancer,
                # so hide_render alone cannot suppress the carrier there.
                emitter.data = bpy.data.meshes.new('particle carrier')
                emitter.hide_render = True
        bpy.context.view_layer.update()
        yield len(objects)
    finally:
        if objects:
            bpy.data.batch_remove(ids=objects)
        if meshes:
            bpy.data.batch_remove(ids=list(meshes.values()))
        for modifier, viewport, render in modifiers:
            modifier.show_viewport, modifier.show_render = viewport, render
        for emitter, data, hidden in hidden_emitters:
            empty = emitter.data
            emitter.data, emitter.hide_render = data, hidden
            bpy.data.meshes.remove(empty)
        bpy.context.view_layer.update()
