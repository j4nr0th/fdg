#ifndef FDG_MESH_OBJECTS_H
#define FDG_MESH_OBJECTS_H

#include "../topology/mesh.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    topo_mesh_t *mesh;
} mesh_object;

FDG_INTERNAL
extern PyType_Spec mesh_type_spec;

#endif // FDG_MESH_OBJECTS_H
