#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


static PyObject* dvs_img(PyObject* self, PyObject* args)
{
    PyArrayObject *in_array;
    PyArrayObject *out_array;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in_array, &PyArray_Type, &out_array))
        return NULL;

    npy_intp dims[3];
    dims[0] = PyArray_DIM(out_array, 0);
    dims[1] = PyArray_DIM(out_array, 1);
    dims[2] = PyArray_DIM(out_array, 2);

    float *in_dataptr  = (float *) PyArray_DATA(in_array);
    float *out_dataptr = (float *) PyArray_DATA(out_array);

    int n_events = PyArray_SIZE(in_array) / 4;
    float t0   = in_dataptr[0];
    for (unsigned long i = 0; i < n_events; ++i) { 
        float t   = in_dataptr[i * 4 + 0];
        int x     = in_dataptr[i * 4 + 1];
        int y     = in_dataptr[i * 4 + 2];
        float p_f = in_dataptr[i * 4 + 3];

        if ((y >= dims[0]) || (x >= dims[1]) || (y < 0) || (x < 0))
            continue;

        unsigned long idx = y * dims[1] + x;
        
        // Time image
        out_dataptr[idx * 3 + 1] += (t - t0);
        // Event-count images
        if (p_f > 0.5) {
            //(*(float*)PyArray_GETPTR3(out_array, y, x, 0)) += 1;
            out_dataptr[idx * 3 + 0] += 1;
        }
        else {
            out_dataptr[idx * 3 + 2] += 1;
        }
    }

    // Normalize time image    
    for (unsigned long i = 0; i < dims[0] * dims[1]; ++i) {
        float div = out_dataptr[i * 3 + 0] + out_dataptr[i * 3 + 2];
        if (div > 0.5) // It can actually only be an integer, like 0, 1, 2...
            out_dataptr[i * 3 + 1] /= div;
    }

    //Py_INCREF(out_array);
    return Py_BuildValue("f", 0);
};


/*  define functions in module */
static PyMethodDef DVSMethods[] =
{
     {"dvs_img", dvs_img, METH_VARARGS,
         "compute dvs image from event cloud"},
     {NULL, NULL, 0, NULL}
};


#if 0 // Python 2.x


/* module initialization */
PyMODINIT_FUNC
initlibdvs(void)
{
     (void) Py_InitModule("libdvs", DVSMethods);
     /* IMPORTANT: this must be called */
     import_array();
}
#endif

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "libdvs",    /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in
                    global variables. */
    DVSMethods
};


/* module initialization */
PyMODINIT_FUNC
PyInit_libdvs(void)
{
     /* IMPORTANT: this must be called */
     import_array();
     return PyModule_Create(&cModPyDem);
};
