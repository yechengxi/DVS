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
    return Py_BuildValue("");
};


static PyObject* dvs_error(PyObject* self, PyObject* args)
{
    PyArrayObject *in_array;
    PyArrayObject *out_array;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in_array, &PyArray_Type, &out_array))
        return NULL;

    npy_intp idims[3];
    idims[0] = PyArray_DIM(in_array, 0);
    idims[1] = PyArray_DIM(in_array, 1);
    idims[2] = PyArray_DIM(in_array, 2);

    npy_intp odims[3];
    odims[0] = PyArray_DIM(out_array, 0);
    odims[1] = PyArray_DIM(out_array, 1);

    for (int i = 0; i < 2; ++i) if (idims[i] != odims[i]) return NULL;
    if (idims[2] != 3) return NULL;

    float *in_dataptr  = (float *) PyArray_DATA(in_array);
    float *out_dataptr = (float *) PyArray_DATA(out_array);

    float dx = 0, dy = 0, rot = 0, div = 0, cnt = 0, nz_avg = 0;
    for (unsigned int y = 1; y < idims[0] - 1; ++y) {
        for (unsigned int x = 1; x < idims[1] - 1; ++x) {
            unsigned long idx0 = (y - 1) * idims[1] + x;
            unsigned long idx1 = (y - 0) * idims[1] + x;
            unsigned long idx2 = (y + 1) * idims[1] + x;

            out_dataptr[idx1 * 2 + 0] = 0;
            out_dataptr[idx1 * 2 + 1] = 0;

            float lcnt = in_dataptr[idx1 * 3 + 0] + in_dataptr[idx1 * 3 + 2];
            if (lcnt < 0.5)
                continue;

            float a00 = in_dataptr[(idx0 - 1) * 3 + 1];
            float a01 = in_dataptr[(idx0 - 0) * 3 + 1];
            float a02 = in_dataptr[(idx0 + 1) * 3 + 1];

            float a10 = in_dataptr[(idx1 - 1) * 3 + 1];
            //float a11 = in_dataptr[(idx1 - 0) * 3 + 1];
            float a12 = in_dataptr[(idx1 + 1) * 3 + 1];

            float a20 = in_dataptr[(idx2 - 1) * 3 + 1];
            float a21 = in_dataptr[(idx2 - 0) * 3 + 1];
            float a22 = in_dataptr[(idx2 + 1) * 3 + 1];

            float dy_ = 3 * (a00 - a02) + 10 * (a10 - a12) + 3 * (a20 - a22);
            float dx_ = 3 * (a00 - a20) + 10 * (a01 - a21) + 3 * (a02 - a22);

            out_dataptr[idx1 * 2 + 0] = dx;
            out_dataptr[idx1 * 2 + 1] = dy;

            nz_avg += lcnt;
            cnt += 1;

            dx += dx_;
            dy += dy_;

            float rx = x - idims[1] / 2, ry = y - idims[0] / 2;
                    
            rot += rx * dy_ - ry * dx_;
            div += rx * dx_ + ry * dy_;
        }
    }

    // Another magic number
    if (cnt < 100) {
        dx = 0;
        dy = 0;    
        rot = 0;
        div = 0;
        nz_avg = 0;
    } else {
        dx /= cnt;
        dy /= cnt;
        rot /= cnt;
        div /= cnt;
        nz_avg /= cnt;
    }

    //Py_INCREF(out_array);
    return Py_BuildValue("ffffff", dx, dy, rot, div, cnt, nz_avg);
};



/*  define functions in module */
static PyMethodDef DVSMethods[] =
{
     {"dvs_img", dvs_img, METH_VARARGS,
         "compute dvs image from event cloud"},
     {"dvs_error", dvs_error, METH_VARARGS,
         "compute errors on the dvs image"},     
     {NULL, NULL, 0, NULL}
};


#if 0 // Python 2.x


/* module initialization */
PyMODINIT_FUNC
initpydvs(void)
{
     (void) Py_InitModule("pydvs", DVSMethods);
     /* IMPORTANT: this must be called */
     import_array();
}
#endif

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "pydvs",    /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in
                    global variables. */
    DVSMethods
};


/* module initialization */
PyMODINIT_FUNC
PyInit_pydvs(void)
{
     /* IMPORTANT: this must be called */
     import_array();
     return PyModule_Create(&cModPyDem);
};
