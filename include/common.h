#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <limits.h>
#include <stdint.h>

#ifdef MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#ifdef OPENMP
    #include <omp.h>
#endif

#ifdef MPI
    #include <mpi.h>
    #ifndef SIZE_MAX
        #error "SIZE_MAX not defined"
    #endif
    #if SIZE_MAX == UCHAR_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
    #elif SIZE_MAX == USHRT_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
    #elif SIZE_MAX == UINT_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED
    #elif SIZE_MAX == ULONG_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
    #elif SIZE_MAX == ULLONG_MAX
       #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
    #else
       #error "No MPI data type fits size_t"
    #endif
#endif

#ifdef STARPU
    #include <starpu.h>
#endif

#ifdef GSL
    #include <gsl/gsl_sf.h>
#endif

#define STARSH_MALLOC_FAILED 1

#define STARSH_ERROR(format, ...)\
{\
    fprintf(stderr, "STARSH ERROR: %s(): ", __func__);\
    fprintf(stderr, format, ##__VA_ARGS__);\
    fprintf(stderr, "\n");\
}

#ifdef SHOW_WARNINGS
    #define STARSH_WARNING(format, ...)\
    {\
        fprintf(stderr, "STARSH WARNING: %s(): ", __func__);\
        fprintf(stderr, format, ##__VA_ARGS__);\
        fprintf(stderr, "\n");\
    }
#else
    #define STARSH_WARNING(...)
#endif

#define STARSH_MALLOC(var, expr_nitems)\
{\
    var = malloc(sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("line %d: malloc() failed", __LINE__);\
        return 1;\
    }\
}

#define STARSH_REALLOC(var, expr_nitems)\
{\
    var = realloc(var, sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        return 1;\
    }\
}

#define STARSH_PMALLOC(var, expr_nitems, var_info)\
{\
    var = malloc(sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        var_info = 1;\
    }\
}

#define STARSH_PREALLOC(var, expr_nitems, var_info)\
{\
    var = realloc(var, sizeof(*var)*(expr_nitems));\
    if(!var)\
    {\
        STARSH_ERROR("malloc() failed");\
        var_info = 1;\
    }\
}

#endif // __COMMON_H__
