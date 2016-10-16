#ifndef _STARS_H_
#define _STARS_H_

typedef struct Array Array;
typedef struct STARS_Problem STARS_Problem;
typedef struct STARS_BLR STARS_BLR;
typedef struct STARS_BLRmatrix STARS_BLRmatrix;

typedef struct Array *(*block_kernel)(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data);

struct Array
// N-dimensional array
{
    int ndim;
    // Number of dimensions of array.
    int *shape;
    // Shape of array.
    int *stride;
    // Size of step to increase value of corresponding axis by 1
    char order;
    // C or Fortran order. C-order means stride is descending, Fortran-order
    // means stride is ascending.
    int size;
    // Number of elements of an array.
    char dtype;
    // Data type of each element of array. Possile value is 's', 'd', 'c' or
    // 'z', much like in names of LAPACK routines.
    size_t nbytes;
    // Size of buffer in bytes.
    void *buffer;
    // Buffer, containing array. Stored in Fortran style.
};

// Routines to work with N-dimensional arrays
Array *Array_new(int ndim, int *shape, char dtype, char order);
// Allocation of memory for array
Array *Array_new_like(Array *array);
// Allocation of memory for array of the same shape and dtype, as given array
Array *Array_copy(Array *array);
// Create copy of array, ordering is the same
void Array_free(Array *array);
// Free data and auxiliary buffers of array
void Array_info(Array *array);
// Print short info about array (shape, stride, size, dtype and order)
void Array_print(Array *array);
// Print elements of array, different rows of array are printed on different
// rows of output
void Array_init(Array *array, char *kind);
// Initialize array in a given manner (by kind)
void Array_init_randn(Array *array);
// Set every element of array to random with normal (0,1) distribution
void Array_init_rand(Array *array);
// Set every element of array to random with uniform distribution on [0;1]
void Array_init_zeros(Array *array);
// Set every element of array to zero
void Array_init_ones(Array *array);
// Set every element of array to one
void Array_tomatrix(Array *array, char kind);
// Reshape array to matrix (2-dimensional array) by assuming array as long rows
// or as long columns
void Array_trans(Array *array);
// Inplace transposition of array without any memory movements, only change is
// in shape, stride and order
Array *Array_dot(Array* A, Array *B);
// Multiplication of two arrays, they should have the same shape of last
// dimension of first array and first dimension of last array
void Array_SVD(Array *array, Array **U, Array **S, Array **V);
// Compute short SVD of 2-dimensional array
void Array_scale(Array *array, char kind, Array *factor);
// Apply row or column scaling to array
double Array_error(Array *array, Array *array2);
// Measure Frobenius error of approximating array with array2
double Array_norm(Array *array);
// Measure Frobenius norm of array
Array *Array_convert(Array *array, char dtype);
// Copy array and convert data type

struct STARS_Problem
// Structure, storing all the necessary data for reconstruction of a matrix,
// generated by given kernel. Matrix elements are not stored in memory, but
// computed on demand.
{
    int nrows, ncols;
    // Number of rows and columns of corresponding matrix.
    char symm;
    // 'S' if problem is symmetric, and 'N' otherwise.
    char dtype;
    // Possible values are 's', 'd', 'c' or 'z', just as in LAPACK routines
    // names.
    void *row_data, *col_data;
    // Pointers to data, corresponding to rows and columns.
    block_kernel kernel;
    // Pointer to a function, returning submatrix on intersection of
    // given rows and columns.
    char *type;
    // Type of problem, useful for debugging and printing additional info.
    // It is up to user to set it as desired.
};

STARS_Problem *STARS_Problem_init(int nrows, int ncols, char symm, char dtype,
        void *row_data, void*col_data, block_kernel kernel, char *type);
void STARS_Problem_info(STARS_Problem *problem);

struct STARS_BLR
// Block-low rank format. Entire matrix is divided into blocks by a grid.
// Some of blocks are low-rank, some are not.
{
    STARS_Problem *problem;
    // Pointer to a problem.
    char symm;
    // 'S' if format and problem are symmetric, and 'N' otherwise.
    int nrows, ncols;
    // Number of rows and columns of corresponding matrix.
    int *row_order, *col_order;
    // Permutation of rows and columns, such that each block is based on rows
    // and columns, going one after another.
    int nbrows, nbcols;
    // Number of block rows and block columns.
    int *ibrow_start, *ibcol_start;
    // Start row/column of each block rows/block column.
    int *ibrow_size, *ibcol_size;
    // Number of rows/columns of each block row/block column.
};

struct STARS_BLRmatrix
// Approximation in block low-rank format.
{
    STARS_Problem *problem;
    // Pointer to a problem.
    STARS_BLR *format;
    // Pointer to block low-rank format.
    int bcount;
    // Number of blocks
    int *bindex;
    // block row and block column index as a pair of integers (bi, bj)
    int *brank;
    // Rank of each block or -1 if block os not low-rank.
    Array **U, **V;
    // Arrays of pointers to low-rank factors U and V of each block.
    Array **A;
    // Array of pointers to data of full-rank blocks.
};

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        int maxrank, double tol);
void STARS_BLRmatrix_info(STARS_BLRmatrix *mat);
void STARS_BLRmatrix_free(STARS_BLRmatrix *mat);
void STARS_BLR_info(STARS_BLR *format);
void STARS_BLR_free(STARS_BLR *format);
void STARS_BLRmatrix_error(STARS_BLRmatrix *mat);
void STARS_BLRmatrix_getblock(STARS_BLRmatrix *mat, int i, int j,
        int *block_size, int *rank, void **U, void **V, void **A);
void STARS_BLR_getblock(STARS_BLR *format, int i, int j, int *block_size,
        void **A);
void STARS_BLRmatrix_printKADIR(STARS_BLRmatrix *mat);
#endif // _STARS_H_
