/*! @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/openmp/blrm/drsdd.c
 * @version 0.3.1
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "plasma.h"
#include "plasma/control/common.h"
#include "common.h"
#include "starsh.h"

struct Param {
    double tol;
    int maxrank;
    STARSH_int nblocks_far;
    STARSH_int *block_far;
    STARSH_kernel *kernel;
    STARSH_cluster *RC, *CC;
    Array **far_U, **far_V, **near_D;
    int *far_rank;
    int oversample;
    int onfly;
};

#define A(m, n) BLKADDR(A, double, m, n)

/***************************************************************************//**
 *  Parallel tile Cholesky factorization - static scheduling
 **/
void drsdd_pdpotrf(plasma_context_t *plasma) {
    const struct Param *param = plasma->aux;
    PLASMA_enum uplo;
    PLASMA_desc A;
    PLASMA_sequence *sequence;
    PLASMA_request *request;

    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int ldak, ldam, ldan;
    int info;
    int tempkn, tempmn;

    double zone = (double) 1.0;
    double mzone = (double) -1.0;

    plasma_unpack_args_4(uplo, A, sequence, request);
    if (sequence->status != PLASMA_SUCCESS)
        return;
    ss_init(A.nt, A.nt, 0);

    k = 0;
    m = PLASMA_RANK;
    while (m >= A.nt) {
        k++;
        m = m - A.nt + k;
    }
    n = 0;

    while (k < A.nt && m < A.nt && !ss_aborted()) {
        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += PLASMA_SIZE;
            while (next_m >= A.nt && next_k < A.nt) {
                next_k++;
                next_m = next_m - A.nt + next_k;
            }
            next_n = 0;
        }

        tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
        tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

        ldak = BLKLDD(A, k);
        ldan = BLKLDD(A, n);
        ldam = BLKLDD(A, m);

        if (m == k) {
            if (n == k) {
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    CORE_dpotrf(
                            PlasmaLower,
                            tempkn,
                            A(k, k), ldak,
                            &info);
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dpotrf(
                            PlasmaUpper,
                            tempkn,
                            A(k, k), ldak,
                            &info);
                }
                if (info != 0) {
                    plasma_request_fail(sequence, request, info + A.nb * k);
                    ss_abort();
                }
                ss_cond_set(k, k, 1);
            } else {
                ss_cond_wait(k, n, 1);
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    CORE_dsyrk(
                            PlasmaLower, PlasmaNoTrans,
                            tempkn, A.nb,
                            -1.0, A(k, n), ldak,
                            1.0, A(k, k), ldak);
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dsyrk(
                            PlasmaUpper, PlasmaTrans,
                            tempkn, A.nb,
                            -1.0, A(n, k), ldan,
                            1.0, A(k, k), ldak);
                }
            }
        } else {
            if (n == k) {
                ss_cond_wait(k, k, 1);
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    CORE_dtrsm(
                            PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
                            tempmn, A.nb,
                            zone, A(k, k), ldak,
                            A(m, k), ldam);
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dtrsm(
                            PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit,
                            A.nb, tempmn,
                            zone, A(k, k), ldak,
                            A(k, m), ldak);
                }
                ss_cond_set(m, k, 1);
            } else {
                ss_cond_wait(k, n, 1);
                ss_cond_wait(m, n, 1);
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    CORE_dgemm(
                            PlasmaNoTrans, PlasmaTrans,
                            tempmn, A.nb, A.nb,
                            mzone, A(m, n), ldam,
                            A(k, n), ldak,
                            zone, A(m, k), ldam);
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dgemm(
                            PlasmaTrans, PlasmaNoTrans,
                            A.nb, tempmn, A.nb,
                            mzone, A(n, k), ldan,
                            A(n, m), ldan,
                            zone, A(k, m), ldak);
                }
            }
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void drsdd_pdpotrf_quark(PLASMA_enum uplo, PLASMA_desc A,
                         PLASMA_sequence *sequence, PLASMA_request *request) {
    abort();
}

int PLASMA_dpotrf_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A,
                             PLASMA_sequence *sequence, PLASMA_request *request) {
    PLASMA_desc descA;
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    if (sequence == NULL) {
        plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
        return PLASMA_ERR_UNALLOCATED;
    }
    if (request == NULL) {
        plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
        return PLASMA_ERR_UNALLOCATED;
    }
    /* Check sequence status */
    if (sequence->status == PLASMA_SUCCESS)
        request->status = PLASMA_SUCCESS;
    else
        return plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);

    /* Check descriptors for correctness */
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dpotrf_Tile_Async", "invalid descriptor");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    } else {
        descA = *A;
    }
    /* Check input arguments */
    if (descA.nb != descA.mb) {
        plasma_error("PLASMA_dpotrf_Tile_Async", "only square tiles supported");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
        return plasma_request_fail(sequence, request, -1);
    }
    /* Quick return */
/*
    if (max(N, 0) == 0)
        return PLASMA_SUCCESS;
*/
    // TODO (Jie): may support dynamic
    plasma->scheduling = 1;
    plasma_parallel_call_4(drsdd_pdpotrf,
                           PLASMA_enum, uplo,
                           PLASMA_desc, descA,
                           PLASMA_sequence*, sequence,
                           PLASMA_request*, request);

    return PLASMA_SUCCESS;
}

int PLASMA_dpotrf(PLASMA_enum uplo, int N,
                  double *A, int LDA) {
    int NB;
    int status;
    plasma_context_t *plasma;
    PLASMA_sequence *sequence = NULL;
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
    PLASMA_desc descA;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("PLASMA_dpotrf", "illegal value of uplo");
        return -1;
    }
    if (N < 0) {
        plasma_error("PLASMA_dpotrf", "illegal value of N");
        return -2;
    }
    if (LDA < max(1, N)) {
        plasma_error("PLASMA_dpotrf", "illegal value of LDA");
        return -4;
    }
    /* Quick return */
    if (max(N, 0) == 0)
        return PLASMA_SUCCESS;

    /* Tune NB depending on M, N & NRHS; Set NBNB */
    status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
    if (status != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
        return status;
    }

    /* Set NT */
    NB = PLASMA_NB;

    plasma_sequence_create(plasma, &sequence);

    if (PLASMA_TRANSLATION == PLASMA_OUTOFPLACE) {
        plasma_dooplap2tile(descA, A, NB, NB, LDA, N, 0, 0, N, N, sequence, &request,
                            plasma_desc_mat_free(&(descA)));
    } else {
        plasma_diplap2tile(descA, A, NB, NB, LDA, N, 0, 0, N, N,
                           sequence, &request);
    }

    /* Call the tile interface */
    PLASMA_dpotrf_Tile_Async(uplo, &descA, sequence, &request);

    if (PLASMA_TRANSLATION == PLASMA_OUTOFPLACE) {
        plasma_dooptile2lap(descA, A, NB, NB, LDA, N, sequence, &request);
        plasma_dynamic_sync();
        plasma_desc_mat_free(&descA);
    } else {
        plasma_diptile2lap(descA, A, NB, NB, LDA, N, sequence, &request);
        plasma_dynamic_sync();
    }

    status = sequence->status;
    plasma_sequence_destroy(plasma, sequence);

    return status;
}

STARSH_int twoD_2_oneD(STARSH_int i, STARSH_int j) {
    return j + (1 + i) * i / 2;
}

int static_malloc_each(const struct Param *param, STARSH_int i, STARSH_int j) {
    double tol = param->tol;
    int maxrank = param->maxrank;
    STARSH_int nblocks_far = param->nblocks_far;
    STARSH_int *block_far = param->block_far;
    STARSH_kernel *kernel = param->kernel;
    STARSH_cluster *RC = param->RC, *CC = param->CC;
    Array **far_U = param->far_U, **far_V = param->far_V, **near_D = param->near_D;
    int *far_rank = param->far_rank;
    const int oversample = param->oversample;
    const int onfly = param->onfly;

    void *RD = RC->data, *CD = CC->data;
    double drsdd_time = 0;
    double kernel_time = 0;
    int BAD_TILE;

    {
        // TODO (Jie): support symm
        STARSH_int bi = j + (1 + i) * i / 2;
        // Get indexes of corresponding block row and block column
        assert(i == block_far[2 * bi]);
        assert(j == block_far[2 * bi + 1]);
        // Get corresponding sizes and minimum of them
        int nrows = RC->size[i];
        int ncols = CC->size[j];
        if (nrows != ncols && BAD_TILE == 0) {
#pragma omp critical
            BAD_TILE = 1;
            STARSH_WARNING("This was only tested on square tiles, error of "
                           "approximation may be much higher, than demanded");
        }
        int mn = nrows < ncols ? nrows : ncols;
        int mn2 = maxrank + oversample;
        if (mn2 > mn)
            mn2 = mn;
        // Get size of temporary arrays
        int lwork = ncols, lwork_sdd = (4 * mn2 + 7) * mn2;
        if (lwork_sdd > lwork)
            lwork = lwork_sdd;
        lwork += (size_t) mn2 * (2 * ncols + nrows + mn2 + 1);
        int liwork = 8 * mn2;
        int info;
        // Allocate temporary arrays
        double *data;
        STARSH_PMALLOC(data, (size_t) nrows * (size_t) ncols, info);
        memset(data, 0, (size_t) nrows * (size_t) ncols * sizeof(*data));


        far_rank[bi] = -1;
        int shape[2] = {nrows, ncols};
        // TODO (Jie): fix memory leaking of D
        array_from_buffer(near_D + bi, 2, shape, 'd', 'F', data);

    }
    return 0;
}

int static_gen_each(const struct Param *param, STARSH_int i, STARSH_int j) {
    double tol = param->tol;
    int maxrank = param->maxrank;
    STARSH_int nblocks_far = param->nblocks_far;
    STARSH_int *block_far = param->block_far;
    STARSH_kernel *kernel = param->kernel;
    STARSH_cluster *RC = param->RC, *CC = param->CC;
    Array **far_U = param->far_U, **far_V = param->far_V, **near_D = param->near_D;
    int *far_rank = param->far_rank;
    const int oversample = param->oversample;
    const int onfly = param->onfly;

    void *RD = RC->data, *CD = CC->data;
    double drsdd_time = 0;
    double kernel_time = 0;
    int BAD_TILE;

    {
        // TODO (Jie): support symm
        STARSH_int bi = j + (1 + i) * i / 2;
        // Get indexes of corresponding block row and block column
        assert(i == block_far[2 * bi]);
        assert(j == block_far[2 * bi + 1]);
        // Get corresponding sizes and minimum of them
        int nrows = RC->size[i];
        int ncols = CC->size[j];
        if (nrows != ncols && BAD_TILE == 0) {
#pragma omp critical
            BAD_TILE = 1;
            STARSH_WARNING("This was only tested on square tiles, error of "
                           "approximation may be much higher, than demanded");
        }
        int mn = nrows < ncols ? nrows : ncols;
        int mn2 = maxrank + oversample;
        if (mn2 > mn)
            mn2 = mn;
        // Get size of temporary arrays
        int lwork = ncols, lwork_sdd = (4 * mn2 + 7) * mn2;
        if (lwork_sdd > lwork)
            lwork = lwork_sdd;
        lwork += (size_t) mn2 * (2 * ncols + nrows + mn2 + 1);
        int liwork = 8 * mn2;
        int info;
        // Compute elements of a block
        double time0 = omp_get_wtime();
        double *D = near_D[bi]->data;


        kernel(nrows, ncols, RC->pivot + RC->start[i], CC->pivot + CC->start[j],
               RD, CD, D, nrows);
        double time1 = omp_get_wtime();
#pragma omp critical
        {
            kernel_time += time1 - time0;
        }
    }
    return 0;
}

int static_compress_each(const struct Param *param, STARSH_int i, STARSH_int j) {
    double tol = param->tol;
    int maxrank = param->maxrank;
    STARSH_int nblocks_far = param->nblocks_far;
    STARSH_int *block_far = param->block_far;
    STARSH_kernel *kernel = param->kernel;
    STARSH_cluster *RC = param->RC, *CC = param->CC;
    Array **far_U = param->far_U, **far_V = param->far_V, **near_D = param->near_D;
    int *far_rank = param->far_rank;
    const int oversample = param->oversample;
    const int onfly = param->onfly;

    void *RD = RC->data, *CD = CC->data;
    double drsdd_time = 0;
    double kernel_time = 0;
    int BAD_TILE;

    {
        // TODO (Jie): support symm
        STARSH_int bi = twoD_2_oneD(i, j);
        // Get indexes of corresponding block row and block column
        assert(i == block_far[2 * bi]);
        assert(j == block_far[2 * bi + 1]);
        // Get corresponding sizes and minimum of them
        int nrows = RC->size[i];
        int ncols = CC->size[j];
        if (nrows != ncols && BAD_TILE == 0) {
#pragma omp critical
            BAD_TILE = 1;
            STARSH_WARNING("This was only tested on square tiles, error of "
                           "approximation may be much higher, than demanded");
        }
        int mn = nrows < ncols ? nrows : ncols;
        int mn2 = maxrank + oversample;
        if (mn2 > mn)
            mn2 = mn;
        // Get size of temporary arrays
        int lwork = ncols, lwork_sdd = (4 * mn2 + 7) * mn2;
        if (lwork_sdd > lwork)
            lwork = lwork_sdd;
        lwork += (size_t) mn2 * (2 * ncols + nrows + mn2 + 1);
        int liwork = 8 * mn2;
        int info;
        // Allocate temporary arrays
        double *D = near_D[bi]->data;
        double *work;
        int *iwork;
        double *backup;
        STARSH_PMALLOC(work, lwork, info);
        STARSH_PMALLOC(iwork, liwork, info);
        STARSH_PMALLOC(backup, (size_t)ncols * (size_t)nrows, info);
        memcpy(backup, D, (size_t)ncols * (size_t)nrows * sizeof(*D));

        double time1 = omp_get_wtime();
        starsh_dense_dlrrsdd(nrows, ncols, D, nrows, far_U[bi]->data, nrows,
                             far_V[bi]->data, ncols, far_rank + bi, maxrank, oversample, tol,
                             work, lwork, iwork);
        double time2 = omp_get_wtime();
#pragma omp critical
        {
            drsdd_time += time2 - time1;
        }

        free(work);
        free(iwork);

//        if (i == j + 1)
        far_rank[bi] = -1;
        // Compute elements of a block
        if (far_rank[bi] == -1 && !onfly) {
            memcpy(D, backup, (size_t)ncols * (size_t)nrows * sizeof(*D));
        } else {
            array_free(near_D[bi]);
        }
        free(backup);
    }
    return 0;
}

int static_gen(struct Param *param) {
    double tol = param->tol;
    int maxrank = param->maxrank;
    STARSH_int nblocks_far = param->nblocks_far;
    STARSH_int *block_far = param->block_far;
    STARSH_kernel *kernel = param->kernel;
    STARSH_cluster *RC = param->RC, *CC = param->CC;
    Array **far_U = param->far_U, **far_V = param->far_V, **near_D, **near_D_final;
    int *far_rank = param->far_rank;
    const int oversample = param->oversample;
    const int onfly = param->onfly;

    void *RD = RC->data, *CD = CC->data;
    double drsdd_time;
    double kernel_time;
    int BAD_TILE;
    STARSH_MALLOC(near_D, nblocks_far);
    param->near_D = near_D;

    for (STARSH_int i = 0; i < RC->nblocks; ++i) {
        for (STARSH_int j = 0; j <= i; ++j) {
            static_malloc_each(param, i, j);
            static_gen_each(param, i, j);
            static_compress_each(param, i, j);
        }
    }
    STARSH_int near_D_counter = 0;
    for (STARSH_int i = 0; i < nblocks_far; ++i) {
        if (far_rank[i] == -1) {
            near_D_counter++;
        }
    }


    STARSH_MALLOC(near_D_final, near_D_counter);
    near_D_counter = 0;
    for (STARSH_int i = 0; i < nblocks_far; ++i) {
        if (far_rank[i] == -1) {
            near_D_final[near_D_counter] = near_D[i];
            near_D_counter++;
        }
    }
    free(near_D);
    param->near_D = near_D_final;
    return 0;
}

void drsdd_pdpotrf_testing(plasma_context_t *plasma) {
    const struct Param *param = plasma->aux;
    PLASMA_enum uplo;
    PLASMA_desc A;
    PLASMA_sequence *sequence;
    PLASMA_request *request;


    double tol = param->tol;
    int maxrank = param->maxrank;
    STARSH_int nblocks_far = param->nblocks_far;
    STARSH_int *block_far = param->block_far;
    STARSH_kernel *kernel = param->kernel;
    STARSH_cluster *RC = param->RC, *CC = param->CC;
    Array **far_U = param->far_U, **far_V = param->far_V, **near_D, **near_D_final;
    int *far_rank = param->far_rank;
    const int oversample = param->oversample;
    const int onfly = param->onfly;

    void *RD = RC->data, *CD = CC->data;
    double drsdd_time;
    double kernel_time;
    int BAD_TILE;
    near_D = param->near_D;


    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int ldak, ldam, ldan;
    int info = 0;
    int tempkn, tempmn;

    double zone = (double) 1.0;
    double mzone = (double) -1.0;

    plasma_unpack_args_3(uplo, sequence, request);

    A.nt = RC->nblocks;
    A.nb = RC->size[0];
    A.n = RC->ndata;

    if (sequence->status != PLASMA_SUCCESS)
        return;
    ss_init(A.nt, A.nt, 0);

    k = 0;
    m = PLASMA_RANK;
    while (m >= A.nt) {
        k++;
        m = m - A.nt + k;
    }
    n = 0;

    while (k < A.nt && m < A.nt && !ss_aborted()) {
        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += PLASMA_SIZE;
            while (next_m >= A.nt && next_k < A.nt) {
                next_k++;
                next_m = next_m - A.nt + next_k;
            }
            next_n = 0;
        }

        tempkn = k == A.nt - 1 ? A.n - k * A.nb : A.nb;
        tempmn = m == A.nt - 1 ? A.n - m * A.nb : A.nb;

        ldak = RC->size[k];
        ldan = RC->size[n];
        ldam = RC->size[m];

        if (m == k) {
            if (n == k) {
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    if (k == 0) {
                        static_malloc_each(param, k, k);
                    }
                    static_gen_each(param, k, k);
                    CORE_dpotrf(
                            PlasmaLower,
                            tempkn,
                            near_D[twoD_2_oneD(k, k)]->data, ldak,
                            &info);
//                    CORE_dpotrf(
//                            PlasmaLower,
//                            tempkn,
//                            A(k, k), ldak,
//                            &info);
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dpotrf(
                            PlasmaUpper,
                            tempkn,
                            A(k, k), ldak,
                            &info);
                }
                if (info != 0) {
                    plasma_request_fail(sequence, request, info + A.nb * k);
                    ss_abort();
                }
                ss_cond_set(k, k, 1);
            } else {
                ss_cond_wait(k, n, 1);
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    STARSH_int index = twoD_2_oneD(k, n);
                    if (near_D[twoD_2_oneD(k, k)] == NULL) {
                        static_malloc_each(param, k, k);
                    }
                    if (far_rank[index] == -1) {
                        CORE_dsyrk(
                                PlasmaLower, PlasmaNoTrans,
                                tempkn, A.nb,
                                -1.0, near_D[twoD_2_oneD(k, n)]->data, ldak,
                                1.0, near_D[twoD_2_oneD(k, k)]->data, ldak);
                    } else {
                        int rank = far_rank[index];
                        double *tmp1, *tmp2;
                        tmp1 = malloc(rank * rank * sizeof(*tmp1));
                        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, rank,
                                    CC->size[n], 1., (double *)far_V[index]->data, CC->size[n], (double *)far_V[index]->data, CC->size[n], 0.,
                                    tmp1, rank);
                        tmp2 = malloc(rank * RC->size[k] * sizeof(*tmp1));
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, rank, RC->size[k],
                                    rank, 1., tmp1, rank, (double *)far_U[index]->data, RC->size[k], 0.,
                                    tmp2, RC->size[k]);
                        free(tmp1);
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, RC->size[k], RC->size[k],
                                    rank, -1., (double *)far_U[index]->data, RC->size[k], tmp2, RC->size[k], 1.,
                                    near_D[twoD_2_oneD(k, k)]->data, ldam);
                        free(tmp2);
                    }
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dsyrk(
                            PlasmaUpper, PlasmaTrans,
                            tempkn, A.nb,
                            -1.0, A(n, k), ldan,
                            1.0, A(k, k), ldak);
                }
            }
        } else {
            if (n == k) {
                ss_cond_wait(k, k, 1);
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    if (k == 0) {
                        static_malloc_each(param, m, k);
                    }
                    static_gen_each(param, m, k);
                    CORE_dtrsm(
                            PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
                            tempmn, A.nb,
                            zone, near_D[twoD_2_oneD(k, k)]->data, ldak,
                            near_D[twoD_2_oneD(m, k)]->data, ldam);
                    static_compress_each(param, m, k);
//                    CORE_dtrsm(
//                            PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
//                            tempmn, A.nb,
//                            zone, A(k, k), ldak,
//                            A(m, k), ldam);
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dtrsm(
                            PlasmaLeft, PlasmaUpper, PlasmaTrans, PlasmaNonUnit,
                            A.nb, tempmn,
                            zone, A(k, k), ldak,
                            A(k, m), ldak);
                }
                ss_cond_set(m, k, 1);
            } else {
                ss_cond_wait(k, n, 1);
                ss_cond_wait(m, n, 1);
                /*
                 *  PlasmaLower
                 */
                if (uplo == PlasmaLower) {
                    if (near_D[twoD_2_oneD(m, k)] == NULL) {
                        static_malloc_each(param, m, k);
                    }
                    STARSH_int indexOfLHS = twoD_2_oneD(m, n), indexOfRHS = twoD_2_oneD(k, n);
                    if (far_rank[indexOfLHS] == -1 && far_rank[indexOfRHS] == -1) {
                        // all not low rank
                        CORE_dgemm(
                                PlasmaNoTrans, PlasmaTrans,
                                tempmn, A.nb, A.nb,
                                mzone, near_D[indexOfLHS]->data, ldam,
                                near_D[indexOfRHS]->data, ldak,
                                zone, near_D[twoD_2_oneD(m, k)]->data, ldam);
                    } else if (far_rank[indexOfLHS] != -1 && far_rank[indexOfRHS] == -1) {
                        // (m, n) is low rank, (k, n) is not
                        int rank = far_rank[indexOfLHS];
                        double *tmp = malloc(rank * CC->size[n] * sizeof(*tmp));
                        cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, rank, RC->size[k],
                                    CC->size[n], 1., (double *)far_V[indexOfLHS]->data, CC->size[n], near_D[indexOfRHS]->data, ldak, 0.,
                                    tmp, CC->size[n]);
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, RC->size[m], CC->size[n],
                                    rank, -1., (double *)far_U[indexOfLHS]->data, RC->size[m], tmp, CC->size[n], 1.,
                                    near_D[twoD_2_oneD(m, k)]->data, ldam);
                        free(tmp);
                    } else if (far_rank[indexOfLHS] == -1 && far_rank[indexOfRHS] != -1) {
                        // (m, n) is not low rank, (k, n) is
                        int rank = far_rank[indexOfRHS];
                        double *tmp = malloc(rank * RC->size[m] * sizeof(*tmp));
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, RC->size[m], rank,
                                    CC->size[n], 1., near_D[indexOfLHS]->data, RC->size[m], (double *)far_V[indexOfRHS]->data, CC->size[n], 0,
                                    tmp, RC->size[m]);
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, RC->size[m], CC->size[n],
                                    rank, -1., tmp, RC->size[m], (double *)far_U[indexOfRHS]->data, RC->size[k], 1.,
                                    near_D[twoD_2_oneD(m, k)]->data, ldam);
                        free(tmp);
                    } else {
                        int rankLHS = far_rank[indexOfLHS];
                        int rankRHS = far_rank[indexOfRHS];
                        double *tmp1, *tmp2;
                        tmp1 = malloc(rankLHS * rankRHS * sizeof(*tmp1));
                        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rankLHS, rankRHS,
                                    CC->size[n], 1., (double *)far_V[indexOfLHS]->data, CC->size[n], (double *)far_V[indexOfRHS]->data, CC->size[n], 0.,
                                    tmp1, rankLHS);
                        tmp2 = malloc(rankLHS * RC->size[k] * sizeof(*tmp1));
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, rankLHS, RC->size[k],
                                    rankRHS, 1., tmp1, rankRHS, (double *)far_U[indexOfRHS]->data, RC->size[k], 0.,
                                    tmp2, RC->size[k]);
                        free(tmp1);
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, RC->size[m], RC->size[k],
                                    rankLHS, -1., (double *)far_U[indexOfLHS]->data, RC->size[m], tmp2, RC->size[k], 1.,
                                    near_D[twoD_2_oneD(m, k)]->data, ldam);
                        free(tmp2);
                    }
                }
                    /*
                     *  PlasmaUpper
                     */
                else {
                    CORE_dgemm(
                            PlasmaTrans, PlasmaNoTrans,
                            A.nb, tempmn, A.nb,
                            mzone, A(n, k), ldan,
                            A(n, m), ldan,
                            zone, A(k, m), ldak);
                }
            }
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void drsdd_pdpotrf_testing_quark(PLASMA_enum uplo,
                                 PLASMA_sequence *sequence,
                                 PLASMA_request *request) {
    abort();
}

int OURS_dpotrf_Tile_Async(PLASMA_enum uplo, PLASMA_sequence *sequence, PLASMA_request *request) {
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    if (sequence == NULL) {
        plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL sequence");
        return PLASMA_ERR_UNALLOCATED;
    }
    if (request == NULL) {
        plasma_fatal_error("PLASMA_dpotrf_Tile_Async", "NULL request");
        return PLASMA_ERR_UNALLOCATED;
    }
    /* Check sequence status */
    if (sequence->status == PLASMA_SUCCESS)
        request->status = PLASMA_SUCCESS;
    else
        return plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);

    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("PLASMA_dpotrf_Tile_Async", "illegal value of uplo");
        return plasma_request_fail(sequence, request, -1);
    }
    /* Quick return */
/*
    if (max(N, 0) == 0)
        return PLASMA_SUCCESS;
*/
    // TODO (Jie): may support dynamic
    plasma->scheduling = 1;
    plasma_parallel_call_3(drsdd_pdpotrf_testing,
                           PLASMA_enum, uplo,
                           PLASMA_sequence*, sequence,
                           PLASMA_request*, request);

    return PLASMA_SUCCESS;
}

int OURS_dpotrf(PLASMA_enum uplo, struct Param *param) {
    double tol = param->tol;
    int maxrank = param->maxrank;
    STARSH_int nblocks_far = param->nblocks_far;
    STARSH_int *block_far = param->block_far;
    STARSH_kernel *kernel = param->kernel;
    STARSH_cluster *RC = param->RC, *CC = param->CC;
    Array **far_U = param->far_U, **far_V = param->far_V, **near_D, **near_D_final;
    int *far_rank = param->far_rank;
    const int oversample = param->oversample;
    const int onfly = param->onfly;

    void *RD = RC->data, *CD = CC->data;
    double drsdd_time;
    double kernel_time;
    int BAD_TILE;
    near_D = malloc(nblocks_far * sizeof(*near_D));
    memset(near_D, 0, nblocks_far * sizeof(*near_D));
    param->near_D = near_D;

    int NB;
    int status;
    plasma_context_t *plasma;
    PLASMA_sequence *sequence = NULL;
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

    plasma = plasma_context_self();
    plasma->aux = param;
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("PLASMA_dpotrf", "illegal value of uplo");
        return -1;
    }

    /* Set NT */
    NB = PLASMA_NB;

    plasma_sequence_create(plasma, &sequence);

    /* Call the tile interface */
    OURS_dpotrf_Tile_Async(uplo, sequence, &request);

    status = sequence->status;
    plasma_sequence_wait(plasma, sequence);
    plasma_sequence_destroy(plasma, sequence);

    STARSH_int near_D_counter = 0;
    for (STARSH_int i = 0; i < nblocks_far; ++i) {
        if (far_rank[i] == -1) {
            near_D_counter++;
        }
    }

    near_D_final = malloc(near_D_counter * sizeof(near_D_final));
    near_D_counter = 0;
    for (STARSH_int i = 0; i < nblocks_far; ++i) {
        if (far_rank[i] == -1) {
            near_D_final[near_D_counter] = near_D[i];
            near_D_counter++;
        }
    }
    free(near_D);
    param->near_D = near_D_final;

    return status;
}

int starsh_blrm__drsdd_potrf_omp(STARSH_blrm **matrix, STARSH_blrf *format,
                                 int maxrank, double tol, int onfly)
//! Approximate each tile by randomized SVD.
/*!
 * @param[out] matrix: Address of pointer to @ref STARSH_blrm object.
 * @param[in] format: Block low-rank format.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] tol: Relative error tolerance.
 * @param[in] onfly: Whether not to store dense blocks.
 * @ingroup blrm
 * */
{
    PLASMA_Init(omp_get_num_threads());
    STARSH_blrf *F = format;
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    STARSH_int nblocks_far = F->nblocks_far;
    STARSH_int nblocks_near = F->nblocks_near;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster;
    STARSH_cluster *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    STARSH_int new_nblocks_far = nblocks_far;
    STARSH_int new_nblocks_near = nblocks_near;
    STARSH_int *block_far = F->block_far;
    STARSH_int *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    struct starsh_block *block;
    int *far_rank = NULL;
    double *alloc_U = NULL, *alloc_V = NULL, *alloc_D = NULL;
    size_t offset_U = 0, offset_V = 0, offset_D = 0;
    STARSH_int bi, bj = 0;
    double drsdd_time = 0, kernel_time = 0;
    int BAD_TILE = 0;
    const int oversample = starsh_params.oversample;
    // Init buffers to store low-rank factors of far-field blocks if needed
    STARSH_MALLOC(block, nblocks_far + nblocks_near);
    if (nblocks_far > 0) {
        STARSH_MALLOC(far_U, nblocks_far);
        STARSH_MALLOC(far_V, nblocks_far);
        STARSH_MALLOC(far_rank, nblocks_far);
        size_t size_U = 0, size_V = 0;
        // Simple cycle over all far-field blocks
        for (bi = 0; bi < nblocks_far; bi++) {
            // Get indexes of corresponding block row and block column
            STARSH_int i = block_far[2 * bi];
            STARSH_int j = block_far[2 * bi + 1];
            // Get corresponding sizes and minimum of them
            size_U += RC->size[i];
            size_V += CC->size[j];
        }
        size_U *= maxrank;
        size_V *= maxrank;
        STARSH_MALLOC(alloc_U, size_U);
        STARSH_MALLOC(alloc_V, size_V);
        for (bi = 0; bi < nblocks_far; bi++) {
            // Get indexes of corresponding block row and block column
            STARSH_int i = block_far[2 * bi];
            STARSH_int j = block_far[2 * bi + 1];
            // Get corresponding sizes and minimum of them
            size_t nrows = RC->size[i], ncols = CC->size[j];
            int shape_U[] = {nrows, maxrank};
            int shape_V[] = {ncols, maxrank};
            double *U = alloc_U + offset_U, *V = alloc_V + offset_V;
            offset_U += nrows * maxrank;
            offset_V += ncols * maxrank;
            array_from_buffer(far_U + bi, 2, shape_U, 'd', 'F', U);
            array_from_buffer(far_V + bi, 2, shape_V, 'd', 'F', V);
        }
        offset_U = 0;
        offset_V = 0;
    }
    // Work variables
    int info;
    // Simple cycle over all far-field admissible blocks

    struct Param param;
    param.tol = tol;
    param.maxrank = maxrank;
    param.nblocks_far = nblocks_far;
    param.block_far = block_far;
    param.kernel = kernel;
    param.RC = RC;
    param.CC = CC;
    param.far_U = far_U;
    param.far_V = far_V;
    param.far_rank = far_rank;
    param.oversample = oversample;
    param.onfly = onfly;
    OURS_dpotrf(PlasmaLower, &param);
//    static_gen(&param);

    near_D = param.near_D;

    // Get number of false far-field blocks
    STARSH_int nblocks_false_far = 0;
    STARSH_int *false_far = NULL;
    for (bi = 0; bi < nblocks_far; bi++)
        if (far_rank[bi] == -1)
            nblocks_false_far++;
    if (nblocks_false_far > 0) {
        // IMPORTANT: `false_far` must to be in ascending order for later code
        // to work normally
        STARSH_MALLOC(false_far, nblocks_false_far);
        bj = 0;
        for (bi = 0; bi < nblocks_far; bi++) {
            block[bi].rank = far_rank[bi];
            if (far_rank[bi] == -1) {
                block[bi].idx_in_near = bj;
                false_far[bj++] = bi;
            } else {
                block[bi].idx_in_far = bi - bj;
            }
        }
    }
    // Update lists of far-field and near-field blocks using previously
    // generated list of false far-field blocks
    if (nblocks_false_far > 0) {
        // Update list of near-field blocks
        new_nblocks_near = nblocks_near + nblocks_false_far;
        STARSH_MALLOC(block_near, 2 * new_nblocks_near);
        // At first get all near-field blocks, assumed to be dense
//#pragma omp parallel for schedule(static)
        for (bi = 0; bi < 2 * nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        // Add false far-field blocks
//#pragma omp parallel for schedule(static)
        for (bi = 0; bi < nblocks_false_far; bi++) {
            STARSH_int bj = false_far[bi];
            block_near[2 * (bi + nblocks_near)] = F->block_far[2 * bj];
            block_near[2 * (bi + nblocks_near) + 1] = F->block_far[2 * bj + 1];
        }
        // Update list of far-field blocks
        new_nblocks_far = nblocks_far - nblocks_false_far;
        if (new_nblocks_far > 0) {
            STARSH_MALLOC(block_far, 2 * new_nblocks_far);
            bj = 0;
            for (bi = 0; bi < nblocks_far; bi++) {
                // `false_far` must be in ascending order for this to work
                if (bj < nblocks_false_far && false_far[bj] == bi) {
                    bj++;
                } else {
                    block_far[2 * (bi - bj)] = F->block_far[2 * bi];
                    block_far[2 * (bi - bj) + 1] = F->block_far[2 * bi + 1];
                }
            }
        }
        // Update format by creating new format
        STARSH_blrf *F2;
        info = starsh_blrf_new_from_coo(&F2, P, F->symm, RC, CC,
                                        new_nblocks_far, block_far, new_nblocks_near, block_near,
                                        F->type);
        // Swap internal data of formats and free unnecessary data
        STARSH_blrf tmp_blrf = *F;
        *F = *F2;
        *F2 = tmp_blrf;
        STARSH_WARNING("`F` was modified due to false far-field blocks");
        starsh_blrf_free(F2);
    }
    // Compute near-field blocks if needed
//    if (onfly == 0 && new_nblocks_near > 0) {
//        STARSH_MALLOC(near_D, new_nblocks_near);
//        size_t size_D = 0;
//        // Simple cycle over all near-field blocks
//        for (bi = 0; bi < new_nblocks_near; bi++) {
//            // Get indexes of corresponding block row and block column
//            STARSH_int i = block_near[2 * bi];
//            STARSH_int j = block_near[2 * bi + 1];
//            // Get corresponding sizes and minimum of them
//            size_t nrows = RC->size[i];
//            size_t ncols = CC->size[j];
//            // Update size_D
//            size_D += nrows * ncols;
//        }
//        STARSH_MALLOC(alloc_D, size_D);
//        // For each near-field block compute its elements
//        //#pragma omp parallel for schedule(dynamic,1)
//        for (bi = 0; bi < new_nblocks_near; bi++) {
//            // Get indexes of corresponding block row and block column
//            STARSH_int i = block_near[2 * bi];
//            STARSH_int j = block_near[2 * bi + 1];
//            // Get corresponding sizes and minimum of them
//            int nrows = RC->size[i];
//            int ncols = CC->size[j];
//            int shape[2] = {nrows, ncols};
//            double *D;
//#pragma omp critical
//            {
//                D = alloc_D + offset_D;
//                array_from_buffer(near_D + bi, 2, shape, 'd', 'F', D);
//                offset_D += near_D[bi]->size;
//            }
//            double time0 = omp_get_wtime();
//            kernel(nrows, ncols, RC->pivot + RC->start[i],
//                   CC->pivot + CC->start[j], RD, CD, D, nrows);
//            double time1 = omp_get_wtime();
//#pragma omp critical
//            kernel_time += time1 - time0;
//        }
//    }
    // Change sizes of far_rank, far_U and far_V if there were false
    // far-field blocks
    if (nblocks_false_far > 0 && new_nblocks_far > 0) {
        bj = 0;
        for (bi = 0; bi < nblocks_far; bi++) {
            if (far_rank[bi] == -1)
                bj++;
            else {
                int shape_U[2] = {far_U[bi]->shape[0], far_rank[bi]};
                int shape_V[2] = {far_V[bi]->shape[0], far_rank[bi]};
                far_V[bi - bj]->data = NULL;
                far_U[bi - bj]->data = NULL;
                array_free(far_V[bi - bj]);
                array_free(far_U[bi - bj]);
                array_from_buffer(far_U + bi - bj, 2, shape_U, 'd', 'F',
                                  far_U[bi]->data);
                array_from_buffer(far_V + bi - bj, 2, shape_V, 'd', 'F',
                                  far_V[bi]->data);
                far_rank[bi - bj] = far_rank[bi];
            }
        }
        for (STARSH_int i = new_nblocks_far; i < nblocks_far; ++i) {
            far_V[i]->data = NULL;
            far_U[i]->data = NULL;
            array_free(far_V[i]);
            array_free(far_U[i]);
        }
        STARSH_REALLOC(far_rank, new_nblocks_far);
        STARSH_REALLOC(far_U, new_nblocks_far);
        STARSH_REALLOC(far_V, new_nblocks_far);
        //STARSH_REALLOC(alloc_U, offset_U);
        //STARSH_REALLOC(alloc_V, offset_V);
    }
    // If all far-field blocks are false, then dealloc buffers
    if (new_nblocks_far == 0 && nblocks_far > 0) {
        block_far = NULL;
        free(far_rank);
        far_rank = NULL;
        for (int i = 0; i < nblocks_far; ++i) {
            far_U[i]->data = NULL;
            far_V[i]->data = NULL;
            array_free(far_U[i]);
            array_free(far_V[i]);
        }
        free(far_U);
        far_U = NULL;
        free(far_V);
        far_V = NULL;
        free(alloc_U);
        alloc_U = NULL;
        free(alloc_V);
        alloc_V = NULL;
    }
    // Dealloc list of false far-field blocks if it is not empty
    if (nblocks_false_far > 0)
        free(false_far);
    // Finish with creating instance of Block Low-Rank Matrix with given
    // buffers
    //STARSH_WARNING("DRSDD kernel total time: %e secs", drsdd_time);
    //STARSH_WARNING("MATRIX kernel total time: %e secs", kernel_time);
    PLASMA_Finalize();
    // TODO (Jie): remove this
    info =  starsh_blrm_new(matrix, F, far_rank, far_U, far_V, onfly, near_D,
                           alloc_U, alloc_V, (void *)1, '1');
    (*matrix)->alloc_D = NULL;
    (*matrix)->factorized = 1;
    (*matrix)->block = block;
    return info;
}

