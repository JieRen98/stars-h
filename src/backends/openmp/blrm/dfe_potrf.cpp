/*! @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/openmp/blrm/dfe.c
 * @version 0.3.1
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

extern "C" {
#include "common.h"
#include "starsh.h"
}

#include <Eigen/Core>
#include <iostream>

double starsh_blrm__dfe_potrf_omp(STARSH_blrm *matrix)
//! Approximation error in Frobenius norm of double precision matrix.
/*! Measure error of approximation of a dense matrix by block-wise low-rank
 * matrix.
 *
 * @param[in] matrix: Block-wise low-rank matrix.
 * @return Error of approximation.
 * @ingroup blrm
 * */
{
    STARSH_blrm *M = matrix;
    STARSH_blrf *F = M->format;
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    // Shortcuts to information about clusters
    STARSH_cluster *R = F->row_cluster;
    STARSH_cluster *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    // Number of far-field and near-field blocks
    STARSH_int nblocks_far = F->nblocks_far;
    STARSH_int nblocks_near = F->nblocks_near, bi;
    STARSH_int nblocks = nblocks_far+nblocks_near;
    // Shortcut to all U and V factors
    Array **U = M->far_U, **V = M->far_V;
    const struct starsh_block * const block = M->block;
    // Special constant for symmetric case
    double sqrt2 = sqrt(2.);
    // Temporary arrays to compute norms more precisely with dnrm2
    double block_norm[nblocks], far_block_diff[nblocks_far];
    double *far_block_norm = block_norm;
    char symm = F->symm;
    int info = 0;
    auto infinityNorm = [](const auto &m) -> double { return m.rowwise().sum().array().abs().maxCoeff(); };

    if (M->factorized) {
        if (symm != 'S') {
            abort();
        }

        for (int k = 0; k < R->nblocks; ++k) {
            auto twoD_2_oneD = [](STARSH_int i, STARSH_int j) {
                return j + (1 + i) * i / 2;
            };
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> m{
                    (double *) M->near_D[block[twoD_2_oneD(k, k)].idx_in_near]->data, R->size[k], C->size[k]};
            m.triangularView<Eigen::StrictlyUpper>().setZero();
        }

        double FNorm = 0;

        for(STARSH_int i = 0; i < R->nblocks; i++) {
            Eigen::Matrix<double, Eigen::Dynamic, 1> blockFNorm{R->size[i]};
            blockFNorm.setZero();
            for(STARSH_int j = 0; j <= i; j++) {
//            if(info != 0)
//                continue;
                // Get indexes and sizes of corresponding block row and column
                int nrows = R->size[i];
                int ncols = C->size[j];
                double *D;
                // Allocate temporary array and fill it with elements of a block
                STARSH_PMALLOC(D, (size_t) nrows * (size_t) ncols, info);
                memset(D, 0, (size_t) nrows * (size_t) ncols * sizeof(*D));
                kernel(nrows, ncols, R->pivot + R->start[i], C->pivot + C->start[j],
                       RD, CD, D, nrows);

                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> ref{D, nrows, ncols};
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> computed{nrows, ncols};
                computed.setZero();

                auto twoD_2_oneD = [](STARSH_int i, STARSH_int j) {
                    return j + (1 + i) * i / 2;
                };

                for (int k = 0; k <= std::min(i, j); ++k) {
                    int indexLHS = twoD_2_oneD(i, k);
                    const int rankLHS = block[indexLHS].rank;
                    int indexRHS = twoD_2_oneD(j, k);
                    const int rankRHS = block[indexRHS].rank;
                    if (rankLHS == -1 && rankRHS == -1) {
                        // all near
                        auto nearIdxLHS = block[indexLHS].idx_in_near;
                        auto nearIdxRHS = block[indexRHS].idx_in_near;
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> left{
                                (double *) M->near_D[nearIdxLHS]->data, R->size[i], C->size[k]};
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> right{
                                (double *) M->near_D[nearIdxRHS]->data, R->size[j], C->size[k]};
                        computed += left * right.transpose();
                    } else if (rankLHS != -1 && rankRHS == -1) {
                        // LHS far, RHS near
                        auto farIdxLHS = block[indexLHS].idx_in_far;
                        auto nearIdxRHS = block[indexRHS].idx_in_near;
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> U{
                                (double *) M->far_U[farIdxLHS]->data, R->size[i], rankLHS};
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> V{
                                (double *) M->far_V[farIdxLHS]->data, C->size[k], rankLHS};

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> right{
                                (double *) M->near_D[nearIdxRHS]->data, R->size[j], C->size[k]};
                        computed += U * V.transpose() * right.transpose();
                    } else if (rankLHS == -1 && rankRHS != -1) {
                        // LHS near, RHS far
                        auto nearIdxLHS = block[indexLHS].idx_in_near;
                        auto farIdxRHS = block[indexRHS].idx_in_far;
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> left{
                                (double *) M->near_D[nearIdxLHS]->data, R->size[i], C->size[k]};
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> U{
                                (double *) M->far_U[farIdxRHS]->data, R->size[j], rankRHS};
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> V{
                                (double *) M->far_V[farIdxRHS]->data, C->size[k], rankRHS};
                        computed += left * (U * V.transpose()).transpose();
                    } else {
                        auto farIdxLHS = block[indexLHS].idx_in_far;
                        auto farIdxRHS = block[indexRHS].idx_in_far;

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> UL{
                                (double *) M->far_U[farIdxLHS]->data, R->size[i], rankLHS};
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> VL{
                                (double *) M->far_V[farIdxLHS]->data, C->size[k], rankLHS};

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> UR{
                                (double *) M->far_U[farIdxRHS]->data, R->size[j], rankLHS};
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> VR{
                                (double *) M->far_V[farIdxRHS]->data, C->size[k], rankLHS};

                        computed += UL * VL.transpose() * (UR * VR.transpose()).transpose();

                    }
                }

//                if (i == 1 && j == 1) {
//                    std::cout << ref << std::endl;
//                    printf("================\n");
//                    std::cout << computed << std::endl;
//                }

//                printf("(%zd, %zd) origin:\n", i, j);
//                std::cout << ref << std::endl;
//                printf("(%zd, %zd) compute:\n", i, j);
//                std::cout << computed << std::endl;
//                printf("(%zd, %zd) factor:\n", i, j);
//                auto b = block[twoD_2_oneD(i, j)];
//                if (b.rank == -1) {
//                    std::cout << Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>{
//                            (double *) M->near_D[b.idx_in_near]->data, R->size[i], C->size[j]} << std::endl;
//                } else {
//                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> U{
//                            (double *) M->far_U[b.idx_in_far]->data, R->size[i], b.rank}, V{(double *) M->far_V[b.idx_in_far]->data, C->size[j], b.rank};
//                    std::cout << "U: " << std::endl << U << std::endl;
//                    std::cout << "V: " << std::endl << V << std::endl;
//                    std::cout << "UVT" << std::endl << U * V.transpose() << std::endl;
//                }

//                near_block_norm[twoD_2_oneD(i, j)] = infinityNorm(ref - computed);
                blockFNorm += (ref - computed).array().abs().rowwise().sum().matrix();

//                printf("(%zd, %zd): %e\n", i, j, near_block_norm[twoD_2_oneD(i, j)]);
                // Free temporary buffer
                free(D);
//            near_block_norm[bi] = cblas_dnrm2(ncols, D_norm, 1);
//            if(i != j && symm == 'S')
//                // Multiply by square root of 2 ub symmetric case
//                near_block_norm[bi] *= sqrt2;
            }
            FNorm = std::max(FNorm, blockFNorm.maxCoeff());
        }
        printf("Inf norm %e\n", FNorm);
        return 0;
    }
}
