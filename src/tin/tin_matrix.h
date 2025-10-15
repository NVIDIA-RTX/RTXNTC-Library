/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include "tin_mma.h"

namespace tin {

    namespace host {
        template <class T, bool ROW_MAJOR>
        class Matrix;
    }

    /*
    //  Base Matrix class that tiles a fixed size MMAMatrix
    */
    template <class T, uint32_t N_COLS>
    class Vector;

    enum class ReducerUpdateMode {
        STORE,
        ATOMIC_ADD
    };

    template < uint32_t NUM_VECTORS, uint32_t N_COLS0, uint32_t N_COLS1, ReducerUpdateMode UPDATE_MODE >
    class OuterProductReducer;

    template <class T, uint32_t N_ROWS, uint32_t N_COLS, bool ROW_MAJOR>
    class MatrixBase {
    public:
        using underlying_type = T;

        static const uint32_t N_INNER = ROW_MAJOR ? N_COLS : N_ROWS;
        static const uint32_t N_OUTER = ROW_MAJOR ? N_ROWS : N_COLS;

#if defined(__CUDACC__)

        //  Element addresses for a row-major matrix
        //  SS: Substride
        //  OS: Outer stride
        //  IS: Inner stride
        //  |0  , 1    , 2    , ... , SS-1      | IS    , 1 + IS    , 2+IS    , ... , SS-1+IS    | 2IS,
        //  |OS , 1+OS , 2+OS , ... , SS-1 +  OS| IS+OS , 1 + IS+OS , 2+IS+OS , ... , IS-1+IS+OS | 2IS+OS,
        //  |2OS, 1+2OS, 2+2OS, ... , SS-1 + 2OS| IS+2OS, 1 + IS+2OS, 2+IS+2OS, ... , IS-1+IS+2OS| 2IS+2OS,

        template <uint32_t SUBSTRIDE = N_INNER>
        TIN_DEVICE void load(
            const T* addr,
            uint32_t outer_stride = SUBSTRIDE,
            uint32_t inner_stide  = SUBSTRIDE * N_OUTER) {

            const uint32_t* ip_packed = (const uint32_t*)addr;

            uint32_t lane_id = _lane_id();
            uint2 target_subreg_rc = MMAReg::lin_to_rc_packed(lane_id);

TIN_UNROLL
            for (uint32_t mma_row = 0; mma_row < MMARows; mma_row++) {
TIN_UNROLL
                for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++) {
                    uint32_t mma_idx = mma_rc_to_idx(mma_row, mma_col);
TIN_UNROLL
                    for (uint32_t reg_row = 0; reg_row < MMAMat::RegRows; reg_row++) {
TIN_UNROLL
                        for (uint32_t reg_col = 0; reg_col < MMAMat::RegCols; reg_col++) {

                            uint32_t row_packed = reg_row * MMAReg::RowsPacked + mma_row * MMAMat::RowsPacked + target_subreg_rc.x;
                            uint32_t col_packed = reg_col * MMAReg::ColsPacked + mma_col * MMAMat::ColsPacked + target_subreg_rc.y;

                            uint32_t idx = rc_to_addr_packed<SUBSTRIDE>(
                                            row_packed, col_packed,
                                            outer_stride, inner_stide);

                            uint32_t reg_idx = MMAMat::reg_rc_to_idx(reg_row, reg_col);
                            m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg = ip_packed[idx];
                        }
                    }
                }
            }
        }

        template <uint32_t SUBSTRIDE = N_INNER>
        TIN_DEVICE void store(
            T* addr,
            uint32_t outer_stride = SUBSTRIDE,
            uint32_t inner_stide  = SUBSTRIDE * N_OUTER) const {

            uint32_t* ip_packed = (uint32_t*)addr;

            uint32_t lane_id = _lane_id();
            uint2 target_subreg_rc = MMAReg::lin_to_rc_packed(lane_id);

TIN_UNROLL
            for (uint32_t mma_row = 0; mma_row < MMARows; mma_row++) {
TIN_UNROLL
                for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++) {
                    uint32_t mma_idx = mma_rc_to_idx(mma_row, mma_col);
TIN_UNROLL
                    for (uint32_t reg_row = 0; reg_row < MMAMat::RegRows; reg_row++) {
TIN_UNROLL
                        for (uint32_t reg_col = 0; reg_col < MMAMat::RegCols; reg_col++) {

                            uint32_t row_packed = reg_row * MMAReg::RowsPacked + mma_row * MMAMat::RowsPacked + target_subreg_rc.x;
                            uint32_t col_packed = reg_col * MMAReg::ColsPacked + mma_col * MMAMat::ColsPacked + target_subreg_rc.y;

                            uint32_t idx = rc_to_addr_packed<SUBSTRIDE>(
                                            row_packed, col_packed,
                                            outer_stride, inner_stide);

                            uint32_t reg_idx = MMAMat::reg_rc_to_idx(reg_row, reg_col);
                            ip_packed[idx] = m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg;
                        }
                    }
                }
            }
        }

        TIN_DEVICE void load_native(const T* addr) {
            uint32_t lane_id = _lane_id();
            auto mat = (const MMAMat*)(addr);

TIN_UNROLL
            for (uint32_t mma_idx = 0; mma_idx < NumMMAs; mma_idx++) {
                m_mma_mat[mma_idx] = mat[mma_idx * WarpSize + lane_id];
            }
        }

        TIN_DEVICE void store_native(T* addr) {
            uint32_t lane_id = _lane_id();
            auto mat = (MMAMat*)(addr);

TIN_UNROLL
            for (uint32_t mma_idx = 0; mma_idx < NumMMAs; mma_idx++) {
                mat[mma_idx * WarpSize + lane_id] = m_mma_mat[mma_idx];
            }
        }

        TIN_DEVICE MatrixBase<T, N_COLS, N_ROWS, !ROW_MAJOR> transpose() const {
            MatrixBase<T, N_COLS, N_ROWS, !ROW_MAJOR> r;

TIN_UNROLL
            for (uint32_t i = 0; i < MMACols; i++) {
TIN_UNROLL
                for (uint32_t j = 0; j < MMARows; j++) {
                    uint32_t src_mma_idx = mma_rc_to_idx(j, i);
                    uint32_t dst_mma_idx = r.mma_rc_to_idx(i, j);
                    r.m_mma_mat[dst_mma_idx] = m_mma_mat[src_mma_idx].transpose();
                }
            }

            return r;
        }

        TIN_DEVICE void clear() {
TIN_UNROLL
            for (uint32_t i = 0; i < MMACols; i++) {
TIN_UNROLL
                for (uint32_t j = 0; j < MMARows; j++) {
                    uint32_t mma_idx = mma_rc_to_idx(j, i);
                    m_mma_mat[mma_idx].clear();
                }
            }
        }

        template <class T1, uint32_t N_ROWS0, uint32_t N_COLS0, bool ROW_MAJOR0>
        friend MatrixBase<T1, N_ROWS0, N_COLS0, ROW_MAJOR0>
        TIN_DEVICE operator+(
                const MatrixBase<T1, N_ROWS0, N_COLS0, ROW_MAJOR0>& a,
                const MatrixBase<T1, N_ROWS0, N_COLS0, ROW_MAJOR0>& b);
        template <class T1, uint32_t N_ROWS0, uint32_t N_COLS0, uint32_t N_COLS1>
        friend MatrixBase<T1, N_ROWS0, N_COLS1, true>
        TIN_DEVICE mad(const MatrixBase<T1, N_ROWS0, N_COLS0, true >& a,
            const MatrixBase<T1, N_COLS0, N_COLS1, false>& b,
            const MatrixBase<T1, N_ROWS0, N_COLS1, true>& c);

        template <uint32_t N_ROWS0, uint32_t N_COLS0, bool ROW_MAJOR0>
        friend MatrixBase<half, N_ROWS0, N_COLS0, !ROW_MAJOR0>
        TIN_DEVICE change_major_axis(
            const MatrixBase<half, N_ROWS0, N_COLS0, ROW_MAJOR0>& x);

        template <uint32_t NUM_VECTORS, uint32_t N_COLS0, uint32_t N_COLS1, ReducerUpdateMode UPDATE_MODE>
        friend class OuterProductReducer;

        template <typename F, class T1, class... T2>
        friend TIN_DEVICE T1 map(F f, T1 v1, T2... v2);

        template <typename F, class T1, class... T2>
        friend TIN_DEVICE T1 map_mma(F f, T1 v1, T2... v2);

        template <typename TAct, class T1, class... T2>
        friend TIN_DEVICE T1 act_forward(TAct act, T1 v1, T2... v2);

        template <typename TAct, class T1, class... T2>
        friend TIN_DEVICE T1 act_backward(TAct act, T1 v1, T2... v2);
#endif

        template<class T1, uint32_t N_ROWS0, uint32_t N_COLS0, bool ROW_MAJOR0>
        friend class MatrixBase;

        template <class T1, bool ROW_MAJOR1>
        friend class tin::host::Matrix;

    protected:

#if defined(__CUDACC__)
        TIN_DEVICE T get_item(uint32_t index) const {

            const uint32_t NumPacked = num_packed<T>();
            uint32_t idx_packed = index / NumPacked;
            uint32_t reg_idx = idx_packed % MMAMat::NumRegs;
            uint32_t mma_idx = idx_packed / MMAMat::NumRegs;

            auto item = (const T*)(&(m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg));
            return item[index % NumPacked];
        }

        TIN_DEVICE void set_item(uint32_t index, T x) {
            const uint32_t NumPacked = num_packed<T>();
            uint32_t idx_packed = index / NumPacked;
            uint32_t reg_idx = idx_packed % MMAMat::NumRegs;
            uint32_t mma_idx = idx_packed / MMAMat::NumRegs;

            auto item = (T*)(&(m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg));
            item[index % NumPacked] = x;
        }
#endif

        constexpr TIN_DEVICE_HOST uint32_t items_per_thread() {
            return (N_ROWS * N_COLS) / WarpSize;
        }

        template <uint32_t SUBSTRIDE>
        static TIN_DEVICE_HOST uint32_t rc_to_addr_packed(
            uint32_t row,
            uint32_t col,
            uint32_t outer_stride,
            uint32_t inner_stide) {

            const uint32_t Substride = SUBSTRIDE / num_packed<T>();
            const uint32_t inner_stride_packed = inner_stide / num_packed<T>();
            const uint32_t outer_stride_packed = outer_stride / num_packed<T>();

            if (ROW_MAJOR) {
                uint32_t ck = col % Substride;
                uint32_t c1 = col / Substride;
                return c1 * inner_stride_packed + row * outer_stride_packed + ck;
            }
            else {
                uint32_t rk = row % Substride;
                uint32_t r1 = row / Substride;
                return r1 * inner_stride_packed + col * outer_stride_packed + rk;
            }
        }

        static TIN_DEVICE_HOST uint32_t get_mma_row(uint32_t row_packed) { return row_packed / MMAMat::RowsPacked; }
        static TIN_DEVICE_HOST uint32_t get_mma_col(uint32_t col_packed) { return col_packed / MMAMat::ColsPacked; }

        static TIN_DEVICE_HOST uint32_t get_submma_row(uint32_t row_packed) { return row_packed % MMAMat::RowsPacked; }
        static TIN_DEVICE      uint32_t get_submma_col(uint32_t col_packed) { return col_packed % MMAMat::ColsPacked; }

        static TIN_DEVICE_HOST uint32_t mma_rc_to_idx(uint32_t mma_row, uint32_t mma_col) {
            return mma_col * MMARows + mma_row;
        }

        using MMAMat = MMAMatrix<T, ROW_MAJOR>;
        using MMAReg = typename MMAMat::MMAReg;

        static const uint32_t MMARows = N_ROWS / MMAMat::Rows;
        static const uint32_t MMACols = N_COLS / MMAMat::Cols;
        static const uint32_t NumMMAs = MMARows * MMACols;

        static const uint32_t RowsPacked = MMARows * MMAMat::RowsPacked;
        static const uint32_t ColsPacked = MMACols * MMAMat::ColsPacked;

        MMAMat m_mma_mat[NumMMAs];
    };

    // Matrix aliases
    template <class T, uint32_t N_ROWS, uint32_t N_COLS> using MatrixB = MatrixBase<T, N_ROWS, N_COLS, false>;
    template <class T, uint32_t N_ROWS, uint32_t N_COLS> using MatrixA = MatrixBase<T, N_ROWS, N_COLS, true >;
    template <class T, uint32_t N_ROWS, uint32_t N_COLS> using Matrix  = MatrixB<T, N_ROWS, N_COLS>;

    template <uint32_t N_ROWS, uint32_t N_COLS> using HMatrixB = MatrixB<half, N_ROWS, N_COLS>;
    template <uint32_t N_ROWS, uint32_t N_COLS> using HMatrixA = MatrixA<half, N_ROWS, N_COLS>;
    template <uint32_t N_ROWS, uint32_t N_COLS> using HMatrix  = HMatrixB<N_ROWS, N_COLS>;


    template <class T, uint32_t N_COLS> struct Array;

#if defined(__CUDACC__)

#define OPTIMIZED_SHUFFLE

    /*
    // SIMT friendly warp-cooperative Vector class
    */
    template <class T, uint32_t N_COLS>
    class Vector : public MatrixA<T, WarpSize, N_COLS> {
    public:
        template <class T1, uint32_t N_COLS1> friend class Array;

        TIN_DEVICE Vector() {};

        TIN_DEVICE Vector(const MatrixA<T, WarpSize, N_COLS>& m) {
            *this = *((Vector<T, N_COLS> *)(&m));
        };

        TIN_DEVICE Vector(const T* input) {
            from_array<N_COLS>(input);
        }

        // Copies a user array into a tensor in registers.
        template <uint32_t N>
        TIN_DEVICE void from_array(const T* input) {

            auto ip_packed_arr = (const uint32_t*)input;
            const uint32_t lane_id = _lane_id();

#ifdef OPTIMIZED_SHUFFLE

TIN_UNROLL
            for (uint32_t col_packed = 0; col_packed < this->ColsPacked; col_packed += MMAReg::ColsPacked) {
                uint32_t regsi[MMAReg::ColsPacked];

TIN_UNROLL
                for (uint32_t i = 0; i < MMAReg::ColsPacked; i++) {

                    if ((col_packed + i) < N / num_packed<T>())
                        regsi[i] = ip_packed_arr[col_packed + i];
                    else
                        regsi[i] = 0;

TIN_UNROLL
                    for (uint32_t j = 1; j < MMAReg::ColsPacked; j++) {
                        uint32_t val_packed = ip_packed_arr[col_packed + (i + j) % MMAReg::ColsPacked];
                        if (lane_id / MMAReg::RowsPacked == j) {
                            regsi[i] = val_packed;
                        }
                    }
                }

TIN_UNROLL
                for (uint32_t i = 0; i < MMAReg::ColsPacked; i++) {
                    uint32_t shfl_idx = ((lane_id + (MMAReg::ColsPacked - i)) % MMAReg::ColsPacked) * MMAReg::RowsPacked + lane_id / MMAReg::ColsPacked;
                    regsi[i] = __shfl_sync(0xFFFFFFFF, regsi[i], shfl_idx);
                }

                uint32_t mma_col = this->get_mma_col(col_packed);
                uint32_t submma_col = this->get_submma_col(col_packed);
                uint32_t r_col = MMAMat::get_reg_col(submma_col);

TIN_UNROLL
                for (uint32_t i = 0; i < MMAReg::ColsPacked; i++) {
                    uint32_t r_row   = i % MMAMat::RegRows;
                    uint32_t mma_row = i / MMAMat::RegRows;

                    uint32_t mma_idx = this->mma_rc_to_idx(mma_row, mma_col);
                    uint32_t reg_idx = MMAMat::reg_rc_to_idx(r_row, r_col);
                    this->m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg = regsi[0];

TIN_UNROLL
                    for (uint32_t j = 1; j < MMAReg::ColsPacked; j++) {
                        if (lane_id % MMAReg::ColsPacked == (i + j) % MMAReg::ColsPacked) {
                            this->m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg = regsi[j];
                        }
                    }
                }

            }
#else
            const uint32_t mma_rows = get_mma_row(WarpSize);
            const uint2 target_subreg_rc = MMAReg::lin_to_rc_packed(lane_id);

TIN_UNROLL
            for (uint32_t col = 0; col < ColsPacked; col++) {

                uint32_t ip_packed = ip_packed_arr[col];
                uint32_t mma_col = this->get_mma_col(col);
                uint32_t submma_col = this->get_submma_col(col);

                uint32_t reg_col = MMAMat::get_reg_col(submma_col);
                uint32_t subreg_col = MMAMat::get_subreg_col(submma_col);

TIN_UNROLL
                for (uint32_t mma_row = 0; mma_row < mma_rows; mma_row++) {
                    uint32_t mma_idx = this->mma_rc_to_idx(mma_row, mma_col);
TIN_UNROLL
                    for (uint32_t reg_row = 0; reg_row < MMAMat::RegRows; reg_row++) {

                        uint32_t target_row = reg_row * MMAReg::RowsPacked + mma_row * MMAMat::RowsPacked + target_subreg_rc.x;
                        uint32_t reg_idx = MMAMat::reg_rc_to_idx(reg_row, reg_col);

                        uint32_t shfl_val = __shfl_sync(0xFFFFFFFF, ip_packed, target_row);
                        if (subreg_col == target_subreg_rc.y) {
                            this->m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg = shfl_val;
                        }
                    }
                }
            }
#endif
        };

        TIN_DEVICE Vector(const Array<T, N_COLS>& input) : Vector{&input[0]} {}

        template <uint32_t N>
        TIN_DEVICE void to_array(T* output) const {
            uint32_t lane_id = _lane_id();

#ifdef OPTIMIZED_SHUFFLE

TIN_UNROLL
            for (uint32_t col_packed = 0; col_packed < this->ColsPacked; col_packed += MMAReg::ColsPacked) {
                uint32_t regsi[MMAReg::ColsPacked];

                uint32_t mma_col = this->get_mma_col(col_packed);
                uint32_t submma_col = this->get_submma_col(col_packed);
                uint32_t r_col = MMAMat::get_reg_col(submma_col);
                uint32_t reg_idx = MMAMat::reg_rc_to_idx(0, r_col);
                uint32_t mma_idx = this->mma_rc_to_idx(0, mma_col);

TIN_UNROLL
                for (uint32_t i = 0; i < MMAReg::ColsPacked; i++) {
                    regsi[i] = this->m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg;
TIN_UNROLL
                    for (uint32_t j = 1; j < MMAReg::ColsPacked; j++) {
                        uint32_t r_row   = j % MMAMat::RegRows;
                        uint32_t mma_row = j / MMAMat::RegRows;

                        uint32_t reg_idx = MMAMat::reg_rc_to_idx(r_row, r_col);
                        uint32_t mma_idx = this->mma_rc_to_idx(mma_row, mma_col);

                        if (lane_id % MMAReg::ColsPacked == (i + j) % MMAReg::ColsPacked) {
                            regsi[i] = this->m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg;
                        }
                    }
                }

TIN_UNROLL
                for (uint32_t i = 0; i < MMAReg::ColsPacked; i++) {
                    uint32_t shfl_idx = (lane_id % MMAReg::RowsPacked) * MMAReg::ColsPacked + (lane_id / MMAReg::RowsPacked + i) % MMAReg::ColsPacked;
                    regsi[i] = __shfl_sync(0xFFFFFFFF, regsi[i], shfl_idx);
                }

TIN_UNROLL
                for (uint32_t i = 0; i < MMAReg::ColsPacked; i++) {
                    uint32_t op = regsi[0];
TIN_UNROLL
                    for (uint32_t j = 1; j < MMAReg::ColsPacked; j++) {
                        if (lane_id / MMAReg::RowsPacked == (MMAReg::ColsPacked - j + i) % MMAReg::ColsPacked) {
                            op = regsi[j];
                        }
                    }

TIN_UNROLL
                    for (uint32_t j = 0; j < num_packed<T>(); j++) {
                        uint32_t idx = (col_packed + i) * num_packed<T>() + j;
                        if (idx < N) {
                            output[idx] = ((const T*)&op)[j];
                        }
                    }
                }
            }
#else

            uint32_t mma_rows = this->get_mma_row(WarpSize);

            uint32_t target_mma_row = this->get_mma_row(lane_id);
            uint32_t target_submma_row = this->get_submma_row(lane_id);

            uint32_t target_reg_row = MMAMat::get_reg_row(target_submma_row);
            uint32_t target_subreg_row = MMAMat::get_subreg_row(target_submma_row);

            static const uint32_t ColsPacked = (N_COLS + num_packed<T>() - 1) / num_packed<T>();

TIN_UNROLL
            for (uint32_t col = 0; col < ColsPacked; col ++) {
                uint32_t mma_col = this->get_mma_col(col);
                uint32_t submma_col = this->get_submma_col(col);

                uint32_t reg_col    = MMAMat::get_reg_col(submma_col);
                uint32_t subreg_col = MMAMat::get_subreg_col(submma_col);
                uint32_t target_idx = MMAReg::rc_to_lin_packed(target_subreg_row, subreg_col);

TIN_UNROLL
                for (uint32_t mma_row = 0; mma_row < mma_rows; mma_row++) {
                    uint32_t mma_idx = this->mma_rc_to_idx(mma_row, mma_col);
TIN_UNROLL
                    for (uint32_t reg_row = 0; reg_row < MMAMat::RegRows; reg_row++) {
                        uint32_t reg_idx = MMAMat::reg_rc_to_idx(reg_row, reg_col);
                        uint32_t reg_val = this->m_mma_mat[mma_idx].m_mma_reg[reg_idx].m_reg;

                        uint32_t shfl_val = __shfl_sync(0xFFFFFFFF, reg_val, target_idx);
                        if (target_reg_row == reg_row && target_mma_row == mma_row) {
TIN_UNROLL
                            for (uint32_t i = 0; i < num_packed<T>(); ++i) {
                                uint32_t idx = col * num_packed<T>() + i;
                                if (idx < N) {
                                    output[idx] = ((const T*)&shfl_val)[i];
                                }
                            }
                        }
                    }
                }
            }
#endif
        }

        template <uint32_t N_COLS0>
        friend void TIN_DEVICE reduce_sum(const Vector<half, N_COLS0>& a, half* dest);

    private:
        using Mat = MatrixA<T, WarpSize, N_COLS>;
        using MMAMat = typename Mat::MMAMat;
        using MMAReg = typename MMAMat::MMAReg;
    };

    /*
    // SIMT Array class for initializing warp-cooperative Vector
    */
    template <class T, uint32_t N_COLS>
    class alignas(32) Array {
    public:

        // Initializing all elements to avoid partial-warp undefs
        __device__ Array() {
        }

        __device__ Array(T init_val) {
TIN_UNROLL
            for (int i = 0; i < N_COLS; i++) {
                (*this)[i] = init_val;
            }
        }

        // Copies tensor into a user array
        TIN_DEVICE Array(const Vector<T, N_COLS>& v) {
            v.template to_array<N_COLS>((T*)(&m_data));
        };

        TIN_DEVICE Array(const T* __restrict__ arr) {
TIN_UNROLL
            for (int i = 0; i < N_COLS; i++) {
                (*this)[i] = arr[i];
            }
        }

        TIN_DEVICE T& operator [](uint32_t index) {
            return ((T*)(&m_data))[index];
        }

        TIN_DEVICE const T& operator [](uint32_t index) const {
            return ((T*)(&m_data))[index];
        }

        TIN_DEVICE void set_packed_element(PackedType<T> x, uint32_t packed_index) {
            m_data[packed_index] = pack<T>(x);
        }

        TIN_DEVICE PackedType<T> get_packed_element(uint32_t packed_index) const {
            return unpack<T>(m_data[packed_index]);
        }

    protected:
        static const uint32_t ColsPacked = N_COLS / num_packed<T>();
        uint32_t m_data[ColsPacked];
    };

    template <uint32_t N_COLS> using HVector = Vector<half, N_COLS>;
    template <uint32_t N_COLS> using HArray  = Array<half, N_COLS>;

    template <uint32_t N_COLS>
    TIN_DEVICE HArray<N_COLS> operator+(const HArray<N_COLS>& a, const HArray<N_COLS>& b) {
        HArray<N_COLS> r;

        const half2* a_data = (half2*)&(a[0]);
        const half2* b_data = (half2*)&(b[0]);
        half2* r_data = (half2*)&(r[0]);

TIN_UNROLL
        for (uint32_t i = 0; i < N_COLS/2; i++) {
            r_data[i] = a_data[i] + b_data[i];
        }

        return r;
    };


    // Matrix and Vector Operations
    template <class T, uint32_t N_ROWS0, uint32_t N_COLS0, uint32_t N_COLS1>
    TIN_DEVICE MatrixA<T, N_ROWS0, N_COLS1>
    mad(const MatrixA<T, N_ROWS0, N_COLS0>& a,
        const MatrixB<T, N_COLS0, N_COLS1>& b,
        const MatrixA<T, N_ROWS0, N_COLS1>& c) {

        MatrixA<T, N_ROWS0, N_COLS1> r = c;

        const uint32_t MMARows = r.MMARows;
        const uint32_t MMACols = r.MMACols;
        const uint32_t K = a.MMACols;

TIN_UNROLL
        for (uint32_t mma_row = 0; mma_row < MMARows; mma_row++) {
TIN_UNROLL
            for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++) {
                uint32_t idx_r = r.mma_rc_to_idx(mma_row, mma_col);
TIN_UNROLL
                for (uint32_t k = 0; k < K; k++) {
                    uint32_t idx_a = a.mma_rc_to_idx(mma_row, k);
                    uint32_t idx_b = b.mma_rc_to_idx(k, mma_col);
                    r.m_mma_mat[idx_r] = mad(a.m_mma_mat[idx_a], b.m_mma_mat[idx_b], r.m_mma_mat[idx_r]);
                }
            }
        }

        return r;
    };

    template <class T, uint32_t N_ROWS0, uint32_t N_COLS0, uint32_t N_COLS1>
    TIN_DEVICE MatrixA<T, N_ROWS0, N_COLS1>
    mul(
        const MatrixA<T, N_ROWS0, N_COLS0>& a,
        const MatrixB<T, N_COLS0, N_COLS1>& b) {

        MatrixA<T, N_ROWS0, N_COLS1> c;
        c.clear();
        return mad(a, b, c);
    };

    template <class T, uint32_t N_ROWS, uint32_t N_COLS, bool ROW_MAJOR>
    TIN_DEVICE MatrixBase<T, N_ROWS, N_COLS, ROW_MAJOR>
    operator+(
               const MatrixBase<T, N_ROWS, N_COLS, ROW_MAJOR>& a,
               const MatrixBase<T, N_ROWS, N_COLS, ROW_MAJOR>& b) {

        MatrixBase<T, N_ROWS, N_COLS, ROW_MAJOR> r;
        const uint32_t MMARows = r.MMARows;
        const uint32_t MMACols = r.MMACols;

TIN_UNROLL
        for (uint32_t i = 0; i < MMARows; i++) {
TIN_UNROLL
            for (uint32_t j = 0; j < MMACols; j++) {
                uint32_t idx = r.mma_rc_to_idx(i, j);
                r.m_mma_mat[idx] = a.m_mma_mat[idx] + b.m_mma_mat[idx];
            }
        }

        return r;
    };

    template <uint32_t N_ROWS, uint32_t N_COLS, bool ROW_MAJOR>
    TIN_DEVICE MatrixBase<half, N_ROWS, N_COLS, !ROW_MAJOR>
    change_major_axis(
        const MatrixBase<half, N_ROWS, N_COLS, ROW_MAJOR>& x) {

        MatrixBase<half, N_ROWS, N_COLS, !ROW_MAJOR > r;
        const uint32_t MMARows = x.MMARows;
        const uint32_t MMACols = x.MMACols;

TIN_UNROLL
        for (uint32_t i = 0; i < MMACols; i++) {
TIN_UNROLL
            for (uint32_t j = 0; j < MMARows; j++) {
                uint32_t mma_idx = x.mma_rc_to_idx(j, i);
                r.m_mma_mat[mma_idx] = change_major_axis(x.m_mma_mat[mma_idx]);
            }
        }
        return r;
    };


    // Reduce elements across warp
    template <uint32_t N_COLS>
    TIN_DEVICE void reduce_sum(const HVector<N_COLS>& a, half* dest) {

        const uint32_t MMARows    = HVector<N_COLS>::MMARows;
        const uint32_t MMACols    = HVector<N_COLS>::MMACols;
        const uint32_t MMAMatCols = HVector<N_COLS>::MMAMat::Cols;

        HVector<N_COLS> acc = a;

        // Reduce MMAs across rows
TIN_UNROLL
        for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++) {
TIN_UNROLL
            for (uint32_t j = 2; j <= MMARows; j <<= 1) {
TIN_UNROLL
                for (uint32_t mma_row = 0; mma_row < MMARows; mma_row += j) {
                    acc.m_mma_mat[a.mma_rc_to_idx(mma_row, mma_col)] = acc.m_mma_mat[a.mma_rc_to_idx(mma_row, mma_col)] +
                                                                       acc.m_mma_mat[a.mma_rc_to_idx(mma_row + j / 2, mma_col)];
                }
            }
        }

        // Reduce inside MMA
TIN_UNROLL
        for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++)
        {
            uint32_t mma_idx = a.mma_rc_to_idx(0, mma_col);
            uint32_t col = mma_col * MMAMatCols;
            reduce_sum(acc.m_mma_mat[mma_idx], dest + col);
        }
    };

    template <uint32_t NUM_VECTORS, uint32_t N_COLS, ReducerUpdateMode UPDATE_MODE = ReducerUpdateMode::STORE>
    class SumReducer {
    public:

        TIN_DEVICE_HOST static constexpr uint32_t shared_mem_size() {
            static_assert(NUM_VECTORS % WarpSize == 0, "Compile error: NUM_VECTORS must be a multiple of 32");
            return NumWarps * N_COLS;
        }

        static TIN_DEVICE void reduce_store(const HVector<N_COLS>& a, half* red_mem, half* dest) {

            // Reduce inside warp
            uint32_t warp_id = _warp_id();
            uint32_t lane_id = _lane_id();

            reduce_sum(a, red_mem + warp_id * N_COLS);
            __syncthreads();

            const uint32_t ColsPerThread = ColsPacked / NUM_VECTORS;
            const uint32_t NumIters = ColsPacked % NUM_VECTORS ? ColsPerThread + 1 : ColsPerThread;

            uint32_t packed_col_idx = warp_id * WarpSize + lane_id;
            half2* red_mem_h2 = (half2*)red_mem;
            auto dst_h2 = (half2*)dest;

TIN_UNROLL
            for (uint32_t j = 0; j < NumIters; j++) {

                if (packed_col_idx < ColsPacked) {

                    half2 red_reg = half2(0.f, 0.f);
TIN_UNROLL
                    for (uint32_t i = 0; i < NumWarps; i++) {
                        red_reg += red_mem_h2[packed_col_idx + i * ColsPacked];
                    }

                    if (UPDATE_MODE == ReducerUpdateMode::ATOMIC_ADD)
                        _atomic_addh2(dst_h2 + packed_col_idx, red_reg);
                    else
                        dst_h2[packed_col_idx] = red_reg;
                }

                packed_col_idx += NUM_VECTORS;
            }
            __syncthreads();
        }

        static TIN_DEVICE void reduce_store(const HVector<N_COLS>& a, half* red_mem, float* dest) {

            // Reduce inside warp
            uint32_t warp_id = _warp_id();
            uint32_t lane_id = _lane_id();

            reduce_sum(a, red_mem + warp_id * N_COLS);
            __syncthreads();

            const uint32_t ColsPerThread = ColsPacked / NUM_VECTORS;
            const uint32_t NumIters = ColsPacked % NUM_VECTORS ? ColsPerThread + 1 : ColsPerThread;

            uint32_t packed_col_idx = warp_id * WarpSize + lane_id;
            half2* red_mem_h2 = (half2*)red_mem;
            auto dst_f2 = (float2*)dest;

TIN_UNROLL
            for (uint32_t j = 0; j < NumIters; j++) {

                if (packed_col_idx < ColsPacked) {
                    float2 red_reg = { 0.f, 0.f };
TIN_UNROLL
                    for (uint32_t i = 0; i < NumWarps; i++) {
                        float2 reg = __half22float2(red_mem_h2[packed_col_idx + i * ColsPacked]);
                        red_reg.x += reg.x;
                        red_reg.y += reg.y;
                    }

                    if (UPDATE_MODE == ReducerUpdateMode::ATOMIC_ADD) { 
                        atomicAdd((float *)(dst_f2 + packed_col_idx)    , red_reg.x);
                        atomicAdd((float* )(dst_f2 + packed_col_idx) + 1, red_reg.y);
                    }
                    else {
                        dst_f2[packed_col_idx] = red_reg;
                    }
                }

                packed_col_idx += NUM_VECTORS;
            }
            __syncthreads();
        }

    protected:
        static const uint32_t NumWarps = NUM_VECTORS / WarpSize;
        static const uint32_t ColsPacked = N_COLS / 2;
    };

    template <uint32_t N_COLS0, uint32_t N_COLS1>
    TIN_DEVICE  HMatrix<N_COLS0, N_COLS1>
    outer_product_accumulate_reduce( const HVector<N_COLS0>& a, const HVector<N_COLS1>& b, HMatrix<N_COLS0, N_COLS1> c) {

        HMatrixB<WarpSize, N_COLS0> a_t = change_major_axis(a);
        HMatrixA<N_COLS1, WarpSize> b_t = change_major_axis(b.transpose());

        HMatrixA<N_COLS1, N_COLS0> c_t = c.transpose();
        HMatrixA<N_COLS1, N_COLS0> w = mad(b_t, a_t, c_t);
        HMatrixB<N_COLS0, N_COLS1> w_t = w.transpose();
        return w_t;
    };

    template <uint32_t N_COLS0, uint32_t N_COLS1>
    TIN_DEVICE  HMatrix<N_COLS0, N_COLS1>
        outer_product_reduce( const HVector<N_COLS0>& a, const HVector<N_COLS1>& b) {

        // Reduce inside warp
        HMatrixB<WarpSize, N_COLS0> at = change_major_axis(a);
        HMatrixA<N_COLS1, WarpSize> bt = change_major_axis(b.transpose());

        HMatrixA<N_COLS1, N_COLS0> w = mul(bt, at);
        HMatrixB<N_COLS0, N_COLS1> wt = w.transpose();
        return wt;
    };

    template < uint32_t NUM_VECTORS, uint32_t N_COLS0, uint32_t N_COLS1, ReducerUpdateMode UPDATE_MODE = ReducerUpdateMode::STORE >
    class OuterProductReducer {
    public:

        static TIN_DEVICE_HOST constexpr uint32_t shared_mem_size() {
            static_assert(NUM_VECTORS % WarpSize == 0, "Compile error: NUM_VECTORS must be a multiple of 32");
            return SmemSize;
        }

        static TIN_DEVICE void reduce_store_native(
            const HVector<N_COLS0>& a,
            const HVector<N_COLS1>& b,
            half* red_mem,
            half* dst) {

            // Reduce inside warp
            HMatrixB<N_COLS0, N_COLS1> wt = outer_product_reduce(a, b);

            // Reduce in shared memory
            uint32_t warp_id = _warp_id();

TIN_UNROLL
            for (uint32_t j = 2; j <= NumWarps; j <<= 1) {
                if (warp_id % j == j / 2) {
                    wt.store_native(&(red_mem[(warp_id / j) * MatSize]));
                }

                __syncthreads();

                if (warp_id % j == 0) {
                    HMatrixB<N_COLS0, N_COLS1> wt_1;
                    wt_1.load_native(&(red_mem[(warp_id / j) * MatSize]));
                    wt = wt + wt_1;
                }

                __syncthreads();
            }

            // Store back to shared memory
            if (warp_id == 0)
            {
                wt.store_native(&(red_mem[0]));
                __syncwarp();

                // Read into a cache friendly layout and update global mem
                const uint32_t MMARows = HMatrixB<N_COLS0, N_COLS1>::MMARows;
                const uint32_t MMACols = HMatrixB<N_COLS0, N_COLS1>::MMACols;
                const uint32_t NumRegs = HMatrixB<N_COLS0, N_COLS1>::MMAMat::NumRegs;

                uint32_t lane_id = _lane_id();
                auto dst_h2 = (half2*)dst;
                auto red_mem_h2 = (half2*)&(red_mem[0]);

TIN_UNROLL
                for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++) {
TIN_UNROLL
                    for (uint32_t mma_row = 0; mma_row < MMARows; mma_row++) {
                        uint32_t mma_idx = (mma_col * MMARows + mma_row) * NumRegs * WarpSize;
TIN_UNROLL
                        for (uint32_t i = 0; i < NumRegs; i++) {
                            uint32_t offset = mma_idx + i * WarpSize + lane_id;

                            half2 reg = red_mem_h2[offset];

                            if (UPDATE_MODE == ReducerUpdateMode::ATOMIC_ADD)
                                _atomic_addh2(dst_h2 + offset, reg);
                            else
                                dst_h2[offset] = reg;
                        }
                    }
                }
            }
            __syncthreads();
        };

        static TIN_DEVICE void reduce_store_native(
            const HVector<N_COLS0>& a,
            const HVector<N_COLS1>& b,
            half* red_mem,
            float* dst) {

            // Reduce inside warp
            HMatrixB<N_COLS0, N_COLS1> wt = outer_product_reduce(a, b);

            // Reduce in shared memory
            uint32_t warp_id = _warp_id();

            TIN_UNROLL
                for (uint32_t j = 2; j <= NumWarps; j <<= 1) {
                    if (warp_id % j == j / 2) {
                        wt.store_native(&(red_mem[(warp_id / j) * MatSize]));
                    }

                    __syncthreads();

                    if (warp_id % j == 0) {
                        HMatrixB<N_COLS0, N_COLS1> wt_1;
                        wt_1.load_native(&(red_mem[(warp_id / j) * MatSize]));
                        wt = wt + wt_1;
                    }

                    __syncthreads();
                }

            // Store back to shared memory
            if (warp_id == 0)
            {
                wt.store_native(&(red_mem[0]));
                __syncwarp();

                // Read into a cache friendly layout and update global mem
                const uint32_t MMARows = HMatrixB<N_COLS0, N_COLS1>::MMARows;
                const uint32_t MMACols = HMatrixB<N_COLS0, N_COLS1>::MMACols;
                const uint32_t NumRegs = HMatrixB<N_COLS0, N_COLS1>::MMAMat::NumRegs;

                uint32_t lane_id = _lane_id();
                auto dst_f2 = (float2*)dst;
                auto red_mem_h2 = (half2*)&(red_mem[0]);

TIN_UNROLL
                for (uint32_t mma_col = 0; mma_col < MMACols; mma_col++) {
TIN_UNROLL
                    for (uint32_t mma_row = 0; mma_row < MMARows; mma_row++) {
                        uint32_t mma_idx = (mma_col * MMARows + mma_row) * NumRegs * WarpSize;
TIN_UNROLL
                        for (uint32_t i = 0; i < NumRegs; i++) {
                            uint32_t offset = mma_idx + i * WarpSize + lane_id;
                            float2 reg = __half22float2(red_mem_h2[offset]);

                            if (UPDATE_MODE == ReducerUpdateMode::ATOMIC_ADD) {
                                atomicAdd((float*)(dst_f2 + offset)    , reg.x);
                                atomicAdd((float*)(dst_f2 + offset) + 1, reg.y);
                            }
                            else {
                                dst_f2[offset] = reg;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        };

    protected:
        static const uint32_t NumWarps = NUM_VECTORS / WarpSize;
        static const uint32_t MatSize = N_COLS0 * N_COLS1;
        static const uint32_t SmemSize = NumWarps > 1 ? MatSize * NumWarps / 2 : MatSize;

    };

    template <class T, uint32_t N_COLS0, uint32_t N_COLS1>
    TIN_DEVICE Vector<T, N_COLS1>
    mad(
        const Vector<T, N_COLS0>& a,
        const MatrixB<T, N_COLS0, N_COLS1>& b,
        const Vector<T, N_COLS1>& c) {

        auto am = (MatrixA<T, WarpSize, N_COLS0>) a;
        auto bm = (MatrixB<T, N_COLS0 , N_COLS1>) b;
        auto cm = (MatrixA<T, WarpSize, N_COLS1>) c;

        Vector<T, N_COLS1> r = mad(am,bm,cm);
        return r;
    };

    template <class T, uint32_t N_COLS0, uint32_t N_COLS1>
    TIN_DEVICE  Vector<T, N_COLS1>
    mul(
        const Vector<T, N_COLS0>& a,
        const MatrixB<T, N_COLS0,
        N_COLS1>& b) {

        Vector<T, N_COLS1> c;
        c.clear();
        return mad(a, b, c);
    };

    template <typename F, class T1, class... T2>
    TIN_DEVICE T1 map(F f, T1 v1, T2... v2) {
        T1 ret;
        const uint32_t items = v1.items_per_thread();

TIN_UNROLL
        for (uint32_t i = 0; i < items; i++) {
            ret.set_item(i, f(v1.get_item(i), v2.get_item(i)...));
        }

        return ret;
    };

    template <typename F, class T1, class... T2>
    TIN_DEVICE T1 map_mma(F f, T1 v1, T2... v2) {
        T1 ret;

        using T1Frag = MMAFragAcc<typename T1::underlying_type>;

TIN_UNROLL
        for (uint32_t i = 0; i < T1::NumMMAs; i++) {
            *(T1Frag*)&ret.m_mma_mat[i] =
                f(*(T1Frag*)&v1.m_mma_mat[i], *(MMAFragAcc<typename T2::underlying_type>*)&v2.m_mma_mat[i]...);
        }

        return ret;
    };

    template <typename TAct, class T1, class... T2>
    TIN_DEVICE T1 act_forward(TAct act, T1 v1, T2... v2) {
        T1 ret;
        const uint32_t items = v1.items_per_thread();

        TIN_UNROLL
            for (uint32_t i = 0; i < items; i++) {
                ret.set_item(i, act.forward(v1.get_item(i), v2.get_item(i)...));
            }

        return ret;
    };

    template <typename TAct, class T1, class... T2>
    TIN_DEVICE T1 act_backward(TAct act, T1 v1, T2... v2) {
        T1 ret;
        const uint32_t items = v1.items_per_thread();

        TIN_UNROLL
        for (uint32_t i = 0; i < items; i++) {
            ret.set_item(i, act.backward(v1.get_item(i), v2.get_item(i)...));
        }

        return ret;
    };


#endif
}
