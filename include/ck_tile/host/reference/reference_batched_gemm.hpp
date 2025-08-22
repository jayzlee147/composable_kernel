// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp0 = ck_tile::identity,
          typename ACCElementOp1 = ck_tile::identity>
CK_TILE_HOST void reference_batched_gemm(const HostTensor<ADataType>& a_b_m_k,
                                         const HostTensor<BDataType>& b_b_n_k,
                                         HostTensor<CDataType>& c_b_m_n,
                                         const AElementOp& a_element_op     = {},
                                         const BElementOp& b_element_op     = {},
                                         const ACCElementOp0& acc_element_op0 = {},
                                         const ACCElementOp1& acc_element_op1 = {})
{
    const auto B = a_b_m_k.mDesc.get_lengths()[0];
    const auto M = a_b_m_k.mDesc.get_lengths()[1];
    const auto N = b_b_n_k.mDesc.get_lengths()[1];
    const auto K = b_b_n_k.mDesc.get_lengths()[2];

    auto f = [&](auto batch, auto m) {
        for(std::size_t n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(std::size_t k = 0; k < K; ++k)
            {
                ADataType v_a = a_element_op(std::make_tuple(batch, m, n, k), a_b_m_k(batch, m, k));
                BDataType v_b = b_element_op(std::make_tuple(batch, m, n, k), b_b_n_k(batch, n, k));

                // ADataType v_a = a_element_op(a_b_m_k(batch, m, k));
                // BDataType v_b = b_element_op(b_b_n_k(batch, n, k));

                AccDataType v_acc0 = ck_tile::type_convert<AccDataType>(v_a) *
                        ck_tile::type_convert<AccDataType>(v_b);
 
                v_acc += acc_element_op0(std::make_tuple(batch, m, n, k), 
                    v_acc0, v_acc, ck_tile::type_convert<AccDataType>(v_a), 
                    ck_tile::type_convert<AccDataType>(v_b));
            }

            // c_b_m_n(batch, m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
            c_b_m_n(batch, m, n) = ck_tile::type_convert<CDataType>(acc_element_op1(std::make_tuple(batch, m, n), v_acc));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.get_lengths()[0], c_b_m_n.mDesc.get_lengths()[1])(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
