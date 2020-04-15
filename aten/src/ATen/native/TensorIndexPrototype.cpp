#include <ATen/native/TensorIndexPrototype.h>

#include <map>
#include <numeric>
#include <iostream>
#include <vector>
#include <string>

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {


DEFINE_DISPATCH(index_select_memcpy_dim0_numel_outer_kernel_stub);
DEFINE_DISPATCH(index_select_ptr_assign_dim0_kernel_true_stub);
DEFINE_DISPATCH(index_select_ptr_assign_dim0_kernel_false_stub);
DEFINE_DISPATCH(index_select_memcpy_dim1_kernel_true_stub);
DEFINE_DISPATCH(index_select_memcpy_dim1_kernel_false_stub);
DEFINE_DISPATCH(index_select_ptr_assign_dim1_kernel_true_stub);
DEFINE_DISPATCH(index_select_ptr_assign_dim1_kernel_false_stub);

DEFINE_DISPATCH(gather_new_stub);


// ==============================================================================================================
// == NOTE: There is a carbon copy of these kernels in TensorIndexPrototypeKernel.cpp ===========================
// ==============================================================================================================

void assert_special_case(const Tensor& self, const Tensor& index, int64_t dim){
    TORCH_CHECK_INDEX(index.dim() == 1, "index must be 1D.");
    TORCH_CHECK_INDEX(self.dim() == 2, "self must be 2D.");
    TORCH_CHECK(dim == 0 || dim == 1, "dim must be zero or one.")
    TORCH_CHECK(self.is_contiguous(), "implementation assumes self is contiguous.")
}

void index_select_memcpy_dim0_numel_outer_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/0);
    constexpr int64_t dim = 0;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    auto self_dim_size = self.size(dim);

    const size_t type_size = elementSize(self.scalar_type());
    const size_t feature_len = sizes[1 - dim];
    const size_t copy_bytes = feature_len * type_size;

    auto index_buffer = index.data_ptr<int64_t>();
    auto self_buffer = static_cast<char*>(self.data_ptr());
    auto result_buffer = static_cast<char*>(result.data_ptr());
    
    for (size_t i = 0; i < numel; ++i){
        auto source_i = index_buffer[i];
        TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

        auto self_data = self_buffer + source_i * copy_bytes;
        auto result_data = result_buffer + i * copy_bytes;
        memcpy(result_data, self_data, copy_bytes);
    }
}

template <bool NUMEL_OUTER>
void index_select_ptr_assign_dim0_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/0);
    constexpr int64_t dim = 0;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    auto self_dim_size = self.size(dim);
    const size_t feature_len = sizes[1 - dim];

    AT_DISPATCH_ALL_TYPES_AND(
        ScalarType::Bool, self.scalar_type(), "index_select_memcpy_dim1_kernel", [&] {
            auto index_buffer = index.data_ptr<int64_t>();
            auto self_buffer = self.data_ptr<scalar_t>();
            auto result_buffer = result.data_ptr<scalar_t>();

            if (NUMEL_OUTER) {
                for (size_t i = 0; i < numel; ++i){
                    auto source_i = index_buffer[i];
                    TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                    auto self_data = self_buffer + source_i * feature_len;
                    auto result_data = result_buffer + i * feature_len;
                    for (size_t j = 0; j < feature_len; ++j){
                        *(result_data + j) = *(self_data + j);
                    }
                }
            } else {
                for (size_t j = 0; j < feature_len; ++j){
                    for (size_t i = 0; i < numel; ++i){
                        auto source_i = index_buffer[i];
                        TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                        auto self_data = self_buffer + source_i * feature_len;
                        auto result_data = result_buffer + i * feature_len;
                        *(result_data + j) = *(self_data + j);
                    }
                }
            }
        });
}

template <bool NUMEL_OUTER>
void index_select_ptr_assign_dim1_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/1);
    constexpr int64_t dim = 1;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    const size_t self_dim_size = self.size(dim);
    const size_t feature_len = sizes[1 - dim];

    AT_DISPATCH_ALL_TYPES_AND(
        ScalarType::Bool, self.scalar_type(), "index_select_memcpy_dim1_kernel", [&] {
            auto index_buffer = index.data_ptr<int64_t>();
            auto self_buffer = self.data_ptr<scalar_t>();
            auto result_buffer = result.data_ptr<scalar_t>();

            if (NUMEL_OUTER) {
                for (int64_t i = 0; i < numel; ++i){
                    auto source_i = index_buffer[i];
                    TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                    for (int64_t j = 0; j < feature_len; ++j){
                        auto self_data = self_buffer + (source_i + j * self_dim_size);
                        auto result_data = result_buffer + (i + j * numel);
                        *result_data = *self_data;
                    }
                }
            } else {
                for (int64_t j = 0; j < feature_len; ++j){
                    for (int64_t i = 0; i < numel; ++i){
                        auto source_i = index_buffer[i];
                        TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                        auto self_data = self_buffer + (source_i + j * self_dim_size);
                        auto result_data = result_buffer + (i + j * numel);
                        *result_data = *self_data;
                    }
                }
            }
        });
}

template <bool NUMEL_OUTER>
void index_select_memcpy_dim1_kernel(Tensor& result, const Tensor& self, const Tensor& index){
    assert_special_case(self, index, /*dim=*/1);
    constexpr int64_t dim = 1;

    auto sizes = self.sizes().vec();
    const int64_t numel = index.numel();
    
    sizes[dim] = numel;
    result.resize_(sizes);
    const size_t self_dim_size = self.size(dim);
    const size_t feature_len = sizes[1 - dim];
    const size_t type_size = elementSize(self.scalar_type());

    auto index_buffer = index.data_ptr<int64_t>();
    auto self_buffer = static_cast<char*>(self.data_ptr());
    auto result_buffer = static_cast<char*>(result.data_ptr());

    if (NUMEL_OUTER) {
        for (int64_t i = 0; i < numel; ++i){
            auto source_i = index_buffer[i];
            TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

            for (int64_t j = 0; j < feature_len; ++j){
                auto self_data = self_buffer + (source_i + j * self_dim_size) * type_size;
                auto result_data = result_buffer + (i + j * numel) * type_size;
                memcpy(result_data, self_data, type_size);
            }
        }
    } else {
        for (int64_t j = 0; j < feature_len; ++j){
            for (int64_t i = 0; i < numel; ++i){
                auto source_i = index_buffer[i];
                TORCH_CHECK_INDEX((source_i >= 0) && (source_i < self_dim_size), "index out of range in self");

                auto self_data = self_buffer + (source_i + j * self_dim_size) * type_size;
                auto result_data = result_buffer + (i + j * numel) * type_size;
                memcpy(result_data, self_data, type_size);
            }
        }
    }
}


// This is a hacky way to do implementation switching, but it's good enough for prototyping.
const std::map<std::pair<int64_t, std::string>, int> impl_map = {
    {{0, "memcpy"},              0},
    {{0, "memcpy_stub"},         1},
    {{0, "at_dispatch_ij"},      2},
    {{0, "at_dispatch_ij_stub"}, 3},
    {{0, "at_dispatch_ji"},      4},
    {{0, "at_dispatch_ji_stub"}, 5},

    {{1, "memcpy_ij"},           100},
    {{1, "memcpy_ij_stub"},      101},
    {{1, "memcpy_ji"},           102},
    {{1, "memcpy_ji_stub"},      103},
    {{1, "at_dispatch_ij"},      104},
    {{1, "at_dispatch_ij_stub"}, 105},
    {{1, "at_dispatch_ji"},      106},
    {{1, "at_dispatch_ji_stub"}, 107},
};


Tensor debug_index(const Tensor& self, int64_t dim, const Tensor& ind, std::string impl) {
    TORCH_CHECK_INDEX(ind.dim() <= 1, "debug_index: Index is supposed to be a vector");
    TORCH_CHECK(ind.scalar_type() == ScalarType::Long, "debug_index(): Expected dtype int64 for index");    

    const Tensor index = ind.contiguous();
    Tensor result = at::empty({0}, self.options());

    switch (impl_map.at({dim, impl})){
        case 0:
            index_select_memcpy_dim0_numel_outer_kernel(result, self, index);
            break;
        case 1:
            index_select_memcpy_dim0_numel_outer_kernel_stub(kCPU, result, self, index);
            break;
        case 2:
            index_select_ptr_assign_dim0_kernel</*NUMEL_OUTER=*/true>(result, self, index);
            break;
        case 3:
            index_select_ptr_assign_dim0_kernel_true_stub(kCPU, result, self, index);
            break;
        case 4:
            index_select_ptr_assign_dim0_kernel</*NUMEL_OUTER=*/false>(result, self, index);
            break;
        case 5:
            index_select_ptr_assign_dim0_kernel_false_stub(kCPU, result, self, index);
            break;
        
        case 100:
            index_select_memcpy_dim1_kernel</*NUMEL_OUTER=*/true>(result, self, index);
            break;
        case 101:
            index_select_memcpy_dim1_kernel_true_stub(kCPU, result, self, index);
            break;
        case 102:
            index_select_memcpy_dim1_kernel</*NUMEL_OUTER=*/false>(result, self, index);
            break;
        case 103:
            index_select_memcpy_dim1_kernel_false_stub(kCPU, result, self, index);
            break;
        case 104:
            index_select_ptr_assign_dim1_kernel</*NUMEL_OUTER=*/true>(result, self, index);
            break;
        case 105:
            index_select_ptr_assign_dim1_kernel_true_stub(kCPU, result, self, index);
            break;
        case 106:
            index_select_ptr_assign_dim1_kernel</*NUMEL_OUTER=*/false>(result, self, index);
            break;
        case 107:
            index_select_ptr_assign_dim1_kernel_false_stub(kCPU, result, self, index);
            break;
        
    }

    return result;
}
















void resize_result(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index, bool broadcast_index = false){
    if (!broadcast_index){
        result.resize_(index.sizes());
        return;
    }

    auto sizes = self.sizes().vec();
    dim = maybe_wrap_dim(dim, self.dim());
    sizes[dim] = index.numel();
    result.resize_(sizes);
}

Tensor& gather_out_cpu_new_(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index,
                            bool sparse_grad, bool broadcast_index = false) {
  resize_result(result, self, dim, index, broadcast_index);
  gather_new_stub(result.device().type(), result, self, dim, index, broadcast_index);
  return result;
}

// Tensor gather_cpu_new_(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
//   Tensor result = at::empty({0}, self.options());
//   return gather_out_cpu(result, self, dim, index, sparse_grad);
// }

Tensor index_select_out_cpu_new_(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index){
    return gather_out_cpu_new_(result, self, dim, index, /*sparse_grad=*/false, /*broadcast_index=*/true);
}

Tensor index_select_cpu_new_(const Tensor & self, int64_t dim, const Tensor & index){
    Tensor result = at::empty({0}, self.options());
    return index_select_out_cpu_new_(result, self, dim, index);
}

}} // namespace at::native
