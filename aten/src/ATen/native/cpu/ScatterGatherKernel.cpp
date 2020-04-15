#include <iostream>
#include <utility>
#include <vector>

#include <ATen/native/ScatterGatherShapeChecks.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

namespace at { namespace native {

namespace {

// Used for `gather`-like methods
// Test:
// 1. index.size(d) == self.size(d) for all d != dim
void gather_shape_check(const Tensor& self, int64_t dim, const Tensor& index) {
  auto self_dims = ensure_nonempty_dim(self.dim());

  TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as input tensor"
  );

  for (int64_t i = 0; i < self_dims; ++i) {
    if (i != dim) {
      TORCH_CHECK(
        ensure_nonempty_size(index, i) == ensure_nonempty_size(self, i),
        "Size does not match at dimension ", i,
        " get ", ensure_nonempty_size(self, i),
        " vs ", ensure_nonempty_size(index, i)
      );
    }
  }
}

// Used for `scatter`-like methods
// Tests:
//  1. index.size(d) <= self.size(d) for all d != dim
//  2. index.size(d) <= src.size(d) for all d if src is a Tensor
void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const c10::optional<Tensor>& src_opt
) {
  bool is_wrong_shape = false;
  int64_t self_dims = ensure_nonempty_dim(self.dim());

  //  Check: index.size(d) <= self.size(d) for all d != dim
  for (int64_t d = 0; d < self_dims; ++d) {
    int64_t index_d_size = ensure_nonempty_size(index, d);
    if (d == dim) continue;
    if (index_d_size > ensure_nonempty_size(self, d)) {
      is_wrong_shape = true;
      break;
    }
  }

  //  Check: index.size(d) <= src.size(d) for all d if src is Tensor
  if (!is_wrong_shape && src_opt.has_value()) {
    auto src = src_opt.value();
    for (int64_t d = 0; d < self_dims; ++d) {
      int64_t index_d_size = ensure_nonempty_size(index, d);
      if (index_d_size > ensure_nonempty_size(src, d)) {
        is_wrong_shape = true;
        break;
      }
    }
  }

  if (src_opt.has_value()) {
    auto src = src_opt.value();
    TORCH_CHECK(!is_wrong_shape,
      "Expected index ", index.sizes(),
      " to be smaller than self ", self.sizes(),
      " apart from dimension ", dim,
      " and to be smaller size than src ", src.sizes()
    );
  }
  else {
    TORCH_CHECK(!is_wrong_shape,
      "Expected index ", index.sizes(),
      " to be smaller than self ", self.sizes(),
      " apart from dimension ", dim
    );
  }
}

template <bool is_scatter_like = true>
struct _cpu_scatter_gather_dim_loop {
  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    scalar_t* src_data, int64_t src_dim_stride,
    int64_t dim, int64_t index_dim_size,
    int64_t index_upper_bound,
    const func_t& f
  ) {

    for (int64_t i = 0; i < index_dim_size; ++i) {
      int64_t idx_dim = index_data[i * index_dim_stride];
      // we are not putting idx_dim in the error message because it disables
      // loop optimization in clang-7
      TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
        "index ", index_data[i * index_dim_stride],
        " is out of bounds for dimension ", dim,
        " with size ", index_upper_bound
      );

      f(
        self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
        src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
      );
    }
  }
};

template <bool is_scatter_like = true>
struct cpu_scatter_gather_base_kernel {
  template <typename func_t>
  void operator()(
    Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name,
    const func_t& f,
    bool serial_exec = true
  ) {
    // no-op if index is empty
    if (index.numel() == 0) {
      return;
    }

    dim = maybe_wrap_dim(dim, self.dim());

    if (is_scatter_like) {
      scatter_shape_check(self, dim, index, src);
    }
    else {
      gather_shape_check(self, dim, index);
    }

    // `dim` is traversed in the kernel. `skip_dim` will set
    // t.stride(dim) = 0 and t.size(dim) = 1. The stride prevents
    // the TensorIterator from advancing along dim, and the size
    //  makes sure that TensorIterator.DimCounter has the following 
    // form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
    // We enforce that shapes are identical for all d != dim, and
    // this defines the number of elements to iterate over.
    auto iter = TensorIterator();
    iter.dont_compute_common_dtype();
    iter.dont_resize_outputs();
    iter.add_output(self,/*skip_dim=*/dim);
    iter.add_input(src, src.device(), src.scalar_type(), /*skip_dim=*/dim);
    iter.add_input(index, /*skip_dim=*/dim);
    iter.build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, iter.dtype(),
      method_name, [&] {
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          constexpr auto SELF_ITER_STRIDE_IDX = 0;
          constexpr auto INDEX_ITER_STRIDE_IDX = 2;
          constexpr auto SRC_ITER_STRIDE_IDX = 1;

          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];

          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension and/or
          // whether `n` is smaller than `index_dim_size`
          if ((dim == self.dim() - 1) || (n < index_dim_size)) {
            for (int64_t nelem = 0; nelem < n; ++nelem) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                (scalar_t*)self_data_bytes, self_dim_stride,
                (int64_t*)index_data_bytes, index_dim_stride,
                (scalar_t*)src_data_bytes, src_dim_stride,
                dim, index_dim_size, index_upper_bound,
                f
              );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (int64_t i = 0; i < index_dim_size; ++i) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              for (int64_t nelem = 0; nelem < n; ++nelem) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                  "index ", *(int64_t*)index_data,
                  " is out of bounds for dimension ", dim,
                  " with size ", index_upper_bound
                );

                f(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
                );

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }

        };

        if (serial_exec) {
          iter.serial_for_each(loop, {0, iter.numel()});
        }
        else {
          iter.for_each(loop);
        }
      }
    );
  }
}; // struct cpu_scatter_gather_base_kernel
















template <bool broadcast_index, bool is_scatter_like>
struct cpu_scatter_gather_base_kernel_new {
  template <typename func_t>
  void operator()(
    Tensor& self,
    int64_t dim,
    const Tensor& index, 
    const Tensor& src,
    const std::string& method_name,
    const func_t& f,
    bool serial_exec = true
  ) {
    auto self_dim = self.dim();
    dim = maybe_wrap_dim(dim, self_dim);

    TORCH_CHECK(dim == 0 || dim < self_dim, method_name, "(): Indexing dim ", dim, " is out of bounds of tensor");
    TORCH_CHECK(index.scalar_type() == ScalarType::Long, method_name, "(): Expected dtype int64 for index");
    TORCH_CHECK(self.scalar_type() == src.scalar_type(), method_name, "(): self and result must have the same scalar type");

    if (broadcast_index){
      TORCH_CHECK_INDEX(index.dim() <= 1, method_name, "(): Index is supposed to be a vector");
      TORCH_CHECK(is_scatter_like ? index.numel() > ensure_nonempty_size(src, dim) : true, 
                  method_name, "(): Index size ", index.numel(), " does not match source size ",
                  ensure_nonempty_size(src, dim), " along dim ", dim, ".");
      // TODO: This is not complete as it does not check `self` and `src` agreement for the scatter (index_put) case.
      //       May be better to just reuse `scatter_shape_check`.
    } else {
      is_scatter_like ? scatter_shape_check(self, dim, index, src)
                      : gather_shape_check(self, dim, index);
    }

    // // `dim` is traversed in the kernel, that is why index.stride(dim) = 0.
    // // Also, size[dim] = 1 makes sure that TensorIterator.DimCounter
    // // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
    // auto dim_erased_sizes = self.sizes().vec();
    // dim_erased_sizes[dim] = 1;

    // // set self.shape = src.shape (= index.shape for scatter/gather kernels),
    // // this defines the number of elements to iterate over,
    // // and set self.stride(dim) = src.stride(dim) = 0,
    // // because `dim` is traversed in the kernel.
    // auto self_restrided = restride_dim(self, dim, dim_erased_sizes);
    // auto src_restrided = restride_dim(src, dim, dim_erased_sizes);

    // // TODO(taylorrobie): look for shortcuts to build iterator more cheaply
    // //   for small inputs. (and consider a `skip_optimizations` arg to `build`)
    // auto iter = TensorIterator();
    // iter.dont_compute_common_dtype();
    // iter.dont_resize_outputs();
    // iter.add_output(self_restrided);
    // iter.add_input(src_restrided, src.device(), src.scalar_type());
    // if (!broadcast_index){
    //   auto index_strides = ensure_nonempty_vec(index.strides().vec());
    //   index_strides[dim] = 0;
    //   auto index_restrided = index.as_strided(dim_erased_sizes, index_strides);
    //   iter.add_input(index_restrided);
    // }
    // iter.build();

    // auto self_dim_stride = ensure_nonempty_stride(self, dim);
    // auto self_dim_size = ensure_nonempty_size(self, dim);

    // auto index_ptr = broadcast_index ? index.data_ptr<int64_t>() : nullptr;
    // auto index_dim_stride = broadcast_index ? 1  : ensure_nonempty_stride(index, dim);
    // auto index_dim_size = broadcast_index ? index.numel() 
    //                                       : ensure_nonempty_size(index, dim);

    // auto src_dim_stride = ensure_nonempty_stride(src, dim);
    // auto src_dim_size = ensure_nonempty_size(src, dim);

    // auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    // AT_DISPATCH_ALL_TYPES_AND2(
    //   ScalarType::Bool, ScalarType::Half, iter.dtype(),
    //   method_name, [&] {
    //     auto loop = [&](char** data, const int64_t* strides, int64_t n) {

    //     };

    //     // serial_exec ? iter.serial_for_each(loop, {0, iter.numel()})
    //     //             : iter.for_each(loop);

    //   }
    // );


  }
}; // struct cpu_scatter_gather_base_kernel_new

void gather_cpu_kernel_new(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index, bool broadcast_index = false) {
  // TODO(taylorrobie): do not submit with broadcast_index special cased.
  cpu_scatter_gather_base_kernel_new<true, /*is_scatter_like=*/false>()(
    result, dim, index, self,
    broadcast_index ? "index_select_out_cpu" : "gather_out_cpu", 
    [] (auto* lhs, const auto* rhs) {
      *lhs = *rhs;
    },
    /*serial_exec=*/false
  );
}













void gather_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  cpu_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
    result, dim, index, self,
    "gather_out_cpu", [] (auto* lhs, const auto* rhs) {
      *lhs = *rhs;
    },
    /*serial_exec=*/false
  );
}

void scatter_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_cpu_", [] (auto* lhs, const auto* rhs) {
      *lhs = *rhs;
    },
    /*serial_exec=*/false
  );
}

void scatter_fill_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, Scalar src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, self,
    "scatter_fill_cpu_", [src] (auto* lhs, const auto* rhs) {
      using scalar_t = typename std::remove_pointer<decltype(lhs)>::type;
      *lhs = src.to<scalar_t>();
    },
    /*serial_exec=*/false
  );
}

void scatter_add_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_add_", [] (auto* lhs, const auto* rhs) {
      *lhs += *rhs;
    },
    /*serial_exec=*/true
  );
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);

REGISTER_DISPATCH(gather_new_stub, &gather_cpu_kernel_new);

}} // namespace at::native
