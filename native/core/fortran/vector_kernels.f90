module kernelyra_vector_kernels
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_int, c_size_t
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
contains
  function kr_fortran_all_finite_f32(values, count) result(ok) bind(C)
    integer(c_size_t), value, intent(in) :: count
    real(c_float), intent(in) :: values(*)
    integer(c_int) :: ok
    integer(c_size_t) :: index

    ok = 1_c_int
    do index = 1_c_size_t, count
      if (.not. ieee_is_finite(values(index))) then
        ok = 0_c_int
        return
      end if
    end do
  end function kr_fortran_all_finite_f32

  function kr_fortran_l2_norm_f32(values, count) result(norm) bind(C)
    integer(c_size_t), value, intent(in) :: count
    real(c_float), intent(in) :: values(*)
    real(c_float) :: norm
    integer(c_size_t) :: index
    real(c_double) :: total

    total = 0.0_c_double
    !$omp simd reduction(+:total)
    do index = 1_c_size_t, count
      total = total + real(values(index), c_double) * real(values(index), c_double)
    end do
    norm = real(sqrt(total), c_float)
  end function kr_fortran_l2_norm_f32

  function kr_fortran_dot_f32(left, right, values) result(total) bind(C)
    integer(c_size_t), value, intent(in) :: values
    real(c_float), intent(in) :: left(*), right(*)
    real(c_float) :: total
    integer(c_size_t) :: index

    total = 0.0_c_float
    !$omp simd reduction(+:total)
    do index = 1_c_size_t, values
      total = total + left(index) * right(index)
    end do
  end function kr_fortran_dot_f32

  subroutine kr_fortran_axpy_f32(output, row, scale, values) bind(C)
    integer(c_size_t), value, intent(in) :: values
    real(c_float), value, intent(in) :: scale
    real(c_float), intent(inout) :: output(*)
    real(c_float), intent(in) :: row(*)
    integer(c_size_t) :: index

    !$omp simd
    do index = 1_c_size_t, values
      output(index) = output(index) + row(index) * scale
    end do
  end subroutine kr_fortran_axpy_f32

  subroutine kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, values) bind(C)
    integer(c_size_t), value, intent(in) :: values
    real(c_float), value, intent(in) :: learning_rate, inverse, decay
    real(c_float), intent(inout) :: weights(*)
    real(c_float), intent(in) :: gradient(*)
    integer(c_size_t) :: index

    !$omp simd
    do index = 1_c_size_t, values
      weights(index) = weights(index) - learning_rate * &
          (gradient(index) * inverse + decay * weights(index))
    end do
  end subroutine kr_fortran_update_f32
end module kernelyra_vector_kernels
