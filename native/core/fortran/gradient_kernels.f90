module kernelyra_gradient_kernels
  use, intrinsic :: iso_c_binding, only: c_float, c_size_t
  use kernelyra_gradient_layout, only: kr_row_offset
  implicit none
contains
  subroutine kr_fortran_gradient_f32(x, errors, rows, features, gradient) bind(C)
    integer(c_size_t), value, intent(in) :: rows, features
    real(c_float), intent(in) :: x(*), errors(*)
    real(c_float), intent(out) :: gradient(*)
    integer(c_size_t) :: row, feature, offset

    if (features >= 32_c_size_t .and. rows >= 1024_c_size_t) then
      ! Each output feature has an independent reduction.  Parallelising this
      ! dimension avoids atomics on wide batches.  Small tables retain the
      ! cache-friendly SIMD row walk below, where thread scheduling costs more.
      !$omp parallel do schedule(static) private(row, offset)
      do feature = 1_c_size_t, features
        gradient(feature) = 0.0_c_float
        do row = 1_c_size_t, rows
          offset = kr_row_offset(row, features)
          gradient(feature) = gradient(feature) + x(offset + feature) * errors(row)
        end do
      end do
      !$omp end parallel do
    else
      do feature = 1_c_size_t, features
        gradient(feature) = 0.0_c_float
      end do
      do row = 1_c_size_t, rows
        offset = kr_row_offset(row, features)
        !$omp simd
        do feature = 1_c_size_t, features
          gradient(feature) = gradient(feature) + x(offset + feature) * errors(row)
        end do
      end do
    end if
  end subroutine kr_fortran_gradient_f32
end module kernelyra_gradient_kernels
