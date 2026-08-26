! Stable softmax over a caller-owned contiguous score vector.
module kernelyra_activation_softmax
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_size_t
  implicit none
contains
  subroutine kr_fortran_softmax_f32(values, count) bind(C)
    integer(c_size_t), value, intent(in) :: count
    real(c_float), intent(inout) :: values(*)
    integer(c_size_t) :: index
    real(c_float) :: maximum
    real(c_double) :: denominator

    if (count == 0_c_size_t) return
    maximum = values(1)
    do index = 2_c_size_t, count
      maximum = max(maximum, values(index))
    end do
    denominator = 0.0_c_double
    do index = 1_c_size_t, count
      values(index) = exp(values(index) - maximum)
      denominator = denominator + real(values(index), c_double)
    end do
    if (denominator <= 0.0_c_double) then
      do index = 1_c_size_t, count
        values(index) = 1.0_c_float / real(count, c_float)
      end do
      return
    end if
    do index = 1_c_size_t, count
      values(index) = real(real(values(index), c_double) / denominator, c_float)
    end do
  end subroutine kr_fortran_softmax_f32
end module kernelyra_activation_softmax
