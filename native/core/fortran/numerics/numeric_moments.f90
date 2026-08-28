! Stable scalar moments for diagnostics, normalization planning and guardrails.
module kernelyra_numeric_moments
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_size_t
  implicit none
contains
  subroutine kr_fortran_moments_f32(values, count, mean, standard_deviation) bind(C)
    integer(c_size_t), value, intent(in) :: count
    real(c_float), intent(in) :: values(*)
    real(c_float), intent(out) :: mean, standard_deviation
    integer(c_size_t) :: index
    real(c_double) :: running_mean, running_square, delta, next_count

    if (count == 0_c_size_t) then
      mean = 0.0_c_float
      standard_deviation = 0.0_c_float
      return
    end if
    running_mean = 0.0_c_double
    running_square = 0.0_c_double
    do index = 1_c_size_t, count
      next_count = real(index, c_double)
      delta = real(values(index), c_double) - running_mean
      running_mean = running_mean + delta / next_count
      running_square = running_square + delta * (real(values(index), c_double) - running_mean)
    end do
    mean = real(running_mean, c_float)
    standard_deviation = real(sqrt(max(0.0_c_double, running_square / real(count, c_double))), c_float)
  end subroutine kr_fortran_moments_f32
end module kernelyra_numeric_moments
