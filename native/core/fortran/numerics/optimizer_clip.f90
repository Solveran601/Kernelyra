! Global-L2 clipping is opt-in through the ABI.  It returns the norm observed
! before clipping so Model Guard can record the size of a rejected update.
module kernelyra_optimizer_clip
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_size_t
  implicit none
contains
  function kr_fortran_clip_l2_f32(values, count, maximum_norm) result(observed_norm) bind(C)
    integer(c_size_t), value, intent(in) :: count
    real(c_float), intent(inout) :: values(*)
    real(c_float), value, intent(in) :: maximum_norm
    real(c_float) :: observed_norm
    integer(c_size_t) :: index
    real(c_double) :: total
    real(c_float) :: scale

    total = 0.0_c_double
    !$omp simd reduction(+:total)
    do index = 1_c_size_t, count
      total = total + real(values(index), c_double) * real(values(index), c_double)
    end do
    observed_norm = real(sqrt(total), c_float)
    if (maximum_norm <= 0.0_c_float .or. observed_norm <= maximum_norm) return
    scale = maximum_norm / observed_norm
    !$omp simd
    do index = 1_c_size_t, count
      values(index) = values(index) * scale
    end do
  end function kr_fortran_clip_l2_f32
end module kernelyra_optimizer_clip
