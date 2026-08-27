! Pre-update numerical guard for the Fortran-owned training path.
!
! It deliberately performs no allocation and never changes parameters.  A
! caller invokes it after accumulating gradients but before any weight update;
! a rejected batch therefore cannot poison a live model or a checkpoint.
module kernelyra_training_guard
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_int, c_size_t
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
contains
  subroutine kr_fortran_guard_scalar_update_f32(gradient, values, bias_gradient, total_loss, status)
    integer(c_size_t), value, intent(in) :: values
    real(c_float), intent(in) :: gradient(*), bias_gradient
    real(c_double), value, intent(in) :: total_loss
    integer(c_int), intent(out) :: status
    integer(c_size_t) :: index

    status = 0_c_int
    if (.not. ieee_is_finite(total_loss) .or. .not. ieee_is_finite(bias_gradient)) then
      status = 1_c_int
      return
    end if
    do index = 1_c_size_t, values
      if (.not. ieee_is_finite(gradient(index))) then
        status = 1_c_int
        return
      end if
    end do
  end subroutine kr_fortran_guard_scalar_update_f32

  subroutine kr_fortran_guard_vector_update_f32( &
      gradient, values, bias_gradient, bias_values, total_loss, status)
    integer(c_size_t), value, intent(in) :: values, bias_values
    real(c_float), intent(in) :: gradient(*), bias_gradient(*)
    real(c_double), value, intent(in) :: total_loss
    integer(c_int), intent(out) :: status
    integer(c_size_t) :: index

    status = 0_c_int
    if (.not. ieee_is_finite(total_loss)) then
      status = 1_c_int
      return
    end if
    do index = 1_c_size_t, values
      if (.not. ieee_is_finite(gradient(index))) then
        status = 1_c_int
        return
      end if
    end do
    do index = 1_c_size_t, bias_values
      if (.not. ieee_is_finite(bias_gradient(index))) then
        status = 1_c_int
        return
      end if
    end do
  end subroutine kr_fortran_guard_vector_update_f32
end module kernelyra_training_guard
