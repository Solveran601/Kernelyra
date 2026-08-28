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
  subroutine kr_fortran_guard_scalar_update_f32( &
      weights, gradient, values, bias, bias_gradient, learning_rate, inverse, decay, total_loss, status)
    integer(c_size_t), value, intent(in) :: values
    real(c_float), intent(in) :: weights(*), gradient(*), bias, bias_gradient
    real(c_float), value, intent(in) :: learning_rate, inverse, decay
    real(c_double), value, intent(in) :: total_loss
    integer(c_int), intent(out) :: status
    integer(c_size_t) :: index
    real(c_double) :: candidate

    status = 0_c_int
    if (.not. ieee_is_finite(total_loss) .or. .not. ieee_is_finite(bias) .or. &
        .not. ieee_is_finite(bias_gradient) .or. .not. ieee_is_finite(learning_rate) .or. &
        .not. ieee_is_finite(inverse) .or. .not. ieee_is_finite(decay)) then
      status = 1_c_int
      return
    end if
    do index = 1_c_size_t, values
      if (.not. ieee_is_finite(weights(index)) .or. .not. ieee_is_finite(gradient(index))) then
        status = 1_c_int
        return
      end if
      candidate = real(weights(index), c_double) - real(learning_rate, c_double) * ( &
          real(gradient(index), c_double) * real(inverse, c_double) + &
          real(decay, c_double) * real(weights(index), c_double))
      if (.not. ieee_is_finite(candidate) .or. &
          abs(candidate) > real(huge(0.0_c_float), c_double)) then
        status = 1_c_int
        return
      end if
    end do
    candidate = real(bias, c_double) - real(learning_rate, c_double) * &
        real(bias_gradient, c_double) * real(inverse, c_double)
    if (.not. ieee_is_finite(candidate) .or. &
        abs(candidate) > real(huge(0.0_c_float), c_double)) status = 1_c_int
  end subroutine kr_fortran_guard_scalar_update_f32

  subroutine kr_fortran_guard_vector_update_f32( &
      weights, gradient, values, biases, bias_gradient, bias_values, &
      learning_rate, inverse, decay, total_loss, status)
    integer(c_size_t), value, intent(in) :: values, bias_values
    real(c_float), intent(in) :: weights(*), gradient(*), biases(*), bias_gradient(*)
    real(c_float), value, intent(in) :: learning_rate, inverse, decay
    real(c_double), value, intent(in) :: total_loss
    integer(c_int), intent(out) :: status
    integer(c_size_t) :: index
    real(c_double) :: candidate

    status = 0_c_int
    if (.not. ieee_is_finite(total_loss) .or. .not. ieee_is_finite(learning_rate) .or. &
        .not. ieee_is_finite(inverse) .or. .not. ieee_is_finite(decay)) then
      status = 1_c_int
      return
    end if
    do index = 1_c_size_t, values
      if (.not. ieee_is_finite(weights(index)) .or. .not. ieee_is_finite(gradient(index))) then
        status = 1_c_int
        return
      end if
      candidate = real(weights(index), c_double) - real(learning_rate, c_double) * ( &
          real(gradient(index), c_double) * real(inverse, c_double) + &
          real(decay, c_double) * real(weights(index), c_double))
      if (.not. ieee_is_finite(candidate) .or. &
          abs(candidate) > real(huge(0.0_c_float), c_double)) then
        status = 1_c_int
        return
      end if
    end do
    do index = 1_c_size_t, bias_values
      if (.not. ieee_is_finite(biases(index)) .or. .not. ieee_is_finite(bias_gradient(index))) then
        status = 1_c_int
        return
      end if
      candidate = real(biases(index), c_double) - real(learning_rate, c_double) * &
          real(bias_gradient(index), c_double) * real(inverse, c_double)
      if (.not. ieee_is_finite(candidate) .or. &
          abs(candidate) > real(huge(0.0_c_float), c_double)) then
        status = 1_c_int
        return
      end if
    end do
  end subroutine kr_fortran_guard_vector_update_f32
end module kernelyra_training_guard
