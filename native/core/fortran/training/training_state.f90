! Parameter-state updates that are common to binary and regression loops.
module kernelyra_training_state
  use, intrinsic :: iso_c_binding, only: c_float, c_size_t
  implicit none
contains
  subroutine kr_update_bias(bias, gradient, learning_rate, inverse)
    real(c_float), intent(inout) :: bias
    real(c_float), value, intent(in) :: gradient, learning_rate, inverse

    bias = bias - learning_rate * gradient * inverse
  end subroutine kr_update_bias

  subroutine kr_update_biases(biases, gradient, learning_rate, inverse, values)
    integer(c_size_t), value, intent(in) :: values
    real(c_float), intent(inout) :: biases(*)
    real(c_float), intent(in) :: gradient(*)
    real(c_float), value, intent(in) :: learning_rate, inverse
    integer(c_size_t) :: index

    !$omp simd
    do index = 1_c_size_t, values
      biases(index) = biases(index) - learning_rate * gradient(index) * inverse
    end do
  end subroutine kr_update_biases
end module kernelyra_training_state
