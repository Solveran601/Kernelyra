module kernelyra_training_kernels
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_size_t
  use kernelyra_binary_loss, only: kr_binary_cross_entropy
  use kernelyra_gradient_kernels, only: kr_fortran_gradient_f32
  use kernelyra_gradient_layout, only: kr_row_offset
  use kernelyra_numeric_precision, only: kr_safe_std, kr_sigmoid
  use kernelyra_regression_loss, only: kr_squared_error
  use kernelyra_training_state, only: kr_update_bias
  use kernelyra_vector_kernels, only: kr_fortran_dot_f32, kr_fortran_update_f32
  implicit none
contains
  subroutine kr_fortran_binary_train_f32( &
      x, y, rows, features, weights, bias, learning_rate, decay, errors, gradient, loss) bind(C)
    integer(c_size_t), value, intent(in) :: rows, features
    real(c_float), intent(in) :: x(*), y(*)
    real(c_float), intent(inout) :: weights(*), bias
    real(c_float), value, intent(in) :: learning_rate, decay
    real(c_float), intent(out) :: errors(*), gradient(*), loss
    integer(c_size_t) :: row, offset
    real(c_float) :: score, probability, error, bias_gradient, inverse
    real(c_double) :: total_loss

    total_loss = 0.0_c_double
    bias_gradient = 0.0_c_float
    do row = 1_c_size_t, rows
      offset = kr_row_offset(row, features)
      score = kr_fortran_dot_f32(x(offset + 1_c_size_t), weights, features) + bias
      probability = kr_sigmoid(score)
      error = probability - y(row)
      errors(row) = error
      total_loss = total_loss + kr_binary_cross_entropy(y(row), probability)
      bias_gradient = bias_gradient + error
    end do
    call kr_fortran_gradient_f32(x, errors, rows, features, gradient)
    inverse = 1.0_c_float / real(rows, c_float)
    call kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, features)
    call kr_update_bias(bias, bias_gradient, learning_rate, inverse)
    loss = real(total_loss / real(rows, c_double), c_float)
  end subroutine kr_fortran_binary_train_f32

  subroutine kr_fortran_regression_train_f32( &
      x, y, rows, features, weights, bias, learning_rate, decay, target_mean, target_std, &
      errors, gradient, loss) bind(C)
    integer(c_size_t), value, intent(in) :: rows, features
    real(c_float), intent(in) :: x(*), y(*)
    real(c_float), intent(inout) :: weights(*), bias
    real(c_float), value, intent(in) :: learning_rate, decay, target_mean, target_std
    real(c_float), intent(out) :: errors(*), gradient(*), loss
    integer(c_size_t) :: row, offset
    real(c_float) :: target, error, bias_gradient, inverse, safe_target_std
    real(c_double) :: total_loss

    safe_target_std = kr_safe_std(target_std)
    total_loss = 0.0_c_double
    bias_gradient = 0.0_c_float
    do row = 1_c_size_t, rows
      offset = kr_row_offset(row, features)
      target = (y(row) - target_mean) / safe_target_std
      error = kr_fortran_dot_f32(x(offset + 1_c_size_t), weights, features) + bias - target
      errors(row) = 2.0_c_float * error
      total_loss = total_loss + kr_squared_error(error)
      bias_gradient = bias_gradient + errors(row)
    end do
    call kr_fortran_gradient_f32(x, errors, rows, features, gradient)
    inverse = 1.0_c_float / real(rows, c_float)
    call kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, features)
    call kr_update_bias(bias, bias_gradient, learning_rate, inverse)
    loss = real(total_loss / real(rows, c_double), c_float)
  end subroutine kr_fortran_regression_train_f32
end module kernelyra_training_kernels
