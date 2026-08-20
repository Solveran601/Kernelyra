module kernelyra_numeric_kernels
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_size_t
  implicit none
contains
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

  subroutine kr_fortran_gradient_f32(x, errors, rows, features, gradient) bind(C)
    integer(c_size_t), value, intent(in) :: rows, features
    real(c_float), intent(in) :: x(*), errors(*)
    real(c_float), intent(out) :: gradient(*)
    integer(c_size_t) :: row, feature, offset

    do feature = 1_c_size_t, features
      gradient(feature) = 0.0_c_float
    end do
    do row = 1_c_size_t, rows
      offset = (row - 1_c_size_t) * features
      !$omp simd
      do feature = 1_c_size_t, features
        gradient(feature) = gradient(feature) + x(offset + feature) * errors(row)
      end do
    end do
  end subroutine kr_fortran_gradient_f32

  subroutine kr_fortran_binary_train_f32( &
      x, y, rows, features, weights, bias, learning_rate, decay, errors, gradient, loss) bind(C)
    integer(c_size_t), value, intent(in) :: rows, features
    real(c_float), intent(in) :: x(*), y(*)
    real(c_float), intent(inout) :: weights(*), bias
    real(c_float), value, intent(in) :: learning_rate, decay
    real(c_float), intent(out) :: errors(*), gradient(*), loss
    integer(c_size_t) :: row, offset
    real(c_float) :: score, probability, error, bounded, bias_gradient, inverse
    real(c_double) :: total_loss

    total_loss = 0.0_c_double
    bias_gradient = 0.0_c_float
    do row = 1_c_size_t, rows
      offset = (row - 1_c_size_t) * features
      score = kr_fortran_dot_f32(x(offset + 1_c_size_t), weights, features) + bias
      probability = 1.0_c_float / (1.0_c_float + exp(-max(-30.0_c_float, min(30.0_c_float, score))))
      error = probability - y(row)
      errors(row) = error
      bounded = max(1.0e-7_c_float, min(1.0_c_float - 1.0e-7_c_float, probability))
      total_loss = total_loss - real(y(row), c_double) * log(real(bounded, c_double)) - &
          real(1.0_c_float - y(row), c_double) * log(real(1.0_c_float - bounded, c_double))
      bias_gradient = bias_gradient + error
    end do
    call kr_fortran_gradient_f32(x, errors, rows, features, gradient)
    inverse = 1.0_c_float / real(rows, c_float)
    call kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, features)
    bias = bias - learning_rate * bias_gradient * inverse
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

    safe_target_std = target_std
    if (abs(safe_target_std) <= 1.0e-12_c_float) safe_target_std = 1.0_c_float
    total_loss = 0.0_c_double
    bias_gradient = 0.0_c_float
    do row = 1_c_size_t, rows
      offset = (row - 1_c_size_t) * features
      target = (y(row) - target_mean) / safe_target_std
      error = kr_fortran_dot_f32(x(offset + 1_c_size_t), weights, features) + bias - target
      errors(row) = 2.0_c_float * error
      total_loss = total_loss + real(error, c_double) * real(error, c_double)
      bias_gradient = bias_gradient + errors(row)
    end do
    call kr_fortran_gradient_f32(x, errors, rows, features, gradient)
    inverse = 1.0_c_float / real(rows, c_float)
    call kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, features)
    bias = bias - learning_rate * bias_gradient * inverse
    loss = real(total_loss / real(rows, c_double), c_float)
  end subroutine kr_fortran_regression_train_f32
end module kernelyra_numeric_kernels
