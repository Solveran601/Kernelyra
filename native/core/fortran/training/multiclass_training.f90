! Full sequential softmax classifier update.  C++ retains batch selection and
! lifetime management; this module owns logits, probabilities, losses and
! gradients when the Fortran component is active.
module kernelyra_multiclass_training
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_int, c_size_t
  use kernelyra_activation_softmax, only: kr_fortran_softmax_f32
  use kernelyra_gradient_layout, only: kr_row_offset
  use kernelyra_multiclass_loss, only: kr_multiclass_cross_entropy
  use kernelyra_matrix_scores, only: kr_fortran_dense_scores_f32
  use kernelyra_training_state, only: kr_update_biases
  use kernelyra_training_guard, only: kr_fortran_guard_vector_update_f32
  use kernelyra_vector_kernels, only: kr_fortran_update_f32
  implicit none
contains
  subroutine kr_fortran_multiclass_train_f32( &
      x, y, rows, features, classes, weights, bias, learning_rate, decay, &
      probabilities, gradient, bias_gradient, report_loss, loss, status) bind(C)
    integer(c_size_t), value, intent(in) :: rows, features, classes
    real(c_float), intent(in) :: x(*), y(*)
    real(c_float), intent(inout) :: weights(*), bias(*)
    real(c_float), value, intent(in) :: learning_rate, decay
    real(c_float), intent(inout) :: probabilities(*)
    real(c_float), intent(out) :: gradient(*), bias_gradient(*), loss
    integer(c_int), value, intent(in) :: report_loss
    integer(c_int), intent(out) :: status
    integer(c_int) :: guard_status
    integer(c_size_t) :: row, feature, category, offset, weight_offset, truth, weight_values
    real(c_float) :: error, inverse, feature_value
    real(c_double) :: total_loss

    status = 0_c_int
    loss = 0.0_c_float
    if (rows == 0_c_size_t .or. features == 0_c_size_t .or. classes < 2_c_size_t) then
      status = 2_c_int
      return
    end if
    weight_values = features * classes
    do category = 1_c_size_t, classes
      bias_gradient(category) = 0.0_c_float
    end do
    do feature = 1_c_size_t, weight_values
      gradient(feature) = 0.0_c_float
    end do
    total_loss = 0.0_c_double
    do row = 1_c_size_t, rows
      if (y(row) < 0.0_c_float .or. y(row) >= real(classes, c_float) .or. &
          abs(y(row) - anint(y(row))) > 1.0e-5_c_float) then
        status = 1_c_int
        return
      end if
      truth = int(y(row), c_size_t)
      offset = kr_row_offset(row, features)
      call kr_fortran_dense_scores_f32(x(offset + 1_c_size_t), features, classes, weights, bias, probabilities)
      call kr_fortran_softmax_f32(probabilities, classes)
      if (report_loss /= 0_c_int) then
        total_loss = total_loss + kr_multiclass_cross_entropy(probabilities, truth)
      end if
      do category = 1_c_size_t, classes
        error = probabilities(category) - merge(1.0_c_float, 0.0_c_float, category == truth + 1_c_size_t)
        probabilities(category) = error
        bias_gradient(category) = bias_gradient(category) + error
      end do
      ! The weight layout is feature-major. Walking categories inside each
      ! feature keeps both gradient and probabilities contiguous for SIMD.
      do feature = 1_c_size_t, features
        feature_value = x(offset + feature)
        weight_offset = (feature - 1_c_size_t) * classes
        !$omp simd
        do category = 1_c_size_t, classes
          gradient(weight_offset + category) = gradient(weight_offset + category) + &
              feature_value * probabilities(category)
        end do
      end do
    end do
    inverse = 1.0_c_float / real(rows, c_float)
    call kr_fortran_guard_vector_update_f32( &
        weights, gradient, weight_values, bias, bias_gradient, classes, &
        learning_rate, inverse, decay, total_loss, guard_status)
    if (guard_status /= 0_c_int) then
      status = 3_c_int
      return
    end if
    call kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, weight_values)
    call kr_update_biases(bias, bias_gradient, learning_rate, inverse, classes)
    loss = real(total_loss / real(rows, c_double), c_float)
  end subroutine kr_fortran_multiclass_train_f32
end module kernelyra_multiclass_training
