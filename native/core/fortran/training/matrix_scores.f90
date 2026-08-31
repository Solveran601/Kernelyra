! Dense row x (feature-by-class) scores.  The flat layout is shared directly
! with the C ABI: weights(feature * classes + class).
module kernelyra_matrix_scores
  use, intrinsic :: iso_c_binding, only: c_float, c_size_t
  implicit none
contains
  subroutine kr_fortran_dense_scores_f32(row, features, classes, weights, bias, scores)
    integer(c_size_t), value, intent(in) :: features, classes
    real(c_float), intent(in) :: row(*), weights(*), bias(*)
    real(c_float), intent(out) :: scores(*)
    integer(c_size_t) :: feature, category, weight_offset
    real(c_float) :: value

    ! This routine is invoked once per input row. Spawning an OpenMP team for
    ! every row dominated wide multiclass batches. Accumulating one contiguous
    ! weight column at a time exposes SIMD across classes and preserves the
    ! original feature-order sum for every score.
    do category = 1_c_size_t, classes
      scores(category) = bias(category)
    end do
    do feature = 1_c_size_t, features
      value = row(feature)
      weight_offset = (feature - 1_c_size_t) * classes
      !$omp simd
      do category = 1_c_size_t, classes
        scores(category) = scores(category) + value * weights(weight_offset + category)
      end do
    end do
  end subroutine kr_fortran_dense_scores_f32
end module kernelyra_matrix_scores
