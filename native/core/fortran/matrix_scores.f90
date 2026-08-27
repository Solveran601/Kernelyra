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

    ! Categories own disjoint score slots, so this loop can use the configured
    ! OpenMP worker pool without a reduction or shared-write race.
    !$omp parallel do if (classes * features >= 4096_c_size_t) schedule(static) private(feature, weight_offset)
    do category = 1_c_size_t, classes
      scores(category) = bias(category)
      do feature = 1_c_size_t, features
        weight_offset = (feature - 1_c_size_t) * classes
        scores(category) = scores(category) + row(feature) * weights(weight_offset + category)
      end do
    end do
    !$omp end parallel do
  end subroutine kr_fortran_dense_scores_f32
end module kernelyra_matrix_scores
