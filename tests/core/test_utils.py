import numpy as np
import pytest

from growing_ca.core.utils import DistanceMode, mat_distance, tup_distance


class TestDistanceMode:
    """Test DistanceMode enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert DistanceMode.EUCLIDEAN == "Euclidean"
        assert DistanceMode.MANHATTAN == "Manhattan"

    def test_enum_membership(self):
        """Test enum membership."""
        assert "Euclidean" in DistanceMode.__members__.values()
        assert "Manhattan" in DistanceMode.__members__.values()


class TestTupDistance:
    """Test tup_distance function."""

    def test_euclidean_distance_default(self):
        """Test Euclidean distance with default mode."""
        result = tup_distance((0.0, 0.0), (3.0, 4.0))
        assert result == 5.0

    def test_euclidean_distance_explicit(self):
        """Test Euclidean distance with explicit mode."""
        result = tup_distance((0.0, 0.0), (3.0, 4.0), DistanceMode.EUCLIDEAN)
        assert result == 5.0

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        result = tup_distance((0.0, 0.0), (3.0, 4.0), DistanceMode.MANHATTAN)
        assert result == 7.0

    def test_same_point(self):
        """Test distance between same point."""
        result = tup_distance((5.0, 5.0), (5.0, 5.0))
        assert result == 0.0

    def test_negative_coordinates(self):
        """Test distance with negative coordinates."""
        result = tup_distance((-1.0, -1.0), (2.0, 3.0), DistanceMode.EUCLIDEAN)
        assert result == 5.0

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized distance mode"):
            tup_distance((0.0, 0.0), (1.0, 1.0), "Invalid")  # type: ignore


class TestMatDistance:
    """Test mat_distance function."""

    def test_euclidean_distance_default(self):
        """Test Euclidean distance with default mode."""
        mat1 = np.array([[0.0, 0.0]])
        mat2 = np.array([[3.0, 4.0]])
        result = mat_distance(mat1, mat2)
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_euclidean_distance_explicit(self):
        """Test Euclidean distance with explicit mode."""
        mat1 = np.array([[0.0, 0.0]])
        mat2 = np.array([[3.0, 4.0]])
        result = mat_distance(mat1, mat2, DistanceMode.EUCLIDEAN)
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        mat1 = np.array([[0.0, 0.0]])
        mat2 = np.array([[3.0, 4.0]])
        result = mat_distance(mat1, mat2, DistanceMode.MANHATTAN)
        np.testing.assert_array_almost_equal(result, [7.0])

    def test_multiple_points(self):
        """Test distance calculation for multiple points."""
        mat1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        mat2 = np.array([[3.0, 4.0], [4.0, 5.0]])
        result = mat_distance(mat1, mat2, DistanceMode.EUCLIDEAN)
        expected = [5.0, 5.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_same_arrays(self):
        """Test distance between same arrays."""
        mat1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = mat_distance(mat1, mat2)
        expected = [0.0, 0.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        mat1 = np.array([[0.0, 0.0]])
        mat2 = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="Unrecognized distance mode"):
            mat_distance(mat1, mat2, "Invalid")  # type: ignore
