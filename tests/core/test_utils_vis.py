import numpy as np
import torch

from growing_ca.core.utils_vis import (
    SamplePool,
    get_living_mask,
    make_circle_masks,
    make_seed,
    make_seeds,
    to_alpha,
    to_rgb,
)


class TestSamplePool:
    """Test SamplePool class."""

    def test_initialization(self):
        """Test SamplePool initialization."""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        pool = SamplePool(x=x, y=y)
        np.testing.assert_array_equal(pool.x, x)
        np.testing.assert_array_equal(pool.y, y)

    def test_sample(self):
        """Test sampling from pool."""
        x = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        pool = SamplePool(x=x, y=y)
        batch = pool.sample(10)
        assert len(batch.x) == 10
        assert len(batch.y) == 10
        assert batch._parent == pool
        assert batch._parent_idx is not None

    def test_commit(self):
        """Test committing changes back to parent."""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        pool = SamplePool(x=x, y=y)
        batch = pool.sample(2)

        # Modify batch
        batch.x[:] = 0
        batch.y[:] = 99

        # Commit changes
        batch.commit()

        # Check that parent was updated
        assert np.sum(pool.x == 0) >= 2
        assert np.sum(pool.y == 99) >= 2

    def test_initialization_with_parent(self):
        """Test initialization with parent."""
        x = np.array([[1, 2]])
        parent_pool = SamplePool(x=np.array([[3, 4]]))
        child = SamplePool(x=x, _parent=parent_pool, _parent_idx=np.array([0]))
        assert child._parent == parent_pool


class TestToAlpha:
    """Test to_alpha function."""

    def test_to_alpha_basic(self):
        """Test basic alpha extraction."""
        x = np.random.rand(10, 10, 5)
        alpha = to_alpha(x)
        assert alpha.shape == (10, 10, 1)
        assert np.all(alpha >= 0)
        assert np.all(alpha < 1.0)

    def test_to_alpha_clipping(self):
        """Test alpha clipping."""
        x = np.ones((5, 5, 5)) * 2.0  # Values > 1
        alpha = to_alpha(x)
        assert np.all(alpha < 1.0)
        np.testing.assert_array_almost_equal(alpha, np.ones((5, 5, 1)) * 0.9999)

    def test_to_alpha_negative(self):
        """Test alpha with negative values."""
        x = np.ones((5, 5, 5)) * -1.0
        alpha = to_alpha(x)
        assert np.all(alpha >= 0)


class TestToRgb:
    """Test to_rgb function."""

    def test_to_rgb_basic(self):
        """Test basic RGB conversion."""
        x = np.random.rand(10, 10, 5)
        rgb = to_rgb(x)
        assert rgb.shape == (10, 10, 3)
        assert np.all(rgb >= 0)
        assert np.all(rgb < 1.0)

    def test_to_rgb_clipping(self):
        """Test RGB clipping."""
        x = np.ones((5, 5, 5)) * 2.0
        rgb = to_rgb(x)
        assert np.all(rgb < 1.0)

    def test_to_rgb_formula(self):
        """Test RGB formula: 1 - alpha + rgb."""
        x = np.zeros((1, 1, 5))
        x[0, 0, :3] = [0.5, 0.6, 0.7]  # RGB
        x[0, 0, 3] = 0.8  # Alpha
        rgb = to_rgb(x)
        # Should be: 1 - alpha + rgb = 1 - 0.8 + [0.5, 0.6, 0.7]
        expected = np.array([[[0.7, 0.8, 0.9]]])
        np.testing.assert_array_almost_equal(rgb, expected)


class TestGetLivingMask:
    """Test get_living_mask function."""

    def test_living_mask_basic(self):
        """Test basic living mask."""
        x = torch.zeros(1, 10, 5, 5)
        x[:, 3:4, 2, 2] = 0.5  # Set alpha channel
        mask = get_living_mask(x)
        assert mask.shape == (1, 1, 5, 5)
        assert mask.dtype == torch.bool

    def test_living_mask_threshold(self):
        """Test living mask threshold at 0.1."""
        x = torch.zeros(1, 10, 5, 5)
        x[:, 3:4, 0, 0] = 0.05  # Below threshold
        x[:, 3:4, 2, 2] = 0.5  # Above threshold, away from edge
        mask = get_living_mask(x)
        # Center point should be True due to max pooling from nearby cell
        assert mask[0, 0, 2, 2]


class TestMakeSeeds:
    """Test make_seeds function."""

    def test_make_seeds_basic(self):
        """Test basic seed creation."""
        seeds = make_seeds((10, 10), 16, n=5)
        assert seeds.shape == (5, 10, 10, 16)
        assert seeds.dtype == np.float32

    def test_make_seeds_center(self):
        """Test that seed is at center."""
        seeds = make_seeds((10, 10), 16, n=1)
        # Center should have non-zero values in channels 3+
        assert np.all(seeds[0, 5, 5, 3:] == 1.0)
        # Other positions should be zero
        assert np.all(seeds[0, 0, 0, :] == 0.0)

    def test_make_seeds_single(self):
        """Test single seed creation."""
        seeds = make_seeds((8, 8), 12, n=1)
        assert seeds.shape == (1, 8, 8, 12)


class TestMakeSeed:
    """Test make_seed function."""

    def test_make_seed_basic(self):
        """Test basic seed creation."""
        seed = make_seed((10, 10), 16)
        assert seed.shape == (10, 10, 16)
        assert seed.dtype == np.float32

    def test_make_seed_center(self):
        """Test that seed is at center."""
        seed = make_seed((10, 10), 16)
        # Center should have non-zero values in channels 3+
        assert np.all(seed[5, 5, 3:] == 1.0)
        # Other positions should be zero
        assert np.all(seed[0, 0, :] == 0.0)

    def test_make_seed_different_channels(self):
        """Test seed with different channel counts."""
        seed = make_seed((8, 8), 32)
        assert seed.shape == (8, 8, 32)
        assert np.all(seed[4, 4, 3:] == 1.0)


class TestMakeCircleMasks:
    """Test make_circle_masks function."""

    def test_make_circle_masks_basic(self):
        """Test basic circle mask creation."""
        masks = make_circle_masks(5, 10, 10)
        assert masks.shape == (5, 10, 10)
        assert masks.dtype == np.float32

    def test_make_circle_masks_values(self):
        """Test that masks contain 0s and 1s."""
        masks = make_circle_masks(3, 20, 20)
        # Should only contain 0 or 1
        unique_vals = np.unique(masks)
        assert np.all(np.isin(unique_vals, [0.0, 1.0]))

    def test_make_circle_masks_multiple(self):
        """Test multiple circle masks."""
        masks = make_circle_masks(10, 15, 15)
        assert masks.shape == (10, 15, 15)
        # Each mask should have some non-zero values (circles)
        for i in range(10):
            assert np.sum(masks[i]) > 0

    def test_make_circle_masks_randomness(self):
        """Test that masks are random."""
        np.random.seed(42)
        masks1 = make_circle_masks(2, 10, 10)
        np.random.seed(43)
        masks2 = make_circle_masks(2, 10, 10)
        # Different seeds should produce different masks
        assert not np.array_equal(masks1, masks2)
